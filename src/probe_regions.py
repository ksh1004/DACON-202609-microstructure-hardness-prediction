"""영역별 경도 예측 → 영역 예측값의 분포(평균·표준편차·최소·최대·분위수)로 이미지 경도 예측 (DINOv2 고정, Ridge 2단).

1) 특징: 이미지를 g×g 영역으로 나눠 영역마다 DINOv2 토큰의 채널별 평균·표준편차 (층 13·15·17·19, 원본 방향 1뷰)
2) 1단 Ridge: 영역 특징 → 그 이미지의 경도 (영역 = 이미지 정답을 물려받은 표본, 다중 인스턴스)
   - 바깥 폴드 k: 학습 이미지의 영역으로 학습 → 검증·test 이미지의 영역 예측
   - 학습 이미지의 영역 예측은 안쪽 5-Fold OOF (2단 학습용, 누수 방지)
3) 이미지별 영역 예측 요약 [평균, 표준편차, 최소, 최대, 25·75% 분위수]
   - 블록 A (mil): 영역 예측 평균 그대로
   - 블록 B (regstat): 요약 6개 → 2단 Ridge
산출물: outputs/emb_reg/regions_<tag>_{mil,regstat}/{oof.csv, test_pred.csv}, outputs/probe2/regions_<tag>/regstats.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_models import build  # noqa: E402
from train_cnn import DATA, OUT, d4, get_folds, load_split  # noqa: E402

PUB_STD = 16.56
ALPHAS = np.logspace(0, 7, 36)


@torch.no_grad()
def region_feats(m, mean, std, X, size, layers, g, bs, views):
    """(n, 뷰, g*g, 층·[평균,표준편차]·C) float16"""
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].cuda().float().div_(255)
        vs = []
        for k in views:
            xi = F.interpolate(d4(xb, k), size=(size, size), mode="bilinear", align_corners=False)
            xi = (xi.repeat(1, 3, 1, 1) - mean) / std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fs = m.forward_intermediates(xi, indices=layers, intermediates_only=True)
            per = []
            for z in fs:
                z = z.float()
                m1 = F.adaptive_avg_pool2d(z, g).flatten(2)          # B,C,R
                m2 = F.adaptive_avg_pool2d(z * z, g).flatten(2)
                per += [m1, (m2 - m1 * m1).clamp_min(0).sqrt()]
            vs.append(torch.cat(per, 1).transpose(1, 2))             # B,R,F
        out.append(torch.stack(vs, 1).half().cpu())                  # B,V,R,F
    return torch.cat(out).numpy()


def fit_predict(Xtr_img, ytr, Xap_list):
    """영역 Ridge: Xtr_img (n,V,R,F) → 영역마다 이미지 정답. 반환: Xap_list 각각의 영역 예측 (n,V,R)"""
    n, V, R, Fd = Xtr_img.shape
    Xf = Xtr_img.reshape(-1, Fd).astype(np.float32)
    sc = StandardScaler().fit(Xf)
    r = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xf), np.repeat(ytr, V * R))
    return [r.predict(sc.transform(X.reshape(-1, Fd).astype(np.float32))).reshape(len(X), V, R) for X in Xap_list]


def summarize(P):
    """P (n,V,R) → (n,6): 뷰별 영역 예측 요약을 뷰 평균"""
    s = np.stack([P.mean(2), P.std(2), P.min(2), P.max(2),
                  np.quantile(P, 0.25, axis=2), np.quantile(P, 0.75, axis=2)], -1)  # n,V,6
    return s.mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_large_patch14_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--layers", type=int, nargs="+", default=[13, 15, 17, 19])
    ap.add_argument("--g", type=int, default=4)
    ap.add_argument("--views", type=int, nargs="+", default=[0], help="D4 뷰 번호 (0=원본)")
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "sample_submission.csv")
    fold = get_folds(train)
    y = train.hardness.values.astype(np.float64)
    tag = f"{a.model.split('.')[0]}_L{'-'.join(map(str, a.layers))}_g{a.g}_v{''.join(map(str, a.views))}"
    cache = OUT / "probe2" / f"regions_{tag}"
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if (cache / "regfeat.npz").exists():
        z = np.load(cache / "regfeat.npz")
        Rtr, Rte = z["train"], z["test"]
    else:
        X, Xte = load_split("train", train.ID), load_split("test", test.ID)
        m, mean, std = build(a.model, a.size)
        Rtr = region_feats(m, mean, std, X, a.size, a.layers, a.g, a.bs, a.views)
        Rte = region_feats(m, mean, std, Xte, a.size, a.layers, a.g, a.bs, a.views)
        del m
        torch.cuda.empty_cache()
        np.savez(cache / "regfeat.npz", train=Rtr, test=Rte)
    print(f"  region features {Rtr.shape} ({time.time() - t0:.0f}s)", flush=True)

    S_oof = np.zeros((len(y), 6))
    S_te = np.zeros((len(test), 6))
    mil_oof, mil_te, st_oof, st_te = np.zeros(len(y)), np.zeros(len(test)), np.zeros(len(y)), np.zeros(len(test))
    for k in range(5):
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        Pva, Pte = fit_predict(Rtr[tr], y[tr], [Rtr[va], Rte])
        Sva, Ste = summarize(Pva), summarize(Pte)
        # 2단 학습용: 학습 이미지의 영역 예측은 안쪽 폴드 OOF
        Str = np.zeros((len(tr), 6))
        ifold = fold[tr]  # 바깥 폴드 번호를 안쪽 분할로 재사용 (k 제외 4개)
        for j in np.unique(ifold):
            itr, iva = ifold != j, ifold == j
            (Pi,) = fit_predict(Rtr[tr[itr]], y[tr[itr]], [Rtr[tr[iva]]])
            Str[iva] = summarize(Pi)
        st = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(Str, y[tr])
        S_oof[va] = Sva
        S_te += Ste / 5
        mil_oof[va], st_oof[va] = Sva[:, 0], st.predict(Sva)
        mil_te += Ste[:, 0] / 5
        st_te += st.predict(Ste) / 5
        print(f"  fold {k}: mil {np.sqrt(np.mean((Sva[:, 0] - y[va]) ** 2)):.2f} | regstat "
              f"{np.sqrt(np.mean((st.predict(Sva) - y[va]) ** 2)):.2f} | coef {np.round(st.coef_, 2)} ({time.time() - t0:.0f}s)",
              flush=True)
    np.savez(cache / "regstats.npz", train=S_oof, test=S_te, cols=np.array(["mean", "std", "min", "max", "q25", "q75"]))
    w = np.exp(-0.5 * ((y - y.mean()) / PUB_STD) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
    w /= w.mean()
    for name, o, t in [("mil", mil_oof, mil_te), ("regstat", st_oof, st_te)]:
        d = OUT / "emb_reg" / f"regions_{tag}_{name}"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ID": train.ID, "fold": fold, "hardness": y, "pred": o}).to_csv(d / "oof.csv", index=False)
        pd.DataFrame({"ID": test.ID, "hardness": t}).to_csv(d / "test_pred.csv", index=False)
        print(f"[regions_{tag}_{name}] OOF {np.sqrt(np.mean((o - y) ** 2)):.3f} | 재가중 {np.sqrt(np.mean(w * (o - y) ** 2)):.3f}")


if __name__ == "__main__":
    main()
