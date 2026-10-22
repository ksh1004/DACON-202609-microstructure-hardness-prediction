"""디노이즈 경로 증강 Ridge: '노이즈 추가 → 디노이저 → DINO 특징' 사본을 같은 정답으로 Ridge 학습에 추가.

목적: 고노이즈(σ≥10) test 이미지가 실제로 거치는 경로(강한 노이즈 → 디노이저 복원)를 train에서 재현해,
Ridge가 디노이저 복원 오차에 강건한 특징 방향을 쓰도록 한다. (원본 특징에 노이즈만 섞던 probe_aug.py와 다름)
- 사본 원본: 디노이저 학습 정답으로 쓰지 않은 저·중노이즈 이미지(σ < --src-max, 디노이저 정답 목록 제외) → 암기 효과 방지
- 추가 노이즈: 목표 총 σ ~ U(--lo, --hi), 추가분 = sqrt(목표² − 원래 σ²) (원래 σ = Haar-MAD 추정), 8-bit 양자화
- 누수 방지: 폴드 k Ridge는 폴드 k 학습 이미지의 원본 + 그 사본만 사용. 검증·test는 원본 디노이즈 특징.

예) python src/probe_aug_dn.py --denoiser weights/denoiser_cm7_am22_l1_i10000_boot.pt --feat outputs/probe2/<boot 프로브 폴더> --copies 3
산출물: outputs/emb_reg/augdn_<tag>/{oof.csv, test_pred.csv}
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_models import build, embed  # noqa: E402
from train_cnn import DATA, OUT, get_folds, load_split  # noqa: E402
from train_denoiser import UNet, degrade, denoise, haar_sigma  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PUB_STD = 16.56


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--denoiser", required=True, help="weights/denoiser_<tag>.pt")
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--feat", required=True, help="같은 디노이저 입력의 원본 프로브 폴더 (emb.npz, 층 9~21 hetero3)")
    ap.add_argument("--use-layers", type=int, nargs="+", default=[1, 2, 3, 4], help="emb 안 층 위치 (기본 L11-17)")
    ap.add_argument("--src-max", type=float, default=10.0)
    ap.add_argument("--lo", type=float, default=10.0)
    ap.add_argument("--hi", type=float, default=21.0)
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--copy-weight", type=float, default=1.0)
    ap.add_argument("--realistic", action="store_true",
                    help="사본을 실제 저품질 모드로 열화(대비 0.65~0.95 → 블러 0.6~1.4px → 노이즈 σ9~22, train_denoiser.degrade)")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "sample_submission.csv")
    fold = get_folds(train)
    y = train.hardness.values.astype(np.float64)
    sig = pd.read_csv(OUT / "features/train_features.csv").set_index("ID").loc[train.ID, "q_noise_sigma"].values
    tgt = ROOT / (str(a.denoiser)[:-3] + "_targets.csv")
    seen = set(pd.read_csv(tgt).ID) if tgt.exists() else set()
    if not seen:  # 목록 저장 전 학습된 디노이저: train_denoiser.py와 같은 규칙으로 재현 (σ<7, 시드 0)
        ci = np.where(sig < 7.0)[0]
        np.random.default_rng(0).shuffle(ci)
        seen = set(train.ID.values[ci[max(4, len(ci) // 10):]])
    src = np.where((sig < a.src_max) & ~train.ID.isin(seen).values)[0]
    print(f"사본 원본 {len(src)}장 (σ<{a.src_max}, 디노이저 정답 {len(seen)}장 제외) × {a.copies}", flush=True)
    t0 = time.time()
    X = load_split("train", train.ID)
    model = UNet(tuple(a.width * 2 ** i for i in range(4))).cuda()
    model.load_state_dict(torch.load(ROOT / a.denoiser, map_location="cuda"))
    m, mean, std = build("vit_large_patch14_dinov2.lvd142m", 448)
    z = np.load(ROOT / a.feat / "emb.npz")
    nl = z["train"].shape[1] // (6 * 1024)
    R = lambda E: E.reshape(len(E), nl, 6, 1024)[:, a.use_layers].reshape(len(E), -1).astype(np.float64)
    E0, Ete = R(z["train"]), R(z["test"])
    layers = [9 + 2 * i for i in range(nl)]
    layers = [layers[i] for i in a.use_layers]
    Ec, owner = [], []
    for c in range(a.copies):
        g = torch.Generator().manual_seed(2000 + c)
        x = X[src].float() / 255
        if a.realistic:
            xn = (degrade(x.cuda(), p_low=1.0, g=torch.Generator(device="cuda").manual_seed(2000 + c)) * 255).round().to(torch.uint8).cpu()
        else:
            s0 = haar_sigma(x).view(-1, 1, 1, 1)
            target = (a.lo + (a.hi - a.lo) * torch.rand(len(x), 1, 1, 1, generator=g)) / 255
            add = (target ** 2 - s0 ** 2).clamp_min(0).sqrt()
            xn = ((x + torch.randn(x.shape, generator=g) * add).clamp(0, 1) * 255).round().to(torch.uint8)
        xd = denoise(model, xn)
        Ec.append(embed(m, mean, std, xd, 448, 4, layers, 8, "hetero3").astype(np.float64))
        owner.append(src)
        print(f"  사본 {c + 1}/{a.copies} 특징 완료 ({time.time() - t0:.0f}s)", flush=True)
    Ec, owner = np.vstack(Ec), np.concatenate(owner)
    oof, oof0, tp = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(test))
    copy0 = np.zeros(len(owner))
    for k in range(5):
        tr, va = fold != k, fold == k
        ck = tr[owner]
        Xa = np.vstack([E0[tr], Ec[ck]])
        ya = np.concatenate([y[tr], y[owner[ck]]])
        wa = np.concatenate([np.ones(tr.sum()), np.full(ck.sum(), a.copy_weight)])
        sc = StandardScaler().fit(E0[tr])
        rg = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(sc.transform(Xa), ya, sample_weight=wa)
        oof[va] = rg.predict(sc.transform(E0[va]))
        tp += rg.predict(sc.transform(Ete)) / 5
        r0 = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(sc.transform(E0[tr]), y[tr])
        oof0[va] = r0.predict(sc.transform(E0[va]))
        mv = va[owner]
        copy0[mv] = r0.predict(sc.transform(Ec[mv]))  # 증강 없는 Ridge로 '검증 폴드 이미지의 열화 사본' 예측 → 정보 손실 크기
    w = np.exp(-0.5 * ((y - y.mean()) / PUB_STD) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
    w /= w.mean()

    def rep(p):
        g = [np.sqrt(np.mean((p[mm] - y[mm]) ** 2)) for mm in (sig < 10, (sig >= 10) & (sig < 14), sig >= 14)]
        return (f"σ<10 {g[0]:.2f} | σ10~14 {g[1]:.2f} | σ≥14 {g[2]:.2f} | 전체 {np.sqrt(np.mean((p - y) ** 2)):.3f} "
                f"| 재가중 {np.sqrt(np.mean(w * (p - y) ** 2)):.3f}")

    tag = (f"augdn_{Path(a.denoiser).stem.replace('denoiser_', '')}_c{a.copies}" + ("_real" if a.realistic else f"_n{a.lo:g}-{a.hi:g}")
           + f"{'_w' + str(a.copy_weight) if a.copy_weight != 1 else ''}")
    d = OUT / "emb_reg" / tag
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": train.ID, "fold": fold, "hardness": y, "pred": oof}).to_csv(d / "oof.csv", index=False)
    pd.DataFrame({"ID": test.ID, "hardness": tp}).to_csv(d / "test_pred.csv", index=False)
    np.savez_compressed(d / "aug_emb.npz", emb=Ec.astype(np.float32), owner=owner)
    print(f"  증강 없음: {rep(oof0)}")
    print(f"  [정보 손실] 사본 원본 이미지 오차 {np.sqrt(np.mean((oof0[src] - y[src]) ** 2)):.2f} → 열화 사본 오차 "
          f"{np.sqrt(np.mean((copy0 - y[owner]) ** 2)):.2f} (원본 예측과 차 RMS {np.sqrt(np.mean((copy0 - np.tile(oof0[src], a.copies)) ** 2)):.2f})")
    print(f"[{tag}] 증강: {rep(oof)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
