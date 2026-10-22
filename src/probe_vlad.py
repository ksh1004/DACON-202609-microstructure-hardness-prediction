"""DINOv2 패치 토큰의 VLAD/BoVW 인코딩 + Ridge (학습은 Ridge뿐, 백본 고정).

1) 코드북: train 이미지(정답 미사용)의 패치 토큰을 층별로 표본 추출 → PCA(d) → k-means(K)
2) 인코딩: 각 이미지·D4 뷰의 토큰을 PCA 투영 → 가장 가까운 군집에 배정 →
   [군집별 비율(K)] + [군집별 잔차 합 VLAD(K×d), intra-norm·power-norm·L2]  → 뷰 평균
   = 이미지 안에 어떤 '패치 유형'(결정립 내부·경계·2상·기공 등)이 얼마나, 어떤 모양으로 있는지
3) 고정 5-Fold Ridge → OOF·test (emb.npz 저장)

예) python src/probe_vlad.py --model vit_large_patch14_dinov2.lvd142m --layers 13 17 21 --K 32 --pca 64
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cnn import DATA, OUT, d4, get_folds, load_split  # noqa: E402

PUB_STD = 16.56


def tokens(m, mean, std, xb, size, layers, k=0):
    xi = F.interpolate(d4(xb, k), size=(size, size), mode="bilinear", align_corners=False)
    xi = (xi.repeat(1, 3, 1, 1) - mean) / std
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fs = m.forward_intermediates(xi, indices=layers, intermediates_only=True)
    return [z.float().flatten(2).transpose(1, 2) for z in fs]  # 층별 (B, N, C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_large_patch14_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--layers", type=int, nargs="+", default=[13, 17, 21])
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--sample", type=int, default=96, help="코드북용 이미지당 토큰 표본 수")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "sample_submission.csv")
    fold = get_folds(train)
    y = train.hardness.values.astype(np.float64)
    X, Xte = load_split("train", train.ID), load_split("test", test.ID)
    m = timm.create_model(a.model, pretrained=True, num_classes=0, img_size=a.size).cuda().eval()
    cfg = m.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
    t0 = time.time()
    rng = np.random.default_rng(0)
    # 1) 코드북 (train만)
    samples = [[] for _ in a.layers]
    with torch.no_grad():
        for i in range(0, len(X), a.bs):
            xb = X[i:i + a.bs].cuda().float().div_(255)
            for li, t in enumerate(tokens(m, mean, std, xb, a.size, a.layers)):
                for b in range(t.shape[0]):
                    idx = rng.choice(t.shape[1], a.sample, replace=False)
                    samples[li].append(t[b, idx].cpu().numpy())
    books = []
    for li in range(len(a.layers)):
        S = np.concatenate(samples[li])
        pca = PCA(a.pca, random_state=0).fit(S)
        km = MiniBatchKMeans(a.K, random_state=0, n_init=3, batch_size=4096).fit(pca.transform(S))
        books.append((torch.tensor(pca.mean_, device="cuda", dtype=torch.float32),
                      torch.tensor(pca.components_.T, device="cuda", dtype=torch.float32),
                      torch.tensor(km.cluster_centers_, device="cuda", dtype=torch.float32)))
    print(f"  codebook ready ({time.time() - t0:.0f}s)", flush=True)

    # 2) 인코딩
    @torch.no_grad()
    def encode(Xs):
        out = []
        for i in range(0, len(Xs), a.bs):
            xb = Xs[i:i + a.bs].cuda().float().div_(255)
            acc = 0
            for k in (0, 1, 4, 5):
                feats = []
                for (mu, comp, cent), t in zip(books, tokens(m, mean, std, xb, a.size, a.layers, k)):
                    p = (t - mu) @ comp  # B,N,d
                    asg = torch.cdist(p, cent.unsqueeze(0).expand(p.shape[0], -1, -1)).argmin(-1)  # B,N
                    oh = F.one_hot(asg, a.K).float()  # B,N,K
                    hist = oh.mean(1)
                    vlad = torch.einsum("bnk,bnd->bkd", oh, p) - oh.sum(1).unsqueeze(-1) * cent.unsqueeze(0)
                    vlad = F.normalize(vlad, dim=-1)  # intra-normalization
                    vlad = vlad.flatten(1)
                    vlad = torch.sign(vlad) * vlad.abs().sqrt()
                    vlad = F.normalize(vlad, dim=-1)
                    feats += [hist, vlad]
                acc = acc + torch.cat(feats, 1)
            out.append((acc / 4).cpu())
        return torch.cat(out).numpy()

    Etr, Ete = encode(X), encode(Xte)
    oof, tp = np.zeros(len(y)), np.zeros(len(test))
    for k in range(5):
        tr, va = fold != k, fold == k
        rg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 7, 60))).fit(Etr[tr], y[tr])
        oof[va] = rg.predict(Etr[va])
        tp += rg.predict(Ete) / 5
    w = np.exp(-0.5 * ((y - y.mean()) / PUB_STD) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
    w /= w.mean()
    tag = f"vlad_{a.model.split('.')[0]}_L{'-'.join(map(str, a.layers))}_K{a.K}_d{a.pca}"
    d = OUT / "probe2" / tag
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "emb.npz", train=Etr.astype(np.float32), test=Ete.astype(np.float32))
    pd.DataFrame({"ID": train.ID, "fold": fold, "hardness": y, "pred": oof}).to_csv(d / "oof.csv", index=False)
    pd.DataFrame({"ID": test.ID, "hardness": tp}).to_csv(d / "test_pred.csv", index=False)
    print(f"[{tag}] dim={Etr.shape[1]} probe OOF RMSE {np.sqrt(np.mean((oof - y) ** 2)):.3f} | "
          f"재가중 {np.sqrt(np.mean(w * (oof - y) ** 2)):.3f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
