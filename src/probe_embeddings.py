"""사전학습 백본 고정(frozen) 임베딩 + Ridge 선형 프로브 — 미세조정 전 백본 표현력 빠른 선별.

각 stage 특징맵의 채널별 평균·표준편차(다중 스케일 텍스처 통계)를 D4 8방향 평균으로 추출하고,
고정 폴드(outputs/folds.csv)로 Ridge CV RMSE를 계산한다. 수작업 특징과 결합(hybrid)한 점수도 함께 보고.

실행: python src/probe_embeddings.py --models resnet50.a1_in1k convnext_tiny.fb_in22k_ft_in1k ...
산출물: outputs/embeddings/<model>.npz, outputs/probe_results.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_cnn import DATA, OUT, d4, load_split

DEFAULT = ["resnet18.a1_in1k", "resnet50.a1_in1k", "efficientnet_b0.ra_in1k", "convnext_nano.in12k_ft_in1k",
           "convnext_tiny.fb_in22k_ft_in1k", "regnety_016.tv2_in1k", "vit_small_patch14_dinov2.lvd142m"]


@torch.no_grad()
def embed(model_name: str, X: torch.Tensor, bs=32) -> np.ndarray:
    vit = "vit" in model_name
    kw = {"img_size": 252} if "dinov2" in model_name else ({"img_size": 256} if vit else {})
    if vit:
        m = timm.create_model(model_name, pretrained=True, num_classes=0, **kw).cuda().eval()
    else:
        m = timm.create_model(model_name, pretrained=True, features_only=True, **kw).cuda().eval()
    cfg = m.pretrained_cfg if hasattr(m, "pretrained_cfg") else {}
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)), device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)), device="cuda").view(1, 3, 1, 1)
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].cuda().float().div_(255).repeat(1, 3, 1, 1)
        if "dinov2" in model_name:
            xb = xb[..., 2:254, 2:254]  # patch14 → 252
        acc = 0
        for k in range(8):
            xi = (d4(xb, k) - mean) / std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if vit:
                    tok = m.forward_features(xi).float()  # B, N, C
                    pt = tok[:, m.num_prefix_tokens:]
                    f = torch.cat([tok[:, 0], pt.mean(1), pt.std(1)], 1)
                else:
                    fs = m(xi)
                    f = torch.cat([torch.cat([z.float().mean((2, 3)), z.float().std((2, 3))], 1) for z in fs], 1)
            acc = acc + f
        out.append((acc / 8).cpu())
    del m
    torch.cuda.empty_cache()
    return torch.cat(out).numpy()


def ridge_cv(Xf, y, fold, Xt=None, pca=None):
    oof = np.zeros(len(y))
    tp = np.zeros(len(Xt)) if Xt is not None else None
    for k in range(5):
        a, b = fold != k, fold == k
        steps = [StandardScaler()]
        if pca:
            steps.append(PCA(pca, random_state=0))
        steps.append(RidgeCV(alphas=np.logspace(-1, 5, 40)))
        m = make_pipeline(*steps).fit(Xf[a], y[a])
        oof[b] = m.predict(Xf[b])
        if Xt is not None:
            tp += m.predict(Xt) / 5
    return float(np.sqrt(np.mean((oof - y) ** 2))), oof, tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT)
    a = ap.parse_args()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "sample_submission.csv")
    fold = pd.read_csv(OUT / "folds.csv").fold.values
    y = train.hardness.values
    X = load_split("train", train.ID)
    Xte = load_split("test", test.ID)
    ft = pd.read_csv(OUT / "features" / "train_features.csv")
    fte = pd.read_csv(OUT / "features" / "test_features.csv")
    hc = [c for c in ft.columns if c != "ID" and not c.startswith("nui_")]
    Htr, Hte = ft[hc].values, fte[hc].values
    edir = OUT / "embeddings"
    edir.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "probe_results.csv"
    rows = pd.read_csv(res_path).to_dict("records") if res_path.exists() else []
    for name in a.models:
        t = time.time()
        path = edir / f"{name}.npz"
        if path.exists():
            z = np.load(path)
            Etr, Ete = z["train"], z["test"]
        else:
            Etr, Ete = embed(name, X), embed(name, Xte)
            np.savez_compressed(path, train=Etr, test=Ete)
        sec = time.time() - t
        r_emb, oof_e, tp_e = ridge_cv(Etr, y, fold, Ete)
        r_hyb, oof_h, tp_h = ridge_cv(np.hstack([Etr, Htr]), y, fold, np.hstack([Ete, Hte]))
        rows = [r for r in rows if r["model"] != name] + [
            {"model": name, "dim": Etr.shape[1], "probe_rmse": round(r_emb, 3), "hybrid_rmse": round(r_hyb, 3),
             "extract_sec": round(sec, 1)}]
        pdir = OUT / "probe" / name
        pdir.mkdir(parents=True, exist_ok=True)
        for tag, oof, tp in [("emb", oof_e, tp_e), ("hybrid", oof_h, tp_h)]:
            pd.DataFrame({"ID": train.ID, "fold": fold, "hardness": y, "pred": oof}).to_csv(pdir / f"oof_{tag}.csv", index=False)
            pd.DataFrame({"ID": test.ID, "hardness": tp}).to_csv(pdir / f"test_{tag}.csv", index=False)
        print(f"{name:40s} dim={Etr.shape[1]:5d} probe={r_emb:.3f} hybrid={r_hyb:.3f} ({sec:.0f}s)", flush=True)
    pd.DataFrame(rows).sort_values("hybrid_rmse").to_csv(res_path, index=False)


if __name__ == "__main__":
    main()
