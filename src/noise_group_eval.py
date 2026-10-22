"""OOF 예측들을 노이즈 구간별 RMSE로 비교하고, 기존 v7 블록과의 결합 효과를 본다 (CPU, 수 초).

예) python src/noise_group_eval.py outputs/probe2/<tag1> outputs/probe2/<tag2> ...
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ss = pd.read_csv(ROOT / "data/sample_submission.csv")
tr = pd.read_csv(ROOT / "data/train.csv")
y = tr.hardness.values
sig = pd.read_csv(ROOT / "outputs/features/train_features.csv").set_index("ID").loc[tr.ID, "q_noise_sigma"].values
w = np.exp(-0.5 * ((y - y.mean()) / 16.56) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
w /= w.mean()


def oof(d):
    return pd.read_csv(Path(d) / "oof.csv").pred.values


E = ROOT / "outputs/emb_reg"
cnn = np.mean([oof(ROOT / f"outputs/cnn/s512_{m}_s{s}") for m in ["convnext_tiny", "effnet_b2", "resnet50"] for s in [42, 43, 44]], 0)
dino = np.mean([oof(E / n) for n in ["dinoL_all12", "dinoL_13-15-17-19", "dinoL_17-19", "dinoB_all12"]], 0)
groups = [("σ<10", sig < 10), ("σ10~14", (sig >= 10) & (sig < 14)), ("σ≥14", sig >= 14), ("전체", sig > -1)]


def row(name, p):
    r = [np.sqrt(np.mean((p[m] - y[m]) ** 2)) for _, m in groups]
    rw = np.sqrt(np.mean(w * (p - y) ** 2))
    print(f"{name[:58]:58s} " + " ".join(f"{v:6.2f}" for v in r) + f" | 재가중 {rw:.3f}")


print(f"{'':58s} " + " ".join(f"{g:>6s}" for g, _ in groups))
row("CNN 9모델", cnn)
row("DINO 4구성 (원본 입력)", dino)
row("v7 = CNN:DINO 1:1", 0.5 * cnn + 0.5 * dino)
for d in sys.argv[1:]:
    p = oof(d)
    row(Path(d).name, p)
    row("  + v7 결합: CNN:DINO:신규 = 1:1:1", (cnn + dino + p) / 3)
    row("  + DINO 대체: CNN:(DINO+신규)/2 = 1:1", 0.5 * cnn + 0.25 * dino + 0.25 * p)
