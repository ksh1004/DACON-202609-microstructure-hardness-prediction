"""두 번째 디노이즈 경로: 공개 사전학습 DRUNet(gray; KAIR v1.0, 2021-09 공개, MIT)으로 원본 이미지를 디노이즈.
노이즈 수준을 입력으로 받는 비블라인드 모델이므로 이미지별 추정 σ(outputs/features/*_features.csv의 q_noise_sigma)를 넣고, D4 8방향 결과를 평균한다.
산출물: outputs/denoised/drunet.npz (train/test uint8). 가중치: weights/pretrained_denoiser/drunet_gray.pth
--weights/--tag: 미세조정한 DRUNet 가중치(src/drunet_finetune.py 산출물)로 같은 추론 → outputs/denoised/drunetft_<tag>.npz"""
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, torch
from PIL import Image
from train_cnn import d4
from drunet_net import load


SHIFTS = [(0, 0), (4, 4), (2, 6), (6, 2), (4, 0), (0, 4), (2, 2), (6, 6)]


def shift(x, dy, dx, back=False):
    """반사 패딩 후 잘라 (dy, dx)만큼 이동 (back=True면 되돌림). DRUNet의 8배 축소 격자 위치를 바꿔 결과를 평균하기 위함"""
    if not (dy or dx):
        return x
    H, W = x.shape[-2:]
    if back:
        return torch.nn.functional.pad(x, (0, dx, 0, dy), mode="reflect")[..., dy:dy + H, dx:dx + W]
    return torch.nn.functional.pad(x, (dx, 0, dy, 0), mode="reflect")[..., :H, :W]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", default="weights/pretrained_denoiser/drunet_gray.pth"); ap.add_argument("--tag", default=None)
    ap.add_argument("--shifts", type=int, default=1, help="격자 이동 TTA 개수 (1 = 기존과 동일, D4 8방향만)")
    a = ap.parse_args()
    m = load(ROOT / a.weights)
    e = torch.arange(16.).view(1, 1, 4, 4); inv = {k: next(j for j in range(8) if torch.equal(d4(d4(e, k), j), e)) for k in range(8)}
    out = {}
    for split, csv in (("train", "train.csv"), ("test", "sample_submission.csv")):
        ids = pd.read_csv(ROOT / "data" / csv).ID
        sg = pd.read_csv(ROOT / f"outputs/features/{split}_features.csv").set_index("ID").loc[ids, "q_noise_sigma"].values
        X = torch.from_numpy(np.stack([np.array(Image.open(ROOT / f"data/{split}/{i}.png").convert("L")) for i in ids])).unsqueeze(1).float().div(255)
        res = []
        with torch.no_grad():
            for i in range(0, len(X), 32):
                xb = X[i:i + 32].cuda()
                nm = torch.tensor(sg[i:i + 32] / 255, dtype=torch.float32, device="cuda").view(-1, 1, 1, 1).expand(-1, 1, xb.shape[-2], xb.shape[-1])
                acc = sum(shift(d4(m(torch.cat([d4(shift(xb, dy, dx), k), nm], 1)), inv[k]), dy, dx, back=True)
                          for dy, dx in SHIFTS[:a.shifts] for k in range(8)) / (8 * a.shifts)
                res.append((acc.clamp(0, 1) * 255).round().byte().squeeze(1).cpu())
        out[split] = torch.cat(res).numpy()
        print(f"[drunet] {split} {out[split].shape}", flush=True)
    (ROOT / "outputs/denoised").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "outputs/denoised" / ("drunet.npz" if a.tag is None else f"drunetft_{a.tag}.npz"), **out)


if __name__ == "__main__":
    main()
