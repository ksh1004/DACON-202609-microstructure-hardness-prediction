"""두 번째 디노이즈 경로: 공개 사전학습 DRUNet(gray; KAIR v1.0, 2021-09 공개, MIT)으로 원본 이미지를 디노이즈.
노이즈 수준을 입력으로 받는 비블라인드 모델이므로 이미지별 추정 σ(outputs/features/*_features.csv의 q_noise_sigma)를 넣고, D4 8방향 결과를 평균한다.
산출물: outputs/denoised/drunet.npz (train/test uint8). 가중치: weights/pretrained_denoiser/drunet_gray.pth"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, torch
from PIL import Image
from train_cnn import d4
from drunet_net import load


def main():
    m = load(ROOT / "weights/pretrained_denoiser/drunet_gray.pth")
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
                acc = sum(d4(m(torch.cat([d4(xb, k), nm], 1)), inv[k]) for k in range(8)) / 8
                res.append((acc.clamp(0, 1) * 255).round().byte().squeeze(1).cpu())
        out[split] = torch.cat(res).numpy()
        print(f"[drunet] {split} {out[split].shape}", flush=True)
    (ROOT / "outputs/denoised").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "outputs/denoised/drunet.npz", **out)


if __name__ == "__main__":
    main()
