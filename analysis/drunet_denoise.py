"""공개 사전학습 디노이저 DRUNet(gray, KAIR v1.0 공개 2021-09, MIT)으로 원본 이미지를 디노이즈 — 지금 쓰는 자체 학습 U-Net과 전혀 다른 경로.
노이즈 수준을 입력으로 받는 비블라인드 모델이라 이미지별 추정 σ(q_noise_sigma)를 그대로 넣는다. D4 8방향 평균.
목적: 모든 구성원이 같은 디노이즈 이미지를 보기 때문에 저품질 이미지에서 함께 틀린다 → 다른 디노이저 경로의 구성원이 오차를 분산시키는지 확인.
산출물: outputs/denoised/drunet[_s<배율>].npz (train/test uint8, 기존 캐시와 같은 형식)   사용: python drunet_denoise.py [--scale 1.0]"""
import argparse, sys
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--scale", type=float, default=1.0, help="추정 σ에 곱할 배율"); a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from train_cnn import d4
class Res(nn.Module):
    def __init__(s, c): super().__init__(); s.res = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.ReLU(True), nn.Conv2d(c, c, 3, 1, 1, bias=False))
    def forward(s, x): return x + s.res(x)
class DRUNet(nn.Module):
    def __init__(s, nc=(64, 128, 256, 512), nb=4):
        super().__init__(); s.m_head = nn.Conv2d(2, nc[0], 3, 1, 1, bias=False)
        s.m_down1 = nn.Sequential(*[Res(nc[0]) for _ in range(nb)], nn.Conv2d(nc[0], nc[1], 2, 2, 0, bias=False))
        s.m_down2 = nn.Sequential(*[Res(nc[1]) for _ in range(nb)], nn.Conv2d(nc[1], nc[2], 2, 2, 0, bias=False))
        s.m_down3 = nn.Sequential(*[Res(nc[2]) for _ in range(nb)], nn.Conv2d(nc[2], nc[3], 2, 2, 0, bias=False))
        s.m_body = nn.Sequential(*[Res(nc[3]) for _ in range(nb)])
        s.m_up3 = nn.Sequential(nn.ConvTranspose2d(nc[3], nc[2], 2, 2, 0, bias=False), *[Res(nc[2]) for _ in range(nb)])
        s.m_up2 = nn.Sequential(nn.ConvTranspose2d(nc[2], nc[1], 2, 2, 0, bias=False), *[Res(nc[1]) for _ in range(nb)])
        s.m_up1 = nn.Sequential(nn.ConvTranspose2d(nc[1], nc[0], 2, 2, 0, bias=False), *[Res(nc[0]) for _ in range(nb)])
        s.m_tail = nn.Conv2d(nc[0], 1, 3, 1, 1, bias=False)
    def forward(s, x0):
        x1 = s.m_head(x0); x2 = s.m_down1(x1); x3 = s.m_down2(x2); x4 = s.m_down3(x3); x = s.m_body(x4)
        x = s.m_up3(x + x4); x = s.m_up2(x + x3); x = s.m_up1(x + x2); return s.m_tail(x + x1)
m = DRUNet(); print(m.load_state_dict(torch.load(ROOT / "weights/pretrained_denoiser/drunet_gray.pth", map_location="cpu"), strict=True)); m = m.cuda().eval()
D4INV = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}
def inv(x, k):       # d4(·, k)의 역변환을 수치로 찾지 않고 검증된 방식으로: 8개 중 d4(d4(e,k),j)=e 인 j
    return d4(x, INV[k])
e = torch.arange(16.).view(1, 1, 4, 4); INV = {k: next(j for j in range(8) if torch.equal(d4(d4(e, k), j), e)) for k in range(8)}
out = {}
for split, csv in (("train", "train.csv"), ("test", "sample_submission.csv")):
    ids = pd.read_csv(ROOT / "data" / csv).ID; sg = pd.read_csv(ROOT / f"outputs/features/{split}_features.csv").set_index("ID").loc[ids, "q_noise_sigma"].values * a.scale
    X = torch.from_numpy(np.stack([np.array(Image.open(ROOT / f"data/{split}/{i}.png").convert("L")) for i in ids])).unsqueeze(1).float().div(255)
    res = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            xb = X[i:i + 32].cuda(); nm = torch.tensor(sg[i:i + 32] / 255, dtype=torch.float32, device="cuda").view(-1, 1, 1, 1).expand(-1, 1, 256, 256); acc = 0
            for k in range(8): acc = acc + inv(m(torch.cat([d4(xb, k), nm], 1)), k) / 8
            res.append((acc.clamp(0, 1) * 255).round().byte().squeeze(1).cpu())
    out[split] = torch.cat(res).numpy(); print(split, out[split].shape, "원본과의 RMS차 %.2f (추정 σ 평균 %.2f)" % (np.sqrt(((out[split].astype(float) - X.squeeze(1).numpy() * 255) ** 2).mean()), sg.mean()), flush=True)
name = "drunet" + ("" if a.scale == 1.0 else f"_s{a.scale:g}"); np.savez_compressed(ROOT / f"outputs/denoised/{name}.npz", **out); print("→", name)
