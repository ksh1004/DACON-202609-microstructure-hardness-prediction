"""미세구조 전용 디노이저 (train 이미지만 사용, 정답 경도 미사용).

근거: 이미지 노이즈는 픽셀 독립(인접 상관 ≈0)·가우시안(왜도 ≈0)·밝기 무관 → 저노이즈 train 이미지에 가우시안 노이즈를
더하면 (노이즈 입력 → 저노이즈 정답) 학습 쌍을 무한히 만들 수 있다. TV/NLM과 달리 '곧은 결정립계·구간별 상수 결정립'
구조를 학습하므로 고노이즈(σ≥10) 이미지를 저노이즈 영역(경도 모델 오차 10.5)으로 옮기는 것이 목적.

- 정답 이미지: train 중 웨이블릿 σ 추정 < --clean-max (기본 7). 고노이즈 이미지는 정답으로 쓰지 않음.
  --target-src 지정 시 같은 이미지의 1차 디노이즈 결과를 정답으로 사용(잔여 σ0 제거, 부트스트랩).
- 입력: 정답 + N(0, s²), s ~ U(0, --add-max)/255, 8-bit 양자화. 128 크롭 + D4.
- 모델: 4단 U-Net(잔차 예측, NCHW + bf16). --nonblind면 Haar-MAD 추정 σ 지도를 입력 채널로 추가. 추론: 256 전체 + D4 8방향 평균.
- test 이미지는 추론에만 사용(학습 없음).

실행: python src/train_denoiser.py --iters 6000
산출물: outputs/denoised/unet_<tag>.npz (train/test N×256×256 uint8, probe_models.py --src 로 사용), weights/denoiser_<tag>.pt
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cnn import DATA, OUT, d4, load_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


def haar_sigma(x):
    """이미지별 노이즈 σ 추정 (Haar 대각 세부계수의 MAD). 학습 입력과 실제 이미지에 같은 추정기를 써서 일관성 유지."""
    x = x.float()
    hh = (x[..., 0::2, 0::2] - x[..., 0::2, 1::2] - x[..., 1::2, 0::2] + x[..., 1::2, 1::2]) / 2
    return hh.abs().flatten(1).median(1).values / 0.6745


class UNet(nn.Module):
    def __init__(self, ch=(48, 96, 192, 384), nonblind=False):
        super().__init__()
        self.nonblind = nonblind  # True: 추정 σ 지도를 두 번째 입력 채널로
        self.enc = nn.ModuleList()
        c = 2 if nonblind else 1
        for co in ch:
            self.enc.append(block(c, co))
            c = co
        self.mid = block(ch[-1], ch[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for co in reversed(ch):
            self.up.append(nn.Conv2d(c, co, 1))
            self.dec.append(block(2 * co, co))
            c = co
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):  # x: B,1,H,W in [0,1] → 노이즈 제거 이미지
        skips, h = [], x - 0.5
        if self.nonblind:
            h = torch.cat([h, (10 * haar_sigma(x)).view(-1, 1, 1, 1).expand_as(x)], 1)
        for i, e in enumerate(self.enc):
            h = e(h)
            skips.append(h)
            h = F.max_pool2d(h, 2)
        h = self.mid(h)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            h = up(F.interpolate(h, scale_factor=2, mode="nearest"))
            h = dec(torch.cat([h, s], 1))
        return x - self.out(h)  # 잔차(노이즈) 예측


class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return (x - mu) / (var + 1e-6).sqrt() * self.w.view(1, -1, 1, 1) + self.b.view(1, -1, 1, 1)


class NAFBlock(nn.Module):
    """NAFNet 블록 (Chen et al., ECCV 2022 'Simple Baselines for Image Restoration'): LN → 1×1 → 3×3 depthwise → SimpleGate
    → 단순 채널 어텐션 → 1×1, 잔차 + LN → FFN(SimpleGate), 잔차. 비선형 활성함수 없음."""

    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = LayerNorm2d(c), LayerNorm2d(c)
        self.c1 = nn.Conv2d(c, 2 * c, 1)
        self.dw = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.c3 = nn.Conv2d(c, c, 1)
        self.f1, self.f2 = nn.Conv2d(c, 2 * c, 1), nn.Conv2d(c, c, 1)
        self.beta, self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1)), nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        h = self.dw(self.c1(self.n1(x)))
        a, b = h.chunk(2, 1)
        h = a * b
        h = self.c3(h * self.sca(h))
        y = x + h * self.beta
        a, b = self.f1(self.n2(y)).chunk(2, 1)
        return y + self.f2(a * b) * self.gamma


class NAFNet(nn.Module):
    """잔차 예측 NAFNet U자 구조. enc/mid/dec = 단계별 블록 수, 단계마다 채널 2배 (stride-2 conv 다운, pixelshuffle 업)."""

    def __init__(self, width=32, enc=(1, 1, 2, 4), mid=4, dec=(1, 1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(1, width, 3, padding=1)
        self.encs, self.downs, self.ups, self.decs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = width
        for n in enc:
            self.encs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, stride=2))
            c *= 2
        self.mid = nn.Sequential(*[NAFBlock(c) for _ in range(mid)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.ending = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, x):
        h = self.intro(x - 0.5)
        skips = []
        for e, d in zip(self.encs, self.downs):
            h = e(h)
            skips.append(h)
            h = d(h)
        h = self.mid(h)
        for u, d, s in zip(self.ups, self.decs, reversed(skips)):
            h = d(u(h) + s)
        return x - self.ending(h)


def add_noise(x, add_max, g=None):
    s = torch.rand(len(x), 1, 1, 1, device=x.device, generator=g) * add_max / 255
    y = x + torch.randn(x.shape, device=x.device, generator=g) * s
    return (y.clamp(0, 1) * 255).round() / 255


def gblur(x, sb):
    """이미지별 가우시안 블러 (sb: B개 σ, 픽셀 단위, 0이면 그대로). 분리형 7탭, reflect 패딩."""
    B = len(x)
    t = torch.arange(-3, 4, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t[None] / sb.clamp_min(1e-3)[:, None]) ** 2)
    k = torch.where(sb[:, None] < 1e-2, (t[None] == 0).float(), k)
    k = k / k.sum(1, keepdim=True)
    h = x.transpose(0, 1)  # 1,B,H,W
    h = F.conv2d(F.pad(h, (3, 3, 0, 0), mode="reflect"), k.view(B, 1, 1, 7), groups=B)
    h = F.conv2d(F.pad(h, (0, 0, 3, 3), mode="reflect"), k.view(B, 1, 7, 1), groups=B)
    return h.transpose(0, 1)


def degrade(x, p_low=0.5, g=None, keep=False):
    """실제 생성기 모사(2026-10-06 파워스펙트럼 분석): 고품질 모드 = 노이즈 σ 0~12만 /
    저품질 모드(확률 p_low) = 대비 0.65~0.95배(이미지 평균 기준) → 가우시안 블러 0.6~1.4px → 노이즈 σ 9~22. 8-bit 양자화.
    keep=True: (노이즈 입력, 노이즈만 뺀 대비↓·블러 정답) 쌍 반환 → 블러·대비는 그대로 두고 노이즈만 제거하도록 학습"""
    B = len(x)
    r = lambda: torch.rand(B, device=x.device, generator=g)
    low = r() < p_low
    c = torch.where(low, 0.65 + 0.30 * r(), torch.ones(B, device=x.device))
    sb = torch.where(low, 0.6 + 0.8 * r(), torch.zeros(B, device=x.device))
    sn = torch.where(low, 9 + 13 * r(), 12 * r()) / 255
    m = x.mean((1, 2, 3), keepdim=True)
    y0 = gblur(m + c.view(B, 1, 1, 1) * (x - m), sb)
    y = y0 + torch.randn(x.shape, device=x.device, generator=g) * sn.view(B, 1, 1, 1)
    y = (y.clamp(0, 1) * 255).round() / 255
    return (y, y0.clamp(0, 1)) if keep else y


@torch.no_grad()
def denoise(model, X, bs=16):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].cuda().float().div_(255)
        acc = 0
        for k in range(8):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(d4(xb, k)).float()
            inv = torch.rot90(z, -(k & 3), dims=(-2, -1))  # d4 역변환: 회전 먼저 되돌린 뒤 전치
            if k & 4:
                inv = inv.transpose(-1, -2)
            acc = acc + inv
        out.append((acc / 8).clamp(0, 1).mul(255).round().to(torch.uint8).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-max", type=float, default=7.0, help="정답으로 쓸 이미지의 웨이블릿 σ 상한")
    ap.add_argument("--add-max", type=float, default=22.0, help="추가 노이즈 σ 상한 (0~255 단위)")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["l1", "l2"], default="l1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-src", default=None,
                    help="정답 이미지를 이 캐시(예: 1차 디노이저 출력 npz)에서 가져옴 → σ0 잔여 노이즈 없는 깨끗한 정답 (부트스트랩)")
    ap.add_argument("--width", type=int, default=48, help="U-Net 첫 단 채널 수 (단마다 2배)")
    ap.add_argument("--arch", choices=["unet", "nafnet"], default="unet")
    ap.add_argument("--nonblind", action="store_true", help="추정 노이즈 σ를 입력 채널로 제공")
    ap.add_argument("--realistic", action="store_true",
                    help="실제 2모드 열화(고품질: 노이즈만 / 저품질: 대비↓+블러+노이즈)로 학습 → 디노이즈+디블러+대비 복원")
    ap.add_argument("--keep-blur", action="store_true",
                    help="--realistic 열화에서 정답도 대비↓·블러 상태로 → 노이즈만 제거 (저품질 이미지에 결정립계를 지어내지 않음)")
    ap.add_argument("--init", default=None, help="시작 가중치 (미세조정)")
    ap.add_argument("--target-frac", type=float, default=1.0,
                    help="정답 후보 중 사용할 비율 (암기 효과 진단용: 나머지는 정답에서 제외, 목록은 weights/denoiser_<tag>_targets.csv)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "sample_submission.csv")
    sig = pd.read_csv(OUT / "features/train_features.csv").set_index("ID").loc[train.ID, "q_noise_sigma"].values
    X, Xte = load_split("train", train.ID), load_split("test", test.ID)
    ci = np.where(sig < a.clean_max)[0]
    rng = np.random.default_rng(a.seed)
    rng.shuffle(ci)
    ci = ci[:int(round(len(ci) * a.target_frac))]
    nval = max(4, len(ci) // 10)
    vi, ti = ci[:nval], ci[nval:]
    T = torch.from_numpy(np.load(ROOT / a.target_src)["train"]).unsqueeze(1) if a.target_src else X
    C = T[ti].cuda().float().div_(255)
    V = T[vi].cuda().float().div_(255)
    print(f"정답 이미지 {len(ci)}장 (학습 {len(ti)} / 검증 {len(vi)}), σ<{a.clean_max}", flush=True)
    if a.arch == "nafnet":
        model = NAFNet(width=a.width).cuda()
    else:
        model = UNet(tuple(a.width * 2 ** i for i in range(4)), nonblind=a.nonblind).cuda()
    print(f"모델 {a.arch} 파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    if a.init:
        model.load_state_dict(torch.load(ROOT / a.init, map_location="cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    gv = torch.Generator(device="cuda").manual_seed(123)
    if a.keep_blur:
        Vn, V = degrade(V, g=gv, keep=True)  # 고정 검증 입력·정답 (정답도 블러 상태)
    else:
        Vn = degrade(V, g=gv) if a.realistic else add_noise(V, a.add_max, gv)  # 고정 검증 입력
    t0 = time.time()
    for it in range(a.iters):
        model.train()
        lr = a.lr * min(1.0, (it + 1) / 200) * 0.5 * (1 + math.cos(math.pi * it / a.iters))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, len(C), (a.batch,), device="cuda")
        oy, ox = np.random.randint(0, 256 - a.crop + 1, 2)
        y = C[idx, :, oy:oy + a.crop, ox:ox + a.crop]
        y = d4(y, int(np.random.randint(8)))
        if a.keep_blur:
            x, y = degrade(y, keep=True)
        else:
            x = degrade(y) if a.realistic else add_noise(y, a.add_max)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x).float()
        loss = (p - y).abs().mean() if a.loss == "l1" else F.mse_loss(p, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (it + 1) % 1000 == 0 or it == 0:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pv = model(Vn).float()
            mse_in = F.mse_loss(Vn, V).item()
            mse_out = F.mse_loss(pv, V).item()
            print(f"  it{it + 1} loss {loss.item():.4f} | 검증 PSNR 입력 {10 * math.log10(1 / mse_in):.2f} → 출력 "
                  f"{10 * math.log10(1 / mse_out):.2f} dB ({time.time() - t0:.0f}s)", flush=True)
    tag = a.tag or (f"cm{a.clean_max:g}_am{a.add_max:g}_{a.loss}_i{a.iters}" + (f"_w{a.width}" if a.width != 48 else "")
                    + ("_boot" if a.target_src else "") + (f"_s{a.seed}" if a.seed else "")
                    + (f"_tf{a.target_frac:g}" if a.target_frac < 1 else "") + ("_nb" if a.nonblind else "")
                    + (f"_c{a.crop}" if a.crop != 128 else "") + ("_real" if a.realistic else "")
                    + ("_kb" if a.keep_blur else ""))
    pd.DataFrame({"ID": train.ID.values[ci], "role": ["val"] * len(vi) + ["train"] * len(ti)}).to_csv(
        ROOT / "weights" / f"denoiser_{tag}_targets.csv", index=False)
    (ROOT / "weights").mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / "weights" / f"denoiser_{tag}.pt")
    # σ별 검증 (정답 노이즈 σ0 위에 더한 노이즈 기준)
    for s in (5, 10, 15, 20):
        xn = (V + torch.randn(V.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(s)) * s / 255).clamp(0, 1)
        xn = (xn * 255).round() / 255
        pv = denoise(model, (xn * 255).to(torch.uint8).cpu()).cuda().float() / 255
        print(f"  +σ{s:2d}: PSNR 입력 {10 * math.log10(1 / F.mse_loss(xn, V).item()):.2f} → D4 출력 "
              f"{10 * math.log10(1 / F.mse_loss(pv, V).item()):.2f} dB", flush=True)
    Dtr, Dte = denoise(model, X), denoise(model, Xte)
    p = OUT / "denoised" / f"unet_{tag}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, train=Dtr.squeeze(1).numpy(), test=Dte.squeeze(1).numpy())
    print(f"[unet_{tag}] 저장 {p.relative_to(ROOT)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
