"""위치 편향 제거 실험: DINOv2 토큰 지도에서 '위치에만 의존하는 성분' b_l(pos) = train 이미지×D4 보기 평균 토큰을 빼고 풀링.
(라벨·test 미사용. 미세구조는 통계적으로 균질하므로 내용의 위치별 평균은 상수 → 위치별 평균의 공간 변동 = 모델의 위치 편향)
1단계: b 추정(train, D4 8보기) → 2단계: 보기별 phase2m 특징(편향 제거) 저장 outputs/probe2_views/<stem>/<split>_s<이동>_k<k>_pb.npy
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_models import build, pool, pool_phase, dark_map  # noqa: E402
from train_cnn import d4  # noqa: E402
from view_embed import SHIFTS  # noqa: E402


def fwd(m, mean, std, xk, size, layers):
    xi = F.interpolate(xk, size=(size, size), mode="bilinear", align_corners=False)
    xi = (xi.repeat(1, 3, 1, 1) - mean) / std
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return [z.float() for z in m.forward_intermediates(xi, indices=layers, intermediates_only=True)]


def shifted(xb, k, shift):
    xk = d4(xb, k); dy, dx = shift
    return F.pad(xk, (dx, 0, dy, 0), mode="reflect")[..., :xb.shape[-2], :xb.shape[-1]] if (dy or dx) else xk


@torch.no_grad()
def posbias(m, mean, std, X, size, layers, bs):
    acc, n, var = None, 0, 0.0
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].cuda().float().div_(255)
        for k in range(8):
            fs = fwd(m, mean, std, d4(xb, k), size, layers)
            s = [z.sum(0) for z in fs]
            acc = s if acc is None else [a + b for a, b in zip(acc, s)]
            var = var + torch.stack([z.flatten(2).var(2).mean(1).sum() for z in fs])  # 이미지 내 토큰 분산(채널 평균)의 합
        n += len(xb) * 8
    b = torch.stack([a / n for a in acc])  # L,C,h,w
    tokvar = var / n
    return b, tokvar


@torch.no_grad()
def embed(m, mean, std, X, size, layers, bs, shift, k, bias):
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].cuda().float().div_(255)
        xk = shifted(xb, k, shift)
        fs = fwd(m, mean, std, xk, size, layers)
        if bias is not None:
            fs = [z - bl for z, bl in zip(fs, bias)]
        g = int(fs[0].shape[-1]); dm = dark_map(xk, g); wd = dm.flatten(1).unsqueeze(1)
        out.append(torch.cat([torch.cat([pool(z, "hetero3"), pool_phase(z, dm), (z.flatten(2) * wd).mean(2)], 1) for z in fs], 1).cpu())
    return torch.cat(out).numpy().astype(np.float16)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--shifts", type=int, nargs="+", default=[0])
    ap.add_argument("--model", default="vit_large_patch14_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--layers", type=int, nargs="+", default=[11, 13, 15, 17])
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    z = np.load(a.src)
    d = ROOT / "outputs/probe2_views" / Path(a.src).stem
    d.mkdir(parents=True, exist_ok=True)
    m, mean, std = build(a.model, a.size)
    bp = d / f"posbias_{a.size}.npy"
    t0 = time.time()
    if bp.exists():
        b = torch.from_numpy(np.load(bp)).cuda()
    else:
        b, tokvar = posbias(m, mean, std, torch.from_numpy(z["train"]).unsqueeze(1), a.size, a.layers, a.bs)
        np.save(bp, b.cpu().numpy())
        pv = (b - b.mean((2, 3), keepdim=True)).flatten(2).var(2).mean(1)  # 층별: 위치 편향의 공간 분산 (채널 평균)
        print(f"[posbias] {time.time() - t0:.0f}s | 층별 위치편향 분산 / 이미지 내 토큰 분산 = " + " ".join(f"{float(p / t):.3f}" for p, t in zip(pv, tokvar)), flush=True)
        # 2×2·4×4 영역 평균 수준에서의 편향 크기 (영역 평균의 표준편차, 채널 RMS)
        for g in (2, 4, 8):
            rb = F.adaptive_avg_pool2d(b - b.mean((2, 3), keepdim=True), g).flatten(2).std(2)  # L,C
            print(f"   편향의 {g}×{g} 영역평균 std (채널 RMS, 층별): " + " ".join(f"{float(x):.4f}" for x in rb.pow(2).mean(1).sqrt()), flush=True)
    bias = [bl - bl.mean((1, 2), keepdim=True) for bl in b]  # 전역 평균은 유지 (평균 블록 불변)
    for sp in a.splits:
        X = torch.from_numpy(z[sp]).unsqueeze(1)
        for s in a.shifts:
            for k in range(8):
                p = d / f"{sp}_s{s}_k{k}_pb.npy"
                if not p.exists():
                    np.save(p, embed(m, mean, std, X, a.size, a.layers, a.bs, SHIFTS[s], k, bias))
            print(f"[{sp} shift {s}] {time.time() - t0:.0f}s", flush=True)
    print("DONE")
