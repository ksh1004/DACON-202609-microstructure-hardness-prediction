"""보기별(D4 8방향 × 격자 이동) phase2m 풀링 특징을 따로 저장 → 임의 보기 부분집합 평균을 나중에 평가.
층은 사용하는 4개(11,13,15,17)만. 산출물: outputs/probe2_views/<stem>/<split>_s<이동>_k<D4>.npy (float16)
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_models import build, pool, pool_phase, dark_map  # noqa: E402
from train_cnn import d4  # noqa: E402

SHIFTS = [(0, 0), (4, 4), (4, 0), (0, 4), (2, 2), (6, 6), (2, 6), (6, 2)]


@torch.no_grad()
def embed(m, mean, std, X, size, layers, bs, shift, k):
    dy, dx = shift
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].cuda().float().div_(255)
        xk = d4(xb, k)
        if dy or dx:
            xk = F.pad(xk, (dx, 0, dy, 0), mode="reflect")[..., :xb.shape[-2], :xb.shape[-1]]
        xi = F.interpolate(xk, size=(size, size), mode="bilinear", align_corners=False)
        xi = (xi.repeat(1, 3, 1, 1) - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fs = m.forward_intermediates(xi, indices=layers, intermediates_only=True)
        g = int(fs[0].shape[-1]); dm = dark_map(xk, g); wd = dm.flatten(1).unsqueeze(1)
        out.append(torch.cat([torch.cat([pool(z, "hetero3"), pool_phase(z, dm), (z.float().flatten(2) * wd).mean(2)], 1) for z in fs], 1).cpu())
    return torch.cat(out).numpy().astype(np.float16)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--shifts", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--model", default="vit_large_patch14_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--layers", type=int, nargs="+", default=[11, 13, 15, 17])
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--suffix", default="", help="파일명 접미사 (같은 모델의 다른 층 구성을 구분)")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    z = np.load(a.src)
    d = ROOT / "outputs/probe2_views" / Path(a.src).stem
    d.mkdir(parents=True, exist_ok=True)
    m, mean, std = build(a.model, a.size)
    t0 = time.time()
    for sp in a.splits:
        X = torch.from_numpy(z[sp]).unsqueeze(1)
        for s in a.shifts:
            for k in range(8):
                mt = "" if a.model == "vit_large_patch14_dinov2.lvd142m" else ("_mbasep14" if a.model == "vit_base_patch14_dinov2.lvd142m" else "_m" + a.model.replace("hf:", "").replace(":", "_").replace("/", "__").replace("vit_", "").replace(".", "-"))
                p = d / f"{sp}_s{s}_k{k}{'' if a.size == 448 else f'_r{a.size}'}{mt}{a.suffix}.npy"
                if not p.exists():
                    np.save(p, embed(m, mean, std, X, a.size, a.layers, a.bs, SHIFTS[s], k))
            print(f"[{sp} shift {s} {SHIFTS[s]}] {time.time() - t0:.0f}s", flush=True)
    print("DONE")
