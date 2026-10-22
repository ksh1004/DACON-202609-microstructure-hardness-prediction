"""회전 불변 특징 실험: 각도별 phase2m 임베딩(D4 4방향 평균 포함)을 따로 저장 → 각도 집합별 평균을 나중에 평가.
회전은 반사 패딩 + 448 격자로 한 번에 쌍선형 표본화(0°는 기존 추출과 동일). 상 지도는 같은 각도로 돌린 256 이미지에서 계산.
예) python analysis/rot_embed.py --src outputs/denoised/<stem>.npz --angles 0 22.5 45 67.5 --splits train
산출물: outputs/probe2_rot/<stem>/a<각도>_<split>.npy
"""
import argparse, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_models import build, pool, pool_phase, dark_map  # noqa: E402
from train_cnn import d4  # noqa: E402


def rot(x, deg, size):
    B = x.shape[0]
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    A = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], device=x.device).expand(B, 2, 3)
    g = F.affine_grid(A, (B, 1, size, size), align_corners=False)
    return F.grid_sample(x, g, mode="bilinear", padding_mode="reflection", align_corners=False)


@torch.no_grad()
def embed(m, mean, std, X, size, layers, bs, deg):
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].cuda().float().div_(255)
        acc = 0
        for k in (0, 1, 4, 5):
            xk = d4(xb, k)
            xi = rot(xk, deg, size)
            xl = rot(xk, deg, xk.shape[-1]) if deg else xk
            xi = (xi.repeat(1, 3, 1, 1) - mean) / std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fs = m.forward_intermediates(xi, indices=layers, intermediates_only=True)
            g = int(fs[0].shape[-1]); dm = dark_map(xl, g); wd = dm.flatten(1).unsqueeze(1)
            acc = acc + torch.cat([torch.cat([pool(z, "hetero3"), pool_phase(z, dm), (z.float().flatten(2) * wd).mean(2)], 1) for z in fs], 1)
        out.append((acc / 4).cpu())
    return torch.cat(out).numpy().astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--angles", type=float, nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--model", default="vit_large_patch14_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--layers", type=int, nargs="+", default=[9, 11, 13, 15, 17, 19, 21])
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    z = np.load(a.src)
    d = ROOT / "outputs/probe2_rot" / Path(a.src).stem
    d.mkdir(parents=True, exist_ok=True)
    m, mean, std = build(a.model, a.size)
    for sp in a.splits:
        X = torch.from_numpy(z[sp]).unsqueeze(1)
        for deg in a.angles:
            p = d / f"a{deg:g}_{sp}.npy"
            if p.exists():
                continue
            t0 = time.time()
            np.save(p, embed(m, mean, std, X, a.size, a.layers, a.bs, deg))
            print(f"[{sp} a{deg:g}] {time.time() - t0:.0f}s", flush=True)
    print("DONE")
