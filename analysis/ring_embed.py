"""위치 부류별 토큰 평균(테두리 거리 고리 5개 + 4×4 격자의 D4 궤도 3부류) — D4 8보기 평균. 가설 H1 검증용.
블록(층당 8): 고리 r=0 | 1 | 2-3 | 4-7 | 8-15 (테두리로부터 토큰 거리) | 4×4 격자 모서리 4칸 | 가장자리 8칸 | 중심 4칸
추가(층당 +5): 고리별 토큰 표준편차. 산출물: outputs/probe2_views/<stem>/ring_<split>.npy (float32, 보기 평균)
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_models import build  # noqa: E402
from train_cnn import d4  # noqa: E402
from pb_embed import fwd  # noqa: E402


def masks(h, dev):
    i = torch.arange(h, device=dev)
    r = torch.minimum(torch.minimum(i[:, None], i[None, :]), torch.minimum(h - 1 - i[:, None], h - 1 - i[None, :]))
    rings = [(r == 0), (r == 1), (r >= 2) & (r <= 3), (r >= 4) & (r <= 7), (r >= 8)]
    q = h // 4; ci = (i // q).clamp(max=3); gy, gx = ci[:, None].expand(h, h), ci[None, :].expand(h, h)
    edge_y, edge_x = (gy == 0) | (gy == 3), (gx == 0) | (gx == 3)
    orb = [edge_y & edge_x, edge_y ^ edge_x, ~edge_y & ~edge_x]
    return [m.float().flatten() for m in rings], [m.float().flatten() for m in orb]


@torch.no_grad()
def embed(m, mean, std, X, size, layers, bs):
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].cuda().float().div_(255)
        acc = 0
        for k in range(8):
            fs = fwd(m, mean, std, d4(xb, k), size, layers)
            rings, orb = masks(fs[0].shape[-1], xb.device)
            feats = []
            for z in fs:
                z = z.flatten(2)  # B,C,N
                mus = [(z * w).sum(2) / w.sum() for w in rings + orb]
                sds = [(((z - mu.unsqueeze(2)) ** 2 * w).sum(2) / w.sum()).clamp_min(1e-12).sqrt() for mu, w in zip(mus[:5], rings)]
                feats.append(torch.cat(mus + sds, 1))
            acc = acc + torch.cat(feats, 1)
        out.append((acc / 8).cpu())
    return torch.cat(out).numpy().astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--model", default="vit_large_patch14_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--layers", type=int, nargs="+", default=[11, 13, 15, 17])
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    z = np.load(a.src)
    d = ROOT / "outputs/probe2_views" / Path(a.src).stem
    m, mean, std = build(a.model, a.size)
    for sp in a.splits:
        t0 = time.time()
        np.save(d / f"ring_{sp}.npy", embed(m, mean, std, torch.from_numpy(z[sp]).unsqueeze(1), a.size, a.layers, a.bs))
        print(f"[{sp}] {time.time() - t0:.0f}s", flush=True)
    print("DONE")
