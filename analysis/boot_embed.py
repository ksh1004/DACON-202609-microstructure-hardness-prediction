"""이미지 내 블록 부트스트랩 특징: 토큰 지도(32×32)를 4×4 블록(각 8×8 토큰)으로 나눠 복원추출로 재배치한 '같은 조직의 다른 표본' R개의 풀링 통계.
같은 이미지의 표본 간 편차 = 유한 시야 때문에 생기는 표본 잡음의 방향 → Ridge 벌점으로 사용 (라벨·test 미사용).
D4 8보기 평균. 산출물: outputs/probe2_views/<stem>/boot_orig_<split>.npy, boot_r<r>_<split>.npy (float16)"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_models import build, pool, pool_phase, dark_map  # noqa: E402
from train_cnn import d4  # noqa: E402
from pb_embed import fwd  # noqa: E402
from sym_embed import d4_inv  # noqa: E402


def stats(z, dm):
    wd = dm.flatten(1).unsqueeze(1)
    return torch.cat([pool(z, "hetero3"), pool_phase(z, dm), (z.float().flatten(2) * wd).mean(2)], 1)


def reblock(z, idx, nb=4):
    """z: (B,C,H,W), idx: (B, nb*nb) 블록 번호 → 블록을 재배치한 지도"""
    B, C, H, W = z.shape; s = H // nb
    b = z.view(B, C, nb, s, nb, s).permute(0, 2, 4, 1, 3, 5).reshape(B, nb * nb, C, s, s)
    b = torch.gather(b, 1, idx.view(B, nb * nb, 1, 1, 1).expand(-1, -1, C, s, s))
    return b.view(B, nb, nb, C, s, s).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


@torch.no_grad()
def embed(m, mean, std, X, size, layers, bs, R, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    o0, oR = [], [[] for _ in range(R)]
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].cuda().float().div_(255); B = len(xb)
        idx = torch.randint(0, 16, (R, B, 16), generator=g, device="cuda")
        a0, aR = 0, [0] * R
        for k in range(8):
            fs = [d4_inv(z, k) for z in fwd(m, mean, std, d4(xb, k), size, layers)]   # 원래 좌표계의 토큰 지도
            gsz = int(fs[0].shape[-1]); dm = dark_map(xb, gsz)
            a0 = a0 + torch.cat([stats(z, dm) for z in fs], 1)
            for r in range(R):
                dmr = reblock(dm, idx[r])
                aR[r] = aR[r] + torch.cat([stats(reblock(z, idx[r]), dmr) for z in fs], 1)
        o0.append((a0 / 8).cpu())
        for r in range(R): oR[r].append((aR[r] / 8).cpu())
    return torch.cat(o0).numpy().astype(np.float16), [torch.cat(v).numpy().astype(np.float16) for v in oR]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--R", type=int, default=6)
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
        e0, eR = embed(m, mean, std, torch.from_numpy(z[sp]).unsqueeze(1), a.size, a.layers, a.bs, a.R, 0)
        np.save(d / f"boot_orig_{sp}.npy", e0)
        for r, e in enumerate(eR): np.save(d / f"boot_r{r}_{sp}.npy", e)
        print(f"[{sp}] {time.time() - t0:.0f}s", flush=True)
    print("DONE")
