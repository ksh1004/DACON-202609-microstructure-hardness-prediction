"""D4 대칭화 DINO 특징: 각 D4 보기의 토큰 지도를 원래 좌표로 되돌려 평균(z̄)한 뒤 풀링 통계(phase2m 블록 배치 11개)를 계산.
기존: 보기별로 통계(std·영역 불균일성)를 구한 뒤 평균 → 보기 잡음이 통계 안에 남음(het2 신뢰도 0.27).
보기 집합 A=[0,1,4,5](기존 묶음), B=[2,3,6,7](여집합), ALL=8보기 를 한 번에 저장. --shifts 로 격자 이동(원본 px, 반사 패딩) 추가:
이동 보기는 토큰 격자가 달라 지도 평균 대신 '이동별 대칭화 통계'의 평균.
산출물: outputs/probe2_rot/<stem>/sym{A,B,ALL}[_shN]_<split>.npy
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_models import build, pool, pool_phase, dark_map  # noqa: E402
from train_cnn import d4  # noqa: E402


def d4_inv(z, k):
    z = torch.rot90(z, -(k & 3), dims=(-2, -1))
    return z.transpose(-1, -2) if k & 4 else z


def stats(z, dm):
    wd = dm.flatten(1).unsqueeze(1)
    return torch.cat([pool(z, "hetero3"), pool_phase(z, dm), (z.float().flatten(2) * wd).mean(2)], 1)


SHIFTS = [(0, 0), (4, 4), (4, 0), (0, 4), (2, 2), (6, 6), (2, 6), (6, 2)]


@torch.no_grad()
def embed(m, mean, std, X, size, layers, bs, nshift):
    out = {"A": [], "B": [], "ALL": []}
    for i in range(0, len(X), bs):
        x0 = X[i:i + bs].cuda().float().div_(255)
        acc = {k: 0 for k in out}
        for dy, dx in SHIFTS[:nshift]:
            xb = F.pad(x0, (dx, 0, dy, 0), mode="reflect")[..., :x0.shape[-2], :x0.shape[-1]] if (dy or dx) else x0
            S = {"A": None, "B": None}
            for k in range(8):
                xi = d4(xb, k)
                xi = F.interpolate(xi, size=(size, size), mode="bilinear", align_corners=False)
                xi = (xi.repeat(1, 3, 1, 1) - mean) / std
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fs = m.forward_intermediates(xi, indices=layers, intermediates_only=True)
                fs = [d4_inv(z.float(), k) for z in fs]
                key = "A" if k in (0, 1, 4, 5) else "B"
                S[key] = fs if S[key] is None else [a + b for a, b in zip(S[key], fs)]
            g = int(S["A"][0].shape[-1]); dm = dark_map(xb, g)
            acc["A"] = acc["A"] + torch.cat([stats(z / 4, dm) for z in S["A"]], 1)
            acc["B"] = acc["B"] + torch.cat([stats(z / 4, dm) for z in S["B"]], 1)
            acc["ALL"] = acc["ALL"] + torch.cat([stats((a + b) / 8, dm) for a, b in zip(S["A"], S["B"])], 1)
        for k in out:
            out[k].append((acc[k] / nshift).cpu())
    return {k: torch.cat(v).numpy().astype(np.float32) for k, v in out.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--shifts", type=int, default=1)
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
    sfx = "" if a.shifts == 1 else f"_sh{a.shifts}"
    for sp in a.splits:
        if (d / f"symALL{sfx}_{sp}.npy").exists():
            continue
        t0 = time.time()
        E = embed(m, mean, std, torch.from_numpy(z[sp]).unsqueeze(1), a.size, a.layers, a.bs, a.shifts)
        for k, v in E.items():
            np.save(d / f"sym{k}{sfx}_{sp}.npy", v)
        print(f"[{sp}{sfx}] {time.time() - t0:.0f}s", flush=True)
    print("DONE")
