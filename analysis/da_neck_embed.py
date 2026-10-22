"""Depth-Anything-V2-L의 깊이 디코더(DPT neck) 다중 해상도 특징 + 예측 깊이 지도 통계 (고정, 학습 없음).

neck 출력 4개(256채널, 448 입력 기준 32·64·128·256 격자)마다 hetero3 + 상 조건부 풀링 + 혼합 기여량(view_embed와 같은 11블록)을 계산하고,
예측 깊이 지도(448²)는 이미지별 표준화 후 분위수·영역 불균일성·2상/기지 차이·기울기 크기·밝기와의 상관을 요약한다.
D4 8방향 평균 (TTA). 산출물: outputs/probe2/da_neck_<stem>/emb.npz (train/test, 층 4개 × 11블록 × 256),
outputs/features_cache/dadepth_<stem>_{train,test}.csv (열 접두어 dd_)
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from probe_models import pool, pool_phase, dark_map  # noqa: E402
from train_cnn import d4  # noqa: E402
from transformers import AutoModelForDepthEstimation  # noqa: E402


def depth_stats(d, xk):
    """d (B,H,W) 예측 깊이, xk (B,1,256,256) 입력 이미지(0~1) → (B, n) 통계"""
    B = d.shape[0]
    raw_m, raw_s = d.flatten(1).mean(1), d.flatten(1).std(1) + 1e-6
    z = (d - raw_m.view(B, 1, 1)) / raw_s.view(B, 1, 1)
    q = torch.quantile(z.flatten(1), torch.tensor([.05, .1, .25, .5, .75, .9, .95], device=d.device), dim=1).T
    zz = z.unsqueeze(1)
    het = [F.adaptive_avg_pool2d(zz, g).flatten(1).std(1) for g in (2, 4, 8, 16)]
    gy, gx = zz[..., 1:, :] - zz[..., :-1, :], zz[..., :, 1:] - zz[..., :, :-1]
    grad = torch.stack([gy.abs().mean((1, 2, 3)), gx.abs().mean((1, 2, 3))], 1).mean(1)
    dm = dark_map(xk, z.shape[-1]).float()                                        # 2상 비율 지도 (z 격자)
    w = dm.flatten(1); zf = z.flatten(1)
    dmean = (zf * w).sum(1) / (w.sum(1) + 1e-6); mmean = (zf * (1 - w)).sum(1) / ((1 - w).sum(1) + 1e-6)
    xi = F.interpolate(xk, size=z.shape[-2:], mode="bilinear", align_corners=False).flatten(1)
    xi = (xi - xi.mean(1, keepdim=True)) / (xi.std(1, keepdim=True) + 1e-6)
    corr = (xi * zf).mean(1)
    return torch.cat([(raw_s / raw_m.abs().clamp_min(1e-6)).unsqueeze(1), q, torch.stack(het, 1), grad.unsqueeze(1),
                      (dmean - mmean).unsqueeze(1), corr.unsqueeze(1)], 1)


@torch.no_grad()
def run(m, X, size, bs, mean, std, patch=14, shifts=1):
    embs, dsts = [], []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].cuda().float().div_(255)
        acc_e, acc_d = 0, 0
        sh = [(0, 0), (4, 0), (0, 4), (4, 4)][:shifts]   # probe_models.embed와 같은 격자 이동 TTA
        for k, (dy, dx) in [(k, s_) for k in range(8) for s_ in sh]:
            xk = d4(xb, k)
            if dy or dx:
                xk = F.pad(xk, (dx, 0, dy, 0), mode="reflect")[..., :xb.shape[-2], :xb.shape[-1]]
            xi = F.interpolate(xk, size=(size, size), mode="bilinear", align_corners=False)
            xi = (xi.repeat(1, 3, 1, 1) - mean) / std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hs = m.backbone.forward_with_filtered_kwargs(xi, output_hidden_states=None, output_attentions=None).feature_maps
                ph = pw = size // patch
                nk = m.neck(hs, ph, pw)
                try:
                    dep = m.head(nk, ph, pw)
                except TypeError:   # DPT 헤드는 격자 크기를 받지 않음
                    dep = m.head(nk)
            feats = []
            for z in nk:
                z = z.float(); g = int(z.shape[-1]); dm = dark_map(xk, g); wd = dm.flatten(1).unsqueeze(1)
                feats.append(torch.cat([pool(z, "hetero3"), pool_phase(z, dm), (z.flatten(2) * wd).mean(2)], 1))
            acc_e = acc_e + torch.cat(feats, 1)
            acc_d = acc_d + depth_stats(dep.float(), xk)
        embs.append((acc_e / (8 * len(sh))).cpu()); dsts.append((acc_d / (8 * len(sh))).cpu())
    return torch.cat(embs).numpy().astype(np.float32), torch.cat(dsts).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--size", type=int, default=448); ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--repo", default="depth-anything/Depth-Anything-V2-Large-hf", help="깊이 모델 (예: facebook/dpt-dinov2-large-nyu, Apache-2.0)")
    ap.add_argument("--prefix", default="da", help="산출물 이름 접두어 (da → da_neck_*, dadepth_*)")
    ap.add_argument("--shifts", type=int, default=1, choices=[1, 2, 4], help="격자 이동 TTA 수 (2 → D4 8 × 이동 2 = 16보기)")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    m = AutoModelForDepthEstimation.from_pretrained(a.repo).cuda().eval()
    mean = torch.tensor((0.485, 0.456, 0.406), device="cuda").view(1, 3, 1, 1); std = torch.tensor((0.229, 0.224, 0.225), device="cuda").view(1, 3, 1, 1)
    z = np.load(ROOT / a.src); stem = Path(a.src).stem; t0 = time.time()
    out_e, out_d = {}, {}
    for split, csv in (("train", "train.csv"), ("test", "sample_submission.csv")):
        X = torch.from_numpy(z[split]).unsqueeze(1)
        out_e[split], out_d[split] = run(m, X, a.size, a.bs, mean, std, m.config.backbone_config.patch_size, a.shifts)
        ids = pd.read_csv(ROOT / "data" / csv).ID
        cols = ["dd_relspread"] + [f"dd_q{p}" for p in (5, 10, 25, 50, 75, 90, 95)] + [f"dd_het{g}" for g in (2, 4, 8, 16)] + ["dd_grad", "dd_phase_diff", "dd_corr_int"]
        df = pd.DataFrame(out_d[split], columns=cols); df.insert(0, "ID", ids.values)
        df.to_csv(ROOT / "outputs/features_cache" / f"{a.prefix}depth_{stem}{'' if a.size == 448 else f'_r{a.size}'}{'' if a.shifts == 1 else f'_sh{a.shifts}'}_{split}.csv", index=False)
        print(f"[da_neck] {split} emb {out_e[split].shape} depth {out_d[split].shape} ({time.time() - t0:.0f}s)", flush=True)
    d = ROOT / "outputs/probe2" / f"{a.prefix}_neck_{stem}{'' if a.size == 448 else f'_r{a.size}'}{'' if a.shifts == 1 else f'_sh{a.shifts}'}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "emb.npz", train=out_e["train"], test=out_e["test"])
    print("DONE", d)


if __name__ == "__main__":
    main()
