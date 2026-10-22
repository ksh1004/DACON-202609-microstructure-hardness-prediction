"""고정 CNN 특징 구성원 탐색: timm features_only로 여러 ImageNet CNN의 단계별 특징맵을 뽑아 DINO와 같은 풀링(hetero3+상풀링+기여량) → Ridge.
디노이즈+조명보정(σ48) train 이미지, D4 8보기 평균. 단계별 단독 OOF와 DINOv2-L(8보기)과의 0.7:0.3 블렌드를 출력.
사용: python cnn_scan.py <timm 모델명> [--size 512] [--idx 1 2 3] [--weights <외부 가중치>]"""
import argparse, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--size", type=int, default=512); ap.add_argument("--idx", type=int, nargs="+", default=None)
ap.add_argument("--weights", default=None); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--random", type=int, default=None, help="무작위 초기화(시드) — 사전학습 없이")
ap.add_argument("--adabn", action="store_true", help="BN 통계를 train 이미지(라벨 미사용)로 재추정한 뒤 고정")
a = ap.parse_args(); sys.argv = sys.argv[:1]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np, torch, torch.nn.functional as F, timm
from probe_models import pool, pool_phase, dark_map
from train_cnn import d4, load_init_weights
from blend_eval import *
from sklearn.linear_model import RidgeCV
S48 = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"
tag = (a.model + (f"_ext-{Path(a.weights).stem}" if a.weights else "") + ("_adabn" if a.adabn else "") + (f"_rand{a.random}" if a.random is not None else "")).replace(".", "-") + f"_{a.size}"
out = ROOT / "outputs/probe2_views" / S48 / f"cnnscan_{tag}_train.npz"
if not out.exists():
    kw = dict(features_only=True, pretrained=not a.weights and a.random is None)
    if a.random is not None: torch.manual_seed(a.random)
    if a.idx: kw["out_indices"] = tuple(a.idx)
    m = timm.create_model(a.model, **kw)
    if a.weights:
        sd = torch.load(ROOT / a.weights, map_location="cpu", weights_only=False); sd = sd.get("state_dict", sd); msd = m.state_dict()
        if sum(k in msd for k in sd) == 0:      # timm features_only는 최상위 'features.' 를 'features_' 로 바꿔 부른다 (DenseNet·DPN의 MicroNet 가중치)
            sd = {(k.replace("features.", "features_", 1) if k.startswith("features.") else k): v for k, v in sd.items()}
        if sum(k in msd for k in sd) > 0:
            ok = {k: v for k, v in sd.items() if k in msd and tuple(v.shape) == tuple(msd[k].shape)}; m.load_state_dict(ok, strict=False)
            print(f"  외부 가중치 {len(ok)}/{len(msd)}개 로드 (미사용 {len(sd) - len(ok)}개)", flush=True)
        else: load_init_weights(m, str(ROOT / a.weights))
    m = m.cuda().eval(); cfg = m.pretrained_cfg
    info = [(i["module"], i["num_chs"], i["reduction"]) for i in m.feature_info.get_dicts()]
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)), device="cuda").view(1, 3, 1, 1); std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)), device="cuda").view(1, 3, 1, 1)
    X = torch.from_numpy(np.load(ROOT / f"outputs/denoised/{S48}.npz")["train"]).unsqueeze(1)
    acc = [[] for _ in info]; t0 = time.time()
    if a.adabn:   # BN 누적 평균 통계를 train 이미지 × D4 8보기로 재추정 (학습 가중치는 그대로)
        bns = [mod for mod in m.modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)]
        for bn in bns: bn.reset_running_stats(); bn.momentum = None; bn.train()
        with torch.no_grad():
            for i in range(0, len(X), a.bs):
                xb = X[i:i + a.bs].cuda().float().div_(255)
                for k in range(8):
                    m((F.interpolate(d4(xb, k), size=(a.size, a.size), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1) - mean) / std)
        m.eval(); print(f"  BN {len(bns)}개 재추정 완료", flush=True)
    with torch.no_grad():
        for i in range(0, len(X), a.bs):
            xb = X[i:i + a.bs].cuda().float().div_(255); part = [0] * len(info)
            for k in range(8):
                xk = d4(xb, k); xi = (F.interpolate(xk, size=(a.size, a.size), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1) - mean) / std
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fs = m(xi)
                for j, z in enumerate(fs):
                    z = z.float()
                    if z.shape[-1] > 64: z = F.adaptive_avg_pool2d(z, 64)        # 너무 조밀한 지도는 64×64로 평균 축소
                    g = int(z.shape[-1]); dm = dark_map(xk, g); wd = dm.flatten(1).unsqueeze(1)
                    part[j] = part[j] + torch.cat([pool(z, "hetero3"), pool_phase(z, dm), (z.flatten(2) * wd).mean(2)], 1)
            for j in range(len(info)): acc[j].append((part[j] / 8).cpu())
    np.savez(out, info=np.array([f"{n}|{c}|{r}" for n, c, r in info]), **{f"f{j}": torch.cat(v).numpy().astype(np.float32) for j, v in enumerate(acc)})
    print(f"[{tag}] 추출 {time.time() - t0:.0f}s, license={cfg.get('license')}", flush=True)
Z = np.load(out, allow_pickle=True); info = [s.split("|") for s in Z["info"]]
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05))
def design(A, t):
    C = A.shape[1] // 11; A3 = A.reshape(len(A), 11, C); outs, ref = [], None
    for blocks, g in GR:
        B = A3[:, list(blocks)].reshape(len(A), -1).astype(np.float64); mu, sd = B[t].mean(0), B[t].std(0); keep = sd > 1e-10
        B = np.clip((B[:, keep] - mu[keep]) / sd[keep], -6, 6)
        if ref is None: ref, s = keep.sum(), 1.0
        else: s = np.sqrt(g * ref / keep.sum())
        outs.append(B * s)
    D_ = np.hstack(outs); return D_ / np.sqrt(D_.shape[1])
def cv(js):
    o = np.zeros(len(y))
    for k in range(5):
        t, v = fold != k, fold == k; D_ = np.hstack([design(Z[f"f{j}"], t) for j in js])
        o[v] = RidgeCV(alphas=np.logspace(-6, 3, 60)).fit(D_[t], y[t]).predict(D_[v])
    return o
vo = np.load(ROOT / "analysis/view_oof.npz")["D4_8보기"]
rm = lambda o: np.sqrt(((o - y) ** 2).mean()); pw = lambda o: np.sqrt(np.average((o - y) ** 2, weights=w13))
print(f"== {tag} ==  (기준 DINOv2-L 8보기: OOF {rm(vo):.3f}, pw13 {pw(vo):.3f})")
best = None
combos = [(j,) for j in range(len(info))] + [(j, j + 1) for j in range(len(info) - 1)]
for js in combos:
    o = cv(js); b = .7 * vo + .3 * o; d = (b - y) ** 2 - (vo - y) ** 2
    nm = "+".join(f"{info[j][0]}(c{info[j][1]},/{info[j][2]})" for j in js)
    print(f"  {nm:44s} 단독 {rm(o):.3f} | 잔차상관 {np.corrcoef(vo - y, o - y)[0, 1]:.3f} | L.7+.3 블렌드 {rm(b):.3f} (pw13 {pw(b):.3f}, Δ² {d.mean():+.2f}±{d.std() / np.sqrt(len(d)):.2f}, 폴드 {sum(d[fold == k].mean() < 0 for k in range(5))}/5)", flush=True)
    if best is None or rm(b) < best[0]: best = (rm(b), js, o)
np.save(ROOT / f"analysis/cnnscan_oof_{tag}.npy", best[2])
