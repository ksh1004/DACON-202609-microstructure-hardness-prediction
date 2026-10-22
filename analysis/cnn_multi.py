"""고정 CNN 구성원 변형: 한 단계 안의 여러 블록 출력(다층, DINO의 4층 구성과 같은 발상)과 확장 합성곱(output_stride=8)으로 stride 8을 유지한 깊은 단계.
모듈 이름으로 훅을 걸어 특징맵 수집 → 같은 풀링(11블록) → 모듈별 설계행렬을 이어 붙여 Ridge. train, σ48, D4 8보기.
사용: python cnn_multi.py <timm 모델> [--os 8] --mods layer2.0 layer2.1 ... --sets "0;3;0,1,2,3" """
import argparse, sys, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--os", type=int, default=None); ap.add_argument("--mods", nargs="+", required=True)
ap.add_argument("--sets", default=None); a = ap.parse_args(); sys.argv = sys.argv[:1]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np, torch, torch.nn.functional as F, timm
from probe_models import pool, pool_phase, dark_map
from train_cnn import d4
from blend_eval import *
from sklearn.linear_model import RidgeCV
S48 = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"; tag = a.model.replace(".", "-") + (f"_os{a.os}" if a.os else "") + "_" + "+".join(a.mods).replace(".", "")
out = ROOT / "outputs/probe2_views" / S48 / f"cnnmulti_{tag}_train.npz"
if not out.exists():
    kw = dict(pretrained=True); 
    if a.os: kw["output_stride"] = a.os
    m = timm.create_model(a.model, **kw).cuda().eval(); cfg = m.pretrained_cfg; named = dict(m.named_modules()); buf = {}
    for n in a.mods: named[n].register_forward_hook(lambda mod, i, o, n=n: buf.__setitem__(n, o))
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)), device="cuda").view(1, 3, 1, 1); std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)), device="cuda").view(1, 3, 1, 1)
    X = torch.from_numpy(np.load(ROOT / f"outputs/denoised/{S48}.npz")["train"]).unsqueeze(1); acc = [[] for _ in a.mods]; t0 = time.time(); shp = None
    with torch.no_grad():
        for i in range(0, len(X), 16):
            xb = X[i:i + 16].cuda().float().div_(255); part = [0] * len(a.mods)
            for k in range(8):
                xk = d4(xb, k); xi = (F.interpolate(xk, size=(512, 512), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1) - mean) / std
                with torch.autocast("cuda", dtype=torch.bfloat16): m.forward_features(xi)
                for j, n in enumerate(a.mods):
                    z = buf[n].float()
                    if shp is None or j >= len(shp): shp = (shp or []) + [tuple(z.shape[1:])]
                    if z.shape[-1] > 64: z = F.adaptive_avg_pool2d(z, 64)
                    g = int(z.shape[-1]); dm = dark_map(xk, g); wd = dm.flatten(1).unsqueeze(1)
                    part[j] = part[j] + torch.cat([pool(z, "hetero3"), pool_phase(z, dm), (z.flatten(2) * wd).mean(2)], 1)
            for j in range(len(a.mods)): acc[j].append((part[j] / 8).cpu())
    np.savez(out, shp=np.array([str(s) for s in shp]), **{f"f{j}": torch.cat(v).numpy().astype(np.float32) for j, v in enumerate(acc)}); print(f"추출 {time.time() - t0:.0f}s", flush=True)
Z = np.load(out, allow_pickle=True)
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
L = np.load(ROOT / "analysis/view_oof.npz")["D4_8보기"]; T = np.load(ROOT / "analysis/model_oof_mso400m_patch14_reg1_tipsv2-webli.npy")
O = dict(np.load(ROOT / "analysis/cnn_stage_oof.npz")); Fn = ("densenet121-tv_in1k_512|/8", "resnet50-a1_in1k_512|/8", "resnet50-a2_in1k_512|/8"); Fc = np.mean([O[q] for q in Fn], 0)
rm = lambda o: np.sqrt(((o - y) ** 2).mean()); cr = lambda p, q: np.corrcoef(p - y, q - y)[0, 1]; b3 = lambda f: .45 * L + .25 * T + .30 * f
print(f"== {tag} ==  모듈 {dict(zip(a.mods, Z['shp']))} | 기준: F 3종 평균 {rm(Fc):.3f}, 3계열 {rm(b3(Fc)):.3f}")
sets = [tuple(int(v) for v in s.split(",")) for s in a.sets.split(";")] if a.sets else [(j,) for j in range(len(a.mods))] + [tuple(range(len(a.mods)))]
for js in sets:
    o = cv(js); f4 = (3 * Fc + o) / 4
    print(f"  {'+'.join(a.mods[j] for j in js):40s} 단독 {rm(o):.3f} | 잔차상관 L {cr(o, L):.3f} F {cr(o, Fc):.3f} | L.45+T.25+이것.30 {rm(b3(o)):.3f} | F에 4번째로 추가 {rm(b3(f4)):.3f}", flush=True)
    np.save(ROOT / f"analysis/cnnmulti_oof_{tag}_{'-'.join(map(str, js))}.npy", o)
