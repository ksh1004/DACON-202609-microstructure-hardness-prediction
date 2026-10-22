"""고정(frozen) CNN 특징 구성원 시험: NASA MicroNet ResNet-50(MIT, 사무국이 허용 명시) vs ImageNet ResNet-50.
디노이즈+조명보정(σ48) 이미지 → 512 입력 → layer2/3/4 특징맵에 DINO와 같은 풀링(hetero3 + 상 조건부 + 기여량), D4 8보기 평균 → 단계별 표준화 Ridge.
사용: python cnn_frozen.py [--weights weights/micronet/resnet50_imagenet-micronet_weights.pth | --timm resnet50.a1_in1k] --tag <이름>"""
import argparse, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="resnet50"); ap.add_argument("--weights", default=None); ap.add_argument("--timm", default=None)
ap.add_argument("--tag", required=True); ap.add_argument("--size", type=int, default=512); ap.add_argument("--stages", type=int, nargs="+", default=[2, 3, 4])
a = ap.parse_args(); sys.argv = sys.argv[:1]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np, torch, torch.nn.functional as F, timm
from probe_models import pool, pool_phase, dark_map
from train_cnn import d4, load_init_weights
from blend_eval import *
from sklearn.linear_model import RidgeCV

S48 = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"
out = ROOT / "outputs/probe2_views" / S48 / f"cnn_{a.tag}_train.npz"
if not out.exists():
    if a.timm: m = timm.create_model(a.timm, pretrained=True, num_classes=0)
    else:
        m = timm.create_model(a.arch, pretrained=False, num_classes=0); load_init_weights(m, str(ROOT / a.weights))
    m = m.cuda().eval()
    mean = torch.tensor((0.485, 0.456, 0.406), device="cuda").view(1, 3, 1, 1); std = torch.tensor((0.229, 0.224, 0.225), device="cuda").view(1, 3, 1, 1)
    X = torch.from_numpy(np.load(ROOT / f"outputs/denoised/{S48}.npz")["train"]).unsqueeze(1)
    acc = {s: [] for s in a.stages}; t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(X), 16):
            xb = X[i:i + 16].cuda().float().div_(255); part = {s: 0 for s in a.stages}
            for k in range(8):
                xk = d4(xb, k); xi = (F.interpolate(xk, size=(a.size, a.size), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1) - mean) / std
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fs = m.forward_intermediates(xi, indices=a.stages, intermediates_only=True)
                for s, z in zip(a.stages, fs):
                    z = z.float(); g = int(z.shape[-1]); dm = dark_map(xk, g); wd = dm.flatten(1).unsqueeze(1)
                    part[s] = part[s] + torch.cat([pool(z, "hetero3"), pool_phase(z, dm), (z.flatten(2) * wd).mean(2)], 1)
            for s in a.stages: acc[s].append((part[s] / 8).cpu())
    np.savez(out, **{f"s{s}": torch.cat(v).numpy().astype(np.float32) for s, v in acc.items()}); print(f"추출 {time.time() - t0:.0f}s → {out.name}")
Z = np.load(out)
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05))
def design(A, t):
    C = A.shape[1] // 11; A3 = A.reshape(len(A), 11, C); outs, ref = [], None
    for blocks, g in GR:
        B = A3[:, list(blocks)].reshape(len(A), -1).astype(np.float64); mu, sd = B[t].mean(0), B[t].std(0); keep = sd > 1e-10
        B = np.clip((B[:, keep] - mu[keep]) / sd[keep], -6, 6)
        if ref is None: ref, s = keep.sum(), 1.0
        else: s = np.sqrt(g * ref / keep.sum())
        outs.append(B * s)
    return np.hstack(outs)
def cv(stages):
    o = np.zeros(len(y))
    for k in range(5):
        t, v = fold != k, fold == k
        D_ = np.hstack([design(Z[f"s{s}"], t) / np.sqrt(design(Z[f"s{s}"], t).shape[1]) for s in stages])   # 단계별 총분산 동일
        o[v] = RidgeCV(alphas=np.logspace(-6, 3, 60)).fit(D_[t], y[t]).predict(D_[v])
    return o
vo = np.load(ROOT / "analysis/view_oof.npz")["D4_8보기"]; rep("DINOv2-L 8보기 (기준)", vo)
res = {}
for st in [(s,) for s in a.stages] + [tuple(a.stages), tuple(a.stages[1:])]:
    res[st] = cv(st); rep(f"{a.tag} 단계 {st}", res[st], vo); sys.stdout.flush()
best = min(res, key=lambda k: ((res[k] - y) ** 2).mean()); ob = res[best]
print(f"최선 {best} | DINOv2-L과 잔차상관 {np.corrcoef(vo - y, ob - y)[0, 1]:.3f}")
for w in (0.1, 0.2, 0.3): rep(f"예측 블렌드 L {1-w:.1f} + {a.tag} {w:.1f}", (1 - w) * vo + w * ob, vo)
np.save(ROOT / f"analysis/model_oof_cnn_{a.tag}.npy", ob)
