"""디노이저 U-Net(부트스트랩, train 이미지만으로 학습)의 내부 특징을 고정 특징 구성원으로: 디노이즈+조명보정(σ48) 이미지를 넣고
인코더·중간·디코더 특징맵에 DINO와 같은 풀링(hetero3 + 상 조건부 + 기여량), D4 8보기 평균 → Ridge. train만."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np, torch, torch.nn.functional as F
from probe_models import pool, pool_phase, dark_map
from train_cnn import d4
from train_denoiser import UNet
from blend_eval import *
from sklearn.linear_model import RidgeCV
S48 = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"
SRC = sys.argv[1] if len(sys.argv) > 1 else "dn"          # dn = 디노이즈 이미지 입력, raw = 원본 입력
out = ROOT / "outputs/probe2_views" / S48 / f"unetfeat_{SRC}_train.npz"
names = ["enc1@128", "enc2@64", "enc3@32", "mid@16", "dec0@32", "dec1@64", "dec2@128"]
if not out.exists():
    net = UNet().cuda().eval(); net.load_state_dict(torch.load(ROOT / "weights/denoiser_cm7_am22_l1_i10000_boot.pt", map_location="cuda"))
    def feats(x):
        skips, h, fs = [], x - 0.5, []
        for e in net.enc:
            h = e(h); skips.append(h); h = F.max_pool2d(h, 2)
        fs += skips[1:]
        h = net.mid(h); fs.append(h)
        for j, (up, dec, s) in enumerate(zip(net.up, net.dec, reversed(skips))):
            h = up(F.interpolate(h, scale_factor=2, mode="nearest")); h = dec(torch.cat([h, s], 1))
            if j < 3: fs.append(h)
        return fs
    if SRC == "dn": X = torch.from_numpy(np.load(ROOT / f"outputs/denoised/{S48}.npz")["train"]).unsqueeze(1)
    else:
        from train_cnn import load_split, DATA
        X = load_split("train", pd.read_csv(DATA / "train.csv").ID)
    Xd = torch.from_numpy(np.load(ROOT / f"outputs/denoised/{S48}.npz")["train"]).unsqueeze(1)   # 상 지도는 항상 디노이즈 이미지에서
    acc = [[] for _ in names]; t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(X), 16):
            xb = X[i:i + 16].cuda().float().div_(255); xd = Xd[i:i + 16].cuda().float().div_(255); part = [0] * len(names)
            for k in range(8):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fs = feats(d4(xb, k))
                xk = d4(xd, k)
                for j, z in enumerate(fs):
                    z = z.float()
                    if z.shape[-1] > 64: z = F.adaptive_avg_pool2d(z, 64)
                    g = int(z.shape[-1]); dm = dark_map(xk, g); wd = dm.flatten(1).unsqueeze(1)
                    part[j] = part[j] + torch.cat([pool(z, "hetero3"), pool_phase(z, dm), (z.flatten(2) * wd).mean(2)], 1)
            for j in range(len(names)): acc[j].append((part[j] / 8).cpu())
    np.savez(out, **{f"f{j}": torch.cat(v).numpy().astype(np.float32) for j, v in enumerate(acc)}); print(f"추출 {time.time() - t0:.0f}s")
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
    D_ = np.hstack(outs); return D_ / np.sqrt(D_.shape[1])
def cv(js):
    o = np.zeros(len(y))
    for k in range(5):
        t, v = fold != k, fold == k; D_ = np.hstack([design(Z[f"f{j}"], t) for j in js])
        o[v] = RidgeCV(alphas=np.logspace(-6, 3, 60)).fit(D_[t], y[t]).predict(D_[v])
    return o
L = np.load(ROOT / "analysis/view_oof.npz")["D4_8보기"]; T = np.load(ROOT / "analysis/model_oof_mso400m_patch14_reg1_tipsv2-webli.npy")
O = dict(np.load(ROOT / "analysis/cnn_stage_oof.npz")); Fc = np.mean([O[k] for k in ("densenet121-tv_in1k_512|/8", "resnet50-a1_in1k_512|/8", "resnet50-a2_in1k_512|/8")], 0)
base3 = .45 * L + .25 * T + .30 * Fc
rm = lambda o: np.sqrt(((o - y) ** 2).mean()); cr = lambda a, b: np.corrcoef(a - y, b - y)[0, 1]
print(f"== U-Net 특징 (입력: {SRC}) ==  기준 3계열 블렌드 {rm(base3):.3f}")
res = {}
for js in [(j,) for j in range(len(names))] + [(1, 2), (2, 3, 4), (1, 2, 4, 5), (4, 5), tuple(range(len(names)))]:
    o = cv(js); res[js] = o
    print(f"  {'+'.join(names[j] for j in js):40s} 단독 {rm(o):.3f} | 잔차상관 L {cr(o, L):.3f}, 고정CNN {cr(o, Fc):.3f} | 기준 .85 + .15 → {rm(.85 * base3 + .15 * o):.3f}, .75+.25 → {rm(.75 * base3 + .25 * o):.3f}", flush=True)
best = min(res, key=lambda k: rm(.85 * base3 + .15 * res[k])); np.save(ROOT / f"analysis/model_oof_unetfeat_{SRC}.npy", res[best])
rep("기준 3계열", base3); rep(f"기준 .85 + U-Net {best} .15", .85 * base3 + .15 * res[best], base3)
