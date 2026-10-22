"""미세조정 CNN(원본 이미지로 학습, 폴드별 가중치)의 얕은 단계 특징을 고정 특징으로: 디노이저에 의존하지 않는 구성원 후보.
폴드 k 모델로 전체 train 특징을 뽑고, 폴드 k를 검증으로 하는 Ridge OOF (모델 k는 폴드 k를 보지 않았으므로 누수 없음).
입력: raw(원본) 또는 dn(디노이즈+조명보정). 풀링은 DINO와 동일(상 지도는 디노이즈 이미지 기준). D4 8보기.
사용: python ft_feat.py <exp> <단계 idx> <raw|dn> [--imagenet]"""
import sys, json, time
from pathlib import Path
EXP, IDX, SRC = sys.argv[1], int(sys.argv[2]), sys.argv[3]; IMNET = "--imagenet" in sys.argv; sys.argv = sys.argv[:1]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np, torch, torch.nn.functional as F
from probe_models import pool, pool_phase, dark_map
from train_cnn import d4, build_model, load_split, DATA, OUT
from cnn_predict import restore_args
from blend_eval import *
from sklearn.linear_model import RidgeCV
S48 = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"
d = OUT / "cnn" / EXP; log = json.loads((d / "log.json").read_text(encoding="utf-8")); a = restore_args(log["args"])
tag = f"{EXP}_i{IDX}_{SRC}" + ("_imnet" if IMNET else "")
out = ROOT / "outputs/probe2_views" / S48 / f"ftfeat_{tag}.npy"
if not out.exists():
    Xd = torch.from_numpy(np.load(ROOT / f"outputs/denoised/{S48}.npz")["train"]).unsqueeze(1)
    X = load_split("train", pd.read_csv(DATA / "train.csv").ID) if SRC == "raw" else Xd
    feats = []; t0 = time.time()
    for k in range(5):
        ck = torch.load(d / "weights" / f"fold{k}.pt", map_location="cpu", weights_only=False)
        if IMNET: a.scratch, a.init_weights = False, None
        else: a.scratch, a.init_weights = True, None
        m = build_model(a, k).cuda().eval()
        if not IMNET: m.load_state_dict(ck["state_dict"])
        mean, std = ck["mean"], ck["std"]
        mean = torch.as_tensor(mean, device="cuda").float(); std = torch.as_tensor(std, device="cuda").float()
        outk = []
        with torch.no_grad():
            for i in range(0, len(X), 16):
                xb = X[i:i + 16].cuda().float().div_(255); xdb = Xd[i:i + 16].cuda().float().div_(255); p = 0
                for v in range(8):
                    xi = (F.interpolate(d4(xb, v), size=(512, 512), mode="bilinear", align_corners=False) - mean) / std
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        z = m.forward_intermediates(xi, indices=[IDX], intermediates_only=True)[0].float()
                    if z.shape[-1] > 64: z = F.adaptive_avg_pool2d(z, 64)
                    g = int(z.shape[-1]); dm = dark_map(d4(xdb, v), g); wd = dm.flatten(1).unsqueeze(1)
                    p = p + torch.cat([pool(z, "hetero3"), pool_phase(z, dm), (z.flatten(2) * wd).mean(2)], 1)
                outk.append((p / 8).cpu())
        feats.append(torch.cat(outk).numpy().astype(np.float32)); del m; torch.cuda.empty_cache()
        if IMNET: feats = feats * 5; break
    np.save(out, np.stack(feats)); print(f"추출 {time.time() - t0:.0f}s {np.stack(feats).shape}")
Fk = np.load(out)   # (5, 500, 11*C)
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
o = np.zeros(len(y))
for k in range(5):
    t, v = fold != k, fold == k; D_ = design(Fk[k], t)
    o[v] = RidgeCV(alphas=np.logspace(-6, 3, 60)).fit(D_[t], y[t]).predict(D_[v])
L = np.load(ROOT / "analysis/view_oof.npz")["D4_8보기"]; T = np.load(ROOT / "analysis/model_oof_mso400m_patch14_reg1_tipsv2-webli.npy")
O = dict(np.load(ROOT / "analysis/cnn_stage_oof.npz")); Fc = np.mean([O[q] for q in ("densenet121-tv_in1k_512|/8", "resnet50-a1_in1k_512|/8", "resnet50-a2_in1k_512|/8")], 0)
base3 = .45 * L + .25 * T + .30 * Fc; cr = lambda p, q: np.corrcoef(p - y, q - y)[0, 1]
cnn_oof = pd.read_csv(d / "oof.csv").set_index("ID").loc[ids, "pred"].values
rep(f"[{tag}] 얕은 단계 고정 특징 + Ridge", o); rep("   같은 모델의 원래 예측(GAP 헤드)", cnn_oof)
print(f"   잔차상관: DINOv2-L {cr(o, L):.3f}, 고정 CNN(ImageNet, 디노이즈 입력) {cr(o, Fc):.3f}, 3계열 블렌드 {cr(o, base3):.3f}, 원래 예측 {cr(o, cnn_oof):.3f}")
rep("기준 3계열", base3)
for w in (0.1, 0.2, 0.3): rep(f"   기준 {1-w:.1f} + 이것 {w:.1f}", (1 - w) * base3 + w * o, base3)
np.save(ROOT / f"analysis/model_oof_ftfeat_{tag}.npy", o)
