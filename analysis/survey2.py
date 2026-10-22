"""후보 구성원(σ48·8보기 스캔 OOF)을 현재 블렌드(v22a 가중 = 미세조정 CNN 없음, 폭 보정 전)에 더했을 때의 효과.
사용: python survey2.py <model_oof 태그 일부> ..."""
import sys, glob, os; sys.path.insert(0, "analysis")
from blend_eval import *
G = dict(P=oof(E + "mr_v14_pp_sc"), A=grp([E + f"dnpp0.25_ff{f}" for f in FF])[0], D=grp([E + f"mv16_dnppm0.05_ff{f}" for f in FF])[0],
         T=grp([E + f"tips16_dnppm0.05_ff{f}" for f in FF])[0], J=grp([E + f"cnnfj4_ff{f}" for f in FF])[0])
W = dict(P=.1765, A=.1456, D=.2250, T=.2059, J=.2471); base = sum(v * G[k] for k, v in W.items())
wg = lambda s: np.exp(-0.5 * ((y - y.mean()) / s) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2); w166 = wg(16.6)
rm = lambda o, w=None: np.sqrt(np.average((o - y) ** 2, weights=w)); q = np.quantile(y, [.1, .9]); mid = (y >= q[0]) & (y < q[1])
print(f"기준 OOF {rm(base):.3f} pw13 {rm(base, w13):.3f}")
pats = sys.argv[1:] or [""]
for f in sorted(glob.glob("analysis/model_oof_*.npy") + glob.glob("analysis/cnnscan_oof_*.npy")):
    nm = os.path.basename(f)[:-4].replace("model_oof_", "").replace("cnnscan_oof_", "cnn:")
    if not any(p in nm for p in pats): continue
    o = np.load(f)
    if o.shape != y.shape: continue
    line = f"{nm[:44]:44s} 단독 {rm(o):.3f} | 잔차상관 기준 {np.corrcoef(o - y, base - y)[0, 1]:.3f} D {np.corrcoef(o - y, G['D'] - y)[0, 1]:.3f} T {np.corrcoef(o - y, G['T'] - y)[0, 1]:.3f} J {np.corrcoef(o - y, G['J'] - y)[0, 1]:.3f} |"
    for w in (.10, .15, .20):
        b = (1 - w) * base + w * o; d = (b - y) ** 2 - (base - y) ** 2
        line += f" +{w:.2f}: {rm(b) - rm(base):+.3f} (pw13 {rm(b, w13) - rm(base, w13):+.3f}, Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f}, {sum(d[fold == k].mean() < 0 for k in range(5))}/5, 가운데 {d[mid].sum() / 500:+.2f})"
    print(line)
