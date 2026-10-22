"""v20 후보: 저품질 특징 배율(모드 분리) 적용 블렌드. 그룹 접미사(_m1.5 등)를 계열별로 바꿔 가며 v19a와 비교.
지표: OOF, Public 꼴(라벨 std 16.6 재가중 — 채점된 6개 제출의 수준과 일치), Private 꼴(std 18), 모드별, 기울기.
사용: python b24.py [접미사=_m1.5] [--save]"""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
SUF = next((a for a in sys.argv[1:] if not a.startswith("--")), "_m1.5"); Lq = sig >= 10; c = 194.16551625321193
lqt = pd.read_csv("outputs/features/test_features.csv").set_index("ID").loc[te, "q_noise_sigma"].values >= 10
wg = lambda s: np.exp(-0.5 * ((y - y.mean()) / s) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2); w166, w18 = wg(16.6), wg(18.0)
def G(s=""):
    co, ct = grp(CNN); a = grp([E + f"dnpp0.25_ff{f}{s}" for f in FF]); d = grp([E + f"mv16_dnppm0.05_ff{f}{s}" for f in FF]); t = grp([E + f"tips16_dnppm0.05_ff{f}{s}" for f in FF])
    f_ = grp([E + f"cnnf_{n}_ff{f}{s}" for n in ("dn121", "r50a1", "r50a2") for f in FF]); return dict(C=(co, ct), A=a, D=d, T=t, F=f_)
B, N = G(""), G(SUF); P0 = (oof(E + "mr_v14_pp_sc"), tp(E + "mr_v14_pp_sc"))
Pm = {k: (oof(E + f"mr_v14_pp_sc{k}"), tp(E + f"mr_v14_pp_sc{k}")) for k in (SUF, SUF.replace("_m", "_mc")) if (pd.io.common.file_exists(E + f"mr_v14_pp_sc{k}/oof.csv"))}
W = dict(C=.15, P=.15, A=.12375, D=.19125, T=.175, F=.21)
def blend(g, p, w=W, i=0): return sum(w[k] * g[k][i] for k in "CADTF") + w["P"] * p[i]
def sl(p, m): cv_ = np.cov(y[m], p[m]); return cv_[0, 1] / cv_[1, 1]
rm = lambda p, m=None, w=None: np.sqrt(np.average(((p - y) ** 2) if m is None else ((p - y)[m] ** 2), weights=w))
base = blend(B, P0); bt = blend(B, P0, i=1)
def show(n, o, t):
    d = (o - y) ** 2 - (base - y) ** 2
    print(f"{n:38s} OOF {rm(o):.3f} | Public꼴 {rm(o, w=w166):.3f} Private꼴 {rm(o, w=w18):.3f} pw13 {rm(o, w=w13):.3f} | 고 {rm(o, ~Lq):.3f} 저 {rm(o, Lq):.3f} | 기울기 고 {sl(o, ~Lq):.3f} 저 {sl(o, Lq):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(len(d)):.2f} 폴드 {sum(d[fold == k].mean() < 0 for k in range(5))}/5 | test RMS차 {np.sqrt(((t - bt) ** 2).mean()):.2f}")
print("— 그룹별 (기존 → 접미사 %s) —" % SUF)
for k, nm in (("A", "DINOv2 4보기"), ("D", "DINOv2 16보기"), ("T", "TIPSv2"), ("F", "고정 CNN")):
    print(f"  {nm:14s} OOF {rm(B[k][0]):.3f} → {rm(N[k][0]):.3f} | 저품질 {rm(B[k][0], Lq):.3f} → {rm(N[k][0], Lq):.3f} | Public꼴 {rm(B[k][0], w=w166):.3f} → {rm(N[k][0], w=w166):.3f}")
for k, v in Pm.items(): print(f"  측정 블록 P{k:8s} OOF {rm(P0[0]):.3f} → {rm(v[0]):.3f} | 저품질 {rm(P0[0], Lq):.3f} → {rm(v[0], Lq):.3f} | Public꼴 {rm(P0[0], w=w166):.3f} → {rm(v[0], w=w166):.3f}")
print("— 블렌드 (가중치는 v19a와 동일) —"); show("v19a (기준)", base, bt)
v20 = (blend(N, P0), blend(N, P0, i=1)); show(f"v20a: 4계열 {SUF}, P 그대로", *v20)
for k, v in Pm.items(): show(f"v20: 4계열 {SUF} + P{k}", blend(N, v), blend(N, v, i=1))
noC = {**W, "C": 0}; s_ = sum(noC.values()); noC = {k: v / s_ for k, v in noC.items()}; show("v20 (C 제거)", blend(N, P0, noC), blend(N, P0, noC, i=1))
print("— 사후 배율을 더하면 (v20a 위에, 기준점 %.2f) —" % c)
for kh, kl in ((1.03, 1.03), (1.05, 1.05), (1.08, 1.08), (1.05, 1.00), (1.08, 1.03)):
    o = c + np.where(Lq, kl, kh) * (v20[0] - c); t = c + np.where(lqt, kl, kh) * (v20[1] - c); show(f"  v20a × (고 {kh}, 저 {kl})", o, t)
if "--save" in sys.argv:
    name = f"sub_v20a_lqgain{SUF.replace('_m', '')}.csv"; pd.DataFrame({"ID": te, "hardness": v20[1]}).to_csv(f"submissions/{name}", index=False)
    print(f"→ submissions/{name} | test 평균 {v20[1].mean():.2f} std {v20[1].std():.2f} (v19a {bt.std():.2f}) | 저품질 {lqt.sum()}장만 변경: 고품질 최대차 {np.abs(v20[1] - bt)[~lqt].max():.2e}")
