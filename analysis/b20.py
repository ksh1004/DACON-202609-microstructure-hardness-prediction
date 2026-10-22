"""v19 후보: 고정 특징 구성원 세 계열(DINOv2-L[4보기+16보기], TIPSv2-So400m, ImageNet CNN stride-8 고정 특징) 블렌드.
성분: C = 미세조정 CNN 9개, P = 측정 블록(mr_v14_pp_sc), A = DINOv2 4보기(dnpp0.25), D = DINOv2 16보기(mv16_dnppm0.05), T = TIPS16, F = 고정 CNN(cnnf_*).
사용: python b20.py [--save]"""
import sys, os; sys.path.insert(0, "analysis")
from blend_eval import *
CNNF = ("dn121", "r50a1", "r50a2")
co, ct = grp(CNN); po, pt = oof(E + "mr_v14_pp_sc"), tp(E + "mr_v14_pp_sc")
ao, at = grp([E + f"dnpp0.25_ff{s}" for s in FF]); do, dt = grp([E + f"mv16_dnppm0.05_ff{s}" for s in FF]); to, tt = grp([E + f"tips16_dnppm0.05_ff{s}" for s in FF])
have = [(n, s) for n in CNNF for s in FF if os.path.exists(E + f"cnnf_{n}_ff{s}/test_pred_full.csv")]
print("고정 CNN 블록:", len(have), "/", len(CNNF) * len(FF))
for n in CNNF:
    ds = [E + f"cnnf_{n}_ff{s}" for (m, s) in have if m == n]
    if ds: rep(f"  고정 CNN {n} ({len(ds)}σ 평균)", grp(ds)[0])
fo, ft = grp([E + f"cnnf_{n}_ff{s}" for n, s in have])
rep("A  DINOv2 4보기", ao); rep("D  DINOv2 16보기", do); rep("T  TIPSv2 16보기", to); rep("F  고정 CNN 그룹", fo); rep("C  미세조정 CNN 9", co); rep("P  측정 블록", po)
cr = lambda a, b: np.corrcoef(a - y, b - y)[0, 1]
print("잔차상관: F–D %.3f, F–T %.3f, F–C %.3f, D–T %.3f | test 예측 RMS차 F–D %.2f" % (cr(fo, do), cr(fo, to), cr(fo, co), cr(do, to), np.sqrt(((ft - dt) ** 2).mean())))
dmo, dmt = (.275 * ao + .425 * do) / .70, (.275 * at + .425 * dt) / .70
def mk(wc, wp, a, b, c):   # 나머지(1-wc-wp)를 (DINOv2 혼합, TIPS, 고정 CNN) = (a, b, c) 로
    r = 1 - wc - wp; return wc * co + wp * po + r * (a * dmo + b * to + c * fo), wc * ct + wp * pt + r * (a * dmt + b * tt + c * ft)
v18o, v18t = mk(.15, .15, .7, .3, 0)
sub = lambda f: pd.read_csv(f"submissions/{f}").set_index("ID").loc[te, "hardness"].values
print("v18a 재구성 검산 RMS %.5f" % np.sqrt(((v18t - sub("sub_v18a_tips.csv")) ** 2).mean()))
rep("v18a (Public 10.1232)", v18o)
cands = {"v19a  C.15 P.15 | D.45 T.25 F.30": (.15, .15, .45, .25, .30), "      C.15 P.15 | D.40 T.30 F.30": (.15, .15, .40, .30, .30), "      C.15 P.15 | D.50 T.25 F.25": (.15, .15, .50, .25, .25),
         "      C.15 P.15 | D.40 T.25 F.35": (.15, .15, .40, .25, .35), "      C.15 P.15 | D.5 T.5 F0 (v18b)": (.15, .15, .5, .5, 0),
         "v19b  C0 P.15 | D.45 T.25 F.30": (0, .15, .45, .25, .30), "v19c  C0 P0 | D.45 T.25 F.30": (0, 0, .45, .25, .30), "      C.15 P0 | D.45 T.25 F.30": (.15, 0, .45, .25, .30),
         "      C.075 P.15 | D.45 T.25 F.30": (.075, .15, .45, .25, .30)}
save = {"v19a": "sub_v19a_cnnf.csv", "v19b": "sub_v19b_cnnf_noC.csv", "v19c": "sub_v19c_cnnf_noC_noP.csv", "(v18b)": "sub_v18b_tips50.csv"}
for nm, w in cands.items():
    o, t = mk(*w); rep(nm, o, v18o)
    print(f"{'':40s} test 예측 RMS차 v18a 대비 {np.sqrt(((t - v18t) ** 2).mean()):.3f} | 표준편차 {t.std():.2f}")
    for k, f in save.items():
        if "--save" in sys.argv and k in nm and len(have) == len(CNNF) * len(FF):
            pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{f}", index=False); print(f"{'':40s} → submissions/{f}")
