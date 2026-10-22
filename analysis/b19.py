"""v18 후보: v17b 구성에 TIPSv2-So400m 16보기 DINO형 블록(tips16_dnppm0.05_ff*)을 구성원으로 추가.
최종 = .15 CNN + .15 P_A + .70 × [ (1−t)·(D_A .275 + D16 .425)/.70 + t·TIPS16 ]"""
import sys, os; sys.path.insert(0, "analysis")
from blend_eval import *
co, ct = grp(CNN); qo, qt = grp([E + f"dnpp0.25_ff{s}" for s in FF]); po, pt = oof(E + "mr_v14_pp_sc"), tp(E + "mr_v14_pp_sc")
do, dt = grp([E + f"mv16_dnppm0.05_ff{s}" for s in FF])
have = [s for s in FF if os.path.exists(E + f"tips16_dnppm0.05_ff{s}/test_pred_full.csv")]
print("TIPS 블록 준비된 σ:", have)
to, tt = grp([E + f"tips16_dnppm0.05_ff{s}" for s in have])
for s in have: rep(f"  TIPS16 σ{s}", oof(E + f"tips16_dnppm0.05_ff{s}"))
rep("D16 (DINOv2-L 16보기) 그룹", do); rep("D_A (DINOv2-L 4보기) 그룹", qo); rep(f"TIPS16 그룹 ({len(have)}종)", to, do)
print("잔차상관: TIPS16–D16 %.3f, TIPS16–D_A %.3f, D16–D_A %.3f | test 예측 RMS차 TIPS16–D16 %.2f, D16–D_A %.2f" % (
    np.corrcoef(to - y, do - y)[0, 1], np.corrcoef(to - y, qo - y)[0, 1], np.corrcoef(do - y, qo - y)[0, 1], np.sqrt(((tt - dt) ** 2).mean()), np.sqrt(((dt - qt) ** 2).mean())))
dmo, dmt = (.275 * qo + .425 * do) / .70, (.275 * qt + .425 * dt) / .70
b_o, b_t = .15 * co + .15 * po + .70 * dmo, .15 * ct + .15 * pt + .70 * dmt
rep("v17b", b_o)
w18 = np.exp(-0.5 * ((y - y.mean()) / 18.15) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
for t in (0.15, 0.2, 0.3, 0.4, 0.5):
    o = .15 * co + .15 * po + .70 * ((1 - t) * dmo + t * to); te_ = .15 * ct + .15 * pt + .70 * ((1 - t) * dmt + t * tt)
    rep(f"v18 (TIPS 몫 t={t})", o, b_o); print(f"{'':40s} test 예측 RMS차 v17b 대비 {np.sqrt(((te_ - b_t) ** 2).mean()):.3f} → Public 비교 잡음 ±{0.062 * np.sqrt(((te_ - b_t) ** 2).mean()):.3f}")
    if "--save" in sys.argv and abs(t - 0.3) < 1e-9 and len(have) == 4:
        pd.DataFrame({"ID": te, "hardness": te_}).to_csv("submissions/sub_v18a_tips.csv", index=False); print(f"{'':40s} → submissions/sub_v18a_tips.csv")
