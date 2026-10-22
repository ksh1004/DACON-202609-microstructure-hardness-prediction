import sys; sys.path.insert(0, "analysis")
from blend_eval import *
P2 = "outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Pphase2_Sunet_cm7_am22_l1_i10000_boot_fflowpass48_low"
o = {t: oof(P2 + s) for t, s in ((0.08, ""), (0.05, "_th0.05"), (0.12, "_th0.12"))}
for t, v in o.items(): rep(f"probe 상풀링 th{t}", v)
rep("probe th 3종 평균", np.mean(list(o.values()), 0), o[0.08])
r = {t: v - y for t, v in o.items()}
print("잔차상관 0.08-0.05 %.3f, 0.08-0.12 %.3f, 0.05-0.12 %.3f" % (np.corrcoef(r[0.08], r[0.05])[0, 1], np.corrcoef(r[0.08], r[0.12])[0, 1], np.corrcoef(r[0.05], r[0.12])[0, 1]))
