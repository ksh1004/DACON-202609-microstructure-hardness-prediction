"""v16b: v16a에서 측정 블록 P를 조명보정 σ 4종 평균으로 (σ48 선택 운 제거 변형). v16a·v15a와 비교하고 제출 파일 생성."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
co, ct = grp(CNN)
do, dt = grp([E + f"mv16_dnppm0.05_ff{s}" for s in FF])
po, pt = oof(E + "mv16_mr_pp_sc"), tp(E + "mv16_mr_pp_sc")
p4o, p4t = grp([E + f"mv16_mr_pp_sc_ff{s}" for s in FF])
v15a_o = np.load("analysis/v15a_oof.npy")
sub = lambda f: pd.read_csv(f"submissions/{f}").set_index("ID").loc[te, "hardness"].values
v15a_t = sub("sub_v15a_dinopp.csv")
a_o, a_t = .15 * co + .55 * do + .3 * po, .15 * ct + .55 * dt + .3 * pt
print("v16a 파일 검산 RMS %.5f" % np.sqrt(((a_t - sub("sub_v16a_mv16.csv")) ** 2).mean()))
rep("P16 σ48", po); rep("P16 σ4종 평균", p4o, po)
rep("v15a", v15a_o); rep("v16a (.15C+.55D16+.30P16 σ48)", a_o, v15a_o)
def cand(name, o, t, save=None):
    rep(name, o, a_o); print(f"{'':40s} test 예측 RMS차: v16a 대비 {np.sqrt(((t - a_t) ** 2).mean()):.3f}, v15a 대비 {np.sqrt(((t - v15a_t) ** 2).mean()):.3f} | 표준편차 {t.std():.2f}")
    if save: pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{save}", index=False); print(f"{'':40s} → submissions/{save}")
cand("v16b (.15C+.55D16+.30P16 σ4평균)", .15 * co + .55 * do + .3 * p4o, .15 * ct + .55 * dt + .3 * p4t, "sub_v16b_mv16_pavg.csv" if "--save" in sys.argv else None)
cand("    (.15C+.85D16, P 없음)", .15 * co + .85 * do, .15 * ct + .85 * dt)
cand("    (.15C+.40D16+.45P16 σ48)", .15 * co + .40 * do + .45 * po, .15 * ct + .40 * dt + .45 * pt)
cand("    (.15C+.40D16+.45P16 σ4평균)", .15 * co + .40 * do + .45 * p4o, .15 * ct + .40 * dt + .45 * p4t)
