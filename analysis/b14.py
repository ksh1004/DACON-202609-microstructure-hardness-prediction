"""v16 후보: DINO 그룹과 측정 블록을 16보기 평균 특징으로 교체. v15a·v15b와 비교하고 제출 파일 생성."""
import sys, os; sys.path.insert(0, "analysis")
from blend_eval import *
co, ct = grp(CNN)
qo, qt = grp([E + f"dnpp0.25_ff{s}" for s in FF]); po, pt = oof(E + "mr_v14_pp_sc"), tp(E + "mr_v14_pp_sc")
mo_, mt_ = grp([E + f"dnppm0.05_ff{s}" for s in FF])
v15a_o, v15a_t = .15 * co + .55 * qo + .3 * po, .15 * ct + .55 * qt + .3 * pt
v15b_o, v15b_t = .15 * co + .55 * mo_ + .3 * po, .15 * ct + .55 * mt_ + .3 * pt
sub = lambda f: pd.read_csv(f"submissions/{f}").set_index("ID").loc[te, "hardness"].values
print("재구성 검산: v15a 제출 파일과 RMS 차 %.4f, v15b %.4f" % (np.sqrt(((v15a_t - sub("sub_v15a_dinopp.csv")) ** 2).mean()), np.sqrt(((v15b_t - sub("sub_v15b_mixrule.csv")) ** 2).mean())))
have = [s for s in FF if os.path.exists(E + f"mv16_dnppm0.05_ff{s}/oof.csv")]
print("16보기 블록이 준비된 σ:", have)
rep("v15a (기존 4보기, 운 포함)", v15a_o); rep("v15b (기존 4보기 + 기여량)", v15b_o, v15a_o)
for s in have:
    rep(f"  DINO 블록 σ{s}: 기존 dnppm", oof(E + f"dnppm0.05_ff{s}")); rep(f"  DINO 블록 σ{s}: 16보기 dnppm", oof(E + f"mv16_dnppm0.05_ff{s}"), oof(E + f"dnppm0.05_ff{s}"))
no, nt = grp([E + f"mv16_dnppm0.05_ff{s}" for s in have]); n25o, n25t = grp([E + f"mv16_dnpp0.25_ff{s}" for s in have])
rep("DINO 그룹 기존 dnppm ff4", mo_); rep(f"DINO 그룹 16보기 dnppm ({len(have)}종)", no, mo_); rep(f"DINO 그룹 16보기 dnpp ({len(have)}종)", n25o, mo_)
npo, npt = oof(E + "mv16_mr_pp_sc"), tp(E + "mv16_mr_pp_sc")
rep("측정 블록 P 기존", po); rep("측정 블록 P 16보기", npo, po)
w18 = np.exp(-0.5 * ((y - y.mean()) / 18.15) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
def cand(name, o, t, save=None):
    rep(name, o, v15a_o)
    print(f"{'':40s} pw18 {np.sqrt(np.average((o - y) ** 2, weights=w18)):.3f} | test 예측 RMS차: v15a 대비 {np.sqrt(((t - v15a_t) ** 2).mean()):.3f}, v15b 대비 {np.sqrt(((t - v15b_t) ** 2).mean()):.3f} | test 평균 {t.mean():.2f} 표준편차 {t.std():.2f}")
    if save:
        pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{save}", index=False); print(f"{'':40s} → submissions/{save}")
print()
cand("v16a = .15C + .55 D16(dnppm) + .30 P16", .15 * co + .55 * no + .3 * npo, .15 * ct + .55 * nt + .3 * npt, "sub_v16a_mv16.csv" if len(have) == 4 and "--save" in sys.argv else None)
cand("     = .15C + .55 D16(dnppm) + .30 P기존", .15 * co + .55 * no + .3 * po, .15 * ct + .55 * nt + .3 * pt)
cand("     = .15C + .55 D16(dnpp) + .30 P16", .15 * co + .55 * n25o + .3 * npo, .15 * ct + .55 * n25t + .3 * npt)
cand("     = .10C + .55 D16(dnppm) + .35 P16", .10 * co + .55 * no + .35 * npo, .10 * ct + .55 * nt + .35 * npt)
cand("     = .15C + .45 D16(dnppm) + .40 P16", .15 * co + .45 * no + .40 * npo, .15 * ct + .45 * nt + .40 * npt)
print("\ntest 예측의 보기 잡음 추정: 기존 D 그룹 vs 16보기 D 그룹 RMS차 %.3f (σ48 블록끼리 %.3f), P 블록 %.3f" % (
    np.sqrt(((mt_ - nt) ** 2).mean()), np.sqrt(((tp(E + "dnppm0.05_ff48") - tp(E + "mv16_dnppm0.05_ff48")) ** 2).mean()), np.sqrt(((pt - npt) ** 2).mean())))
print("OOF에서 같은 양: D 그룹 %.3f, σ48 블록 %.3f, P 블록 %.3f" % (np.sqrt(((mo_ - no) ** 2).mean()), np.sqrt(((oof(E + "dnppm0.05_ff48") - oof(E + "mv16_dnppm0.05_ff48")) ** 2).mean()), np.sqrt(((po - npo) ** 2).mean())))
