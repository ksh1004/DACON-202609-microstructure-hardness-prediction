"""v29b 배율 k: Public 곡선(세 점수로 정확) + test 전체 ≈ train OOF 곡선 가정 → Private 곡선 = (1000·전체 − 300·Public)/700."""
import sys, io, contextlib; sys.path.insert(0, "analysis"); sys.argv = ["b39_da.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b39_da.py", encoding="utf-8").read())
W = P3["v29b: FoL→(FoL+DAo)/2 + 재조정 0.5"]
C0 = 16.5662646014 ** 2; P1 = 9.7191411630 ** 2; P2 = 9.7570990478 ** 2; s2 = 1.06 / 1.03
a_, b_ = np.linalg.solve(np.array([[1, 1], [s2 ** 2, s2]]), [P1 - C0, P2 - C0]); pub = lambda k: a_ * (k / 1.03) ** 2 + b_ * (k / 1.03) + C0
oofm = lambda k: np.mean((bl(W, 0, k) - y) ** 2)
# 전체 test MSE 수준 보정: k=1.03에서 'test 전체 = OOF × γ'로 두되 γ는 1(그대로)과, Public이 OOF보다 낮은 만큼 전체도 낮다고 보는 경우 둘 다
for gam_name, gam in (("γ=1 (test 전체 = OOF)", 1.0), ("γ=Public/OOF 비 (낙관)", pub(1.03) / oofm(1.03))):
    print(gam_name)
    best = None
    for k in np.arange(0.96, 1.16, 0.01):
        tot = gam * oofm(k); prv = (1000 * tot - 300 * pub(k)) / 700
        if best is None or prv < best[1]: best = (k, prv)
        if abs(k * 100 - round(k * 100)) < 1e-6 and round(k * 100) % 2 == 0 or abs(k - 1.03) < 1e-6:
            print(f"  k {k:.2f} | Public {np.sqrt(pub(k)):.4f} | 전체(추정) {np.sqrt(tot):.4f} | Private(추정) {np.sqrt(prv):.4f}")
    print(f"  → Private 최적 k ≈ {best[0]:.2f} (Private {np.sqrt(best[1]):.4f})")
