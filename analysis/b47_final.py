"""최종 선택: v29b와 v30n의 배율별 Public(정확/보정 추정)·Private(추정) 비교.
v29b: 세 점수(상수·k1.03·k1.06)로 Public 2차 곡선 정확. ||p−c||²의 Public/전체 비를 v29b에서 구해 v30n에 적용(v30n Public 점수 1개로 곡선 추정)."""
import sys, io, contextlib; sys.path.insert(0, "analysis"); sys.argv = ["b44_depth.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b44_depth.py", encoding="utf-8").read())
C0 = 16.5662646014 ** 2
W29, W30 = P5["v29b (기준)"], P5["v30n: v29b + Nk 재조정 0.5"]
t29, t30 = bl(W29, 1, 1.03), bl(W30, 1, 1.03)
# v29b 정확 곡선 f(s) = a s² + b s + C0, s = k/1.03
P1, P2, s2 = 9.7191411630 ** 2, 9.7570990478 ** 2, 1.06 / 1.03
a29, b29 = np.linalg.solve(np.array([[1, 1], [s2 ** 2, s2]]), [P1 - C0, P2 - C0])
ratio = a29 / np.mean((t29 - c) ** 2); print(f"v29b: Public ||p−c||² = {a29:.2f}, test 전체 {np.mean((t29 - c) ** 2):.2f} → 비 {ratio:.3f}")
a30 = ratio * np.mean((t30 - c) ** 2); b30_ = 9.7211481325 ** 2 - C0 - a30
pub = {"v29b": lambda k: a29 * (k / 1.03) ** 2 + b29 * (k / 1.03) + C0, "v30n": lambda k: a30 * (k / 1.03) ** 2 + b30_ * (k / 1.03) + C0}
oofm = {"v29b": lambda k: np.mean((bl(W29, 0, k) - y) ** 2), "v30n": lambda k: np.mean((bl(W30, 0, k) - y) ** 2)}
print(f"{'':6s} {'k':>5s} | {'Public':>7s} | Private 추정(γ=1) | Private 추정(낙관)")
for nm in ("v29b", "v30n"):
    gam = pub[nm](1.03) / oofm[nm](1.03)
    for k in (1.00, 1.03, 1.06, 1.08, 1.10):
        p1 = (1000 * oofm[nm](k) - 300 * pub[nm](k)) / 700; p2 = (1000 * gam * oofm[nm](k) - 300 * pub[nm](k)) / 700
        print(f"{nm:6s} {k:5.2f} | {np.sqrt(pub[nm](k)):7.4f} | {np.sqrt(p1):8.4f}         | {np.sqrt(p2):8.4f}")
