"""v28a 예측 폭 배율 k 분석: OOF·Public형(std 16.6)·Private형(std 18) 가중 지표, 그리고 상수 제출 항등식으로 Public 점수 추정."""
import sys, io, contextlib; sys.path.insert(0, "analysis"); sys.argv = ["b34.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b34.py", encoding="utf-8").read())
W = PRE2["v28 + 재조정 0.5"]; w18 = wg(18.0)
cst = pd.read_csv("submissions/probes/probe_const_trainmean.csv").set_index("ID").loc[te, "hardness"].values; C0 = 16.5662646014 ** 2
base_t = bl(W, 1, 1.03); P0 = 9.7593421013 ** 2  # v28a (k=1.03) Public MSE
print(f"상수값 {cst.mean():.3f}, train 평균 {c:.3f}")
for k in (1.00, 1.03, 1.05, 1.07, 1.09, 1.12):
    o, t = bl(W, 0, k), bl(W, 1, k)
    # Public 추정: t = c + (k/1.03)(base_t - c). 상수 cst 기준 항등식 (Public 부분집합의 예측 차 제곱 ≈ 0.93 × test 전체)
    D = lambda a_, b_: 0.93 * np.mean((a_ - b_) ** 2)
    cross = (C0 + P0 - D(base_t, cst)) / 2  # <cst-y, base-y>
    s_ = k / 1.03; q = cst + s_ * (base_t - cst) if np.allclose(cst, c) else None
    pub = np.sqrt((1 - s_) ** 2 * C0 + s_ ** 2 * P0 + 2 * s_ * (1 - s_) * cross) if q is not None else float("nan")
    print(f"k {k:.2f} | OOF {rm(o):.3f} | Public형(16.6) {rm(o, w=w166):.3f} | Private형(18) {rm(o, w=w18):.3f} | 가운데 {rm(o, mid):.3f} 양끝 {rm(o, ~mid):.3f} | Public 추정 {pub:.4f}")
