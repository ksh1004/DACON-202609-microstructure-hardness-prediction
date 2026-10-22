"""모드 간 전이: 학습 집합 구성(고품질 비율 fH, 저품질 비율 fL)에 따른 모드별 OOF. 32보기 σ48 특징, v15b 블록 구성."""
import sys; sys.path.insert(0, "analysis")
from view_eval import *
X = avg(S(ALL, (0, 1, 2, 3)))
hq = sig < 10
def cv_sub(fH, fL, reps=3):
    """폴드 train 중 고품질의 fH, 저품질의 fL 비율만 사용. 반환: (고품질 RMSE, 저품질 RMSE, 평균 학습 표본 수)"""
    eh, el, ns = [], [], []
    for r in range(reps if (fH < 1 or fL < 1) else 1):
        rng = np.random.default_rng(100 + r); o = np.zeros(len(y))
        for k in range(5):
            t = fold != k; v = fold == k
            keep = t & np.where(hq, rng.random(len(y)) < fH, rng.random(len(y)) < fL)
            Z = design(X, keep); o[v] = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Z[keep], y[keep]).predict(Z[v]); ns.append(keep.sum())
        eh.append(np.sqrt(((o - y)[hq] ** 2).mean())); el.append(np.sqrt(((o - y)[~hq] ** 2).mean()))
    return np.mean(eh), np.mean(el), np.mean(ns)
print(f"고품질 {hq.sum()}장, 저품질 {(~hq).sum()}장 (폴드 train 기준 약 {hq.sum()*.8:.0f} / {(~hq).sum()*.8:.0f})")
print(f"{'학습 구성 (고품질 비율, 저품질 비율)':36s} {'n':>5s} {'고품질':>7s} {'저품질':>7s}")
for fH, fL in ((1, 1), (1, 0), (0, 1), (1, .5), (.5, 1), (.5, .5), (1, .25), (.25, 1), (.72, 0), (0.36, 0)):
    eh, el, n = cv_sub(fH, fL); print(f"  fH={fH:<5g} fL={fL:<5g} {'':20s} {n:5.0f} {eh:7.3f} {el:7.3f}", flush=True)
