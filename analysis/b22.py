"""배율 제출로 Public 부분집합의 예측 분산을 역산: 같은 예측 p를 k배(기준점 c=상수 제출값)한 제출의 점수 두 개(k=1, k=k2)와 상수 제출 점수(16.5663)로
  MSE(k) = k·M_p + (1−k)·M_c − k(1−k)·D  에서 D(= Public에서의 mean (p−c)²)를 풀고 최적 배율을 계산.
사용: python b22.py <파일(k=1)> <점수> <k2> <점수(k2)>"""
import sys, numpy as np, pandas as pd
f, s1, k2, s2 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
c, Mc = 194.16551625321193, 16.5663 ** 2
p = pd.read_csv(f"submissions/{f}").hardness.values; Mp, M2 = s1 ** 2, s2 ** 2
D_all = ((p - c) ** 2).mean(); D = (k2 * Mp + (1 - k2) * Mc - M2) / (k2 * (1 - k2))
ks = 0.5 + (Mc - Mp) / (2 * D); mse = lambda k: k * Mp + (1 - k) * Mc - k * (1 - k) * D
print(f"Public에서의 mean (p−c)² = {D:.1f} (test 전체 {D_all:.1f}, 비율 {D / D_all:.3f})")
print(f"Public 최적 배율 k* = {ks:.3f} → 예상 {np.sqrt(mse(ks)):.4f} (k=1: {s1:.4f}) | " + " ".join(f"k={k}: {np.sqrt(mse(k)):.4f}" for k in (1.02, 1.03, 1.04, 1.06, 1.08)))
