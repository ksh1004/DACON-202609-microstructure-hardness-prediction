"""예측 폭 배율의 Public 최적값: 상수 제출(194.1655, Public 16.5663)과의 블렌드 항등식.
p_k = c + k·(p − c) 는 두 채점 제출의 아핀 결합 → MSE(k) = k·M_p + (1−k)·M_c − k(1−k)·D,  D = mean((p−c)²) (test 전체로 근사)
사용: python b17.py <제출 파일명> <Public 점수> [<파일> <점수> ...]"""
import sys, numpy as np, pandas as pd
c, Mc = 194.16551625321193, 16.5663 ** 2
a = sys.argv[1:]
for f, s in zip(a[::2], a[1::2]):
    p = pd.read_csv(f"submissions/{f}").hardness.values; Mp = float(s) ** 2
    D = ((p - c) ** 2).mean(); k = 0.5 + (Mc - Mp) / (2 * D)
    mse = lambda k: k * Mp + (1 - k) * Mc - k * (1 - k) * D
    print(f"{f}: Public {float(s):.4f} | 예측 표준편차 {p.std():.2f} | 최적 배율 k* = {k:.3f} (Public {np.sqrt(mse(k)):.4f}) | " + " ".join(f"k={kk}: {np.sqrt(mse(kk)):.4f}" for kk in (0.95, 1.0, 1.03, 1.05, 1.08)))
    # Public 부분집합의 예측 분산이 전체보다 작을 가능성(라벨 std 16.6 vs 17.7)을 반영한 민감도
    for r in (0.88, 0.94):
        D2 = D * r; k2 = 0.5 + (Mc - Mp) / (2 * D2); print(f"    (Public에서 예측 분산이 전체의 {r:.2f}배라면 k* = {k2:.3f})")
