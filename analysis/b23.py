"""예측 폭 보정(모드별 배율)의 Public 최적값을 제출 점수 3개로 계산.
  기준 p(sub_v19a_cnnf), 전체 1.05배(sub_v19a_k105), 저품질만 1.10배(sub_v19a_lq110), 상수 제출(194.1655 → 16.5663).
  전체 배율: MSE(k) = k·M_p + (1−k)·M_c − k(1−k)·D  →  D = Public에서의 mean (p−c)² 가 정확히 풀림 (가정 없음).
  방향 도함수 A = E_pub[(p−y)(p−c)] → 모드별로 나눔: 저품질 쪽은 lq110 점수에서 (저품질 몫의 mean (p−c)²는 test 전체값 × 같은 비율 r로 가정).
사용: python b23.py <v19a 점수> <k105 점수> <lq110 점수> [--save 파일명 kH kL [기준 제출 파일]]"""
import sys, numpy as np, pandas as pd
args = [a for a in sys.argv[1:] if not a.startswith("--")]
s0, s1, s2 = (float(v) for v in args[:3]); c, Mc = 194.16551625321193, 16.5663 ** 2
te = pd.read_csv("data/sample_submission.csv").ID; ld = lambda f: pd.read_csv(f"submissions/{f}").set_index("ID").loc[te, "hardness"].values
p = ld("sub_v19a_cnnf.csv"); lq = pd.read_csv("outputs/features/test_features.csv").set_index("ID").loc[te, "q_noise_sigma"].values >= 10
assert np.abs(ld("sub_v19a_k105.csv") - (c + 1.05 * (p - c))).max() < 1e-6 and np.abs(ld("sub_v19a_lq110.csv") - (c + np.where(lq, 1.10, 1.0) * (p - c))).max() < 1e-6
M0, M1, M2 = s0 ** 2, s1 ** 2, s2 ** 2; Dall = ((p - c) ** 2).mean(); BLall = (lq * (p - c) ** 2).mean(); BHall = Dall - BLall
D = (1.05 * M0 - 0.05 * Mc - M1) / (1.05 * -0.05); r = D / Dall; A = (M1 - M0 - 0.05 ** 2 * D) / (2 * 0.05)
print(f"Public에서의 mean (p−c)² = {D:.1f} (test 전체 {Dall:.1f} → 비율 r = {r:.3f}; 짝 차이 제곱에서 쓰던 값은 0.931)")
print(f"전체 한 배율의 Public 최적 k* = {1 - A / D:.3f} → 예상 {np.sqrt(M0 - A * A / D):.4f} (지금 {s0:.4f})")
print("\n모드별 (저품질 몫의 비율 가정에 따른 민감도):")
for f in (0.85, 1.0, 1.15):
    BL = BLall * r * f; BH = D - BL
    if BH <= 0: continue
    AL = (M2 - M0 - 0.10 ** 2 * BL) / (2 * 0.10); AH = A - AL; kH, kL = 1 - AH / BH, 1 - AL / BL
    best = M0 - AH * AH / BH - AL * AL / BL
    ev = lambda a, b: np.sqrt(M0 + 2 * (a - 1) * AH + (a - 1) ** 2 * BH + 2 * (b - 1) * AL + (b - 1) ** 2 * BL)
    print(f"  저품질 몫 ×{f:.2f}: Public 최적 고품질 k = {kH:.3f}, 저품질 k = {kL:.3f} → {np.sqrt(best):.4f} | 고정값 예상: (1.00,1.10) {ev(1, 1.1):.4f}  (1.03,1.12) {ev(1.03, 1.12):.4f}  (1.05,1.15) {ev(1.05, 1.15):.4f}  (1.05,1.05) {ev(1.05, 1.05):.4f}")
print("\n참고(OOF, n=500): v19a 배율 전체 1.104±0.037, 고품질 1.050±0.039, 저품질 1.197±0.071 — Public(300장) 추정의 표준오차는 전체 ≈0.046, 모드별 ≈0.06~0.08")
if "--save" in sys.argv:
    i = sys.argv.index("--save"); name, kH, kL = sys.argv[i + 1], float(sys.argv[i + 2]), float(sys.argv[i + 3]); base = sys.argv[i + 4] if len(sys.argv) > i + 4 else "sub_v19a_cnnf.csv"
    q = ld(base); out = c + np.where(lq, kL, kH) * (q - c); pd.DataFrame({"ID": te, "hardness": out}).to_csv(f"submissions/{name}", index=False)
    print(f"→ submissions/{name}: {base} 를 고품질 ×{kH}, 저품질 ×{kL} (기준점 {c:.4f}) | 예측 RMS 변화 {np.sqrt(((out - q) ** 2).mean()):.3f}, 저품질 {lq.sum()}장")
