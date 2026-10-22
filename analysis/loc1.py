"""경도가 '어디'를 보는가: 위치별 국소 2상 분율·어둡기·경계 밀도로 경도를 설명할 때 공간 가중치가 균일한지 검사.
전역 통계로 정해지는 라벨이면 중심부/테두리 어느 영역의 분율이든 (면적이 같으면) 경도와의 상관이 같아야 한다."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from scipy import ndimage as ndi
from sklearn.linear_model import RidgeCV
X = np.load("outputs/denoised/unet_cm7_am22_l1_i10000_boot_fflowpass48_low.npz")["train"].astype(np.float32)
hq = sig < 10
G = 16; cs = 256 // G
def maps(x):
    c = ndi.grey_closing(x, size=5); m = np.quantile(c, .7); rel = (m - c) / m
    gx, gy = np.gradient(ndi.gaussian_filter(x, 1.0)); gm = np.hypot(gx, gy) / m
    cell = lambda a: a.reshape(G, cs, G, cs).mean((1, 3))
    return cell((rel > 0.08).astype(np.float32)), cell(rel), cell(gm), cell((rel > 0.25).astype(np.float32))
M = [np.stack(v) for v in zip(*[maps(x) for x in X])]  # 4 × (500, G, G)
names = ["2상 분율(>8%)", "상대 어둡기", "경계(기울기) 밀도", "기공(>25%)"]
ii, jj = np.mgrid[0:G, 0:G]; ring = np.minimum(np.minimum(ii, jj), np.minimum(G - 1 - ii, G - 1 - jj))  # 0(테두리)~7(중심)
rad = np.hypot(ii - (G - 1) / 2, jj - (G - 1) / 2)
def corr(a, m): return np.corrcoef(a[m], y[m])[0, 1]
for k, nm in enumerate(names):
    F = M[k]
    print(f"\n=== {nm} ===  전역 평균과 경도의 상관: 전체 {corr(F.mean((1,2)), slice(None)):+.3f} / 고품질 {corr(F.mean((1,2)), hq):+.3f}")
    print("  테두리 거리 고리별(0=테두리 … 7=중심) 국소 평균과 경도의 상관 [전체 | 고품질] (괄호: 칸 수)")
    print("   " + " ".join(f"r{r}({(ring==r).sum()}) {corr(F[:, ring==r].mean(1), slice(None)):+.3f}|{corr(F[:, ring==r].mean(1), hq):+.3f}" for r in range(8)))
    print("  중심 k×k 칸 평균 vs 같은 면적의 테두리 띠:")
    for kk in (2, 4, 6, 8, 12):
        lo = (G - kk) // 2; cen = F[:, lo:lo+kk, lo:lo+kk].mean((1, 2))
        print(f"   중심 {kk*cs:3d}px 정사각: 전체 {corr(cen, slice(None)):+.3f} 고품질 {corr(cen, hq):+.3f}")
    # 칸별 단순 상관 지도 (D4 대칭화, 왼쪽 위 8×8 사분면만)
    cm = np.array([[np.corrcoef(F[:, i, j], y)[0, 1] for j in range(G)] for i in range(G)])
    sym = np.mean([np.rot90(c, r) for c in (cm, cm.T) for r in range(4)], 0)
    print("  칸별 상관 지도 (D4 대칭화, 왼쪽 위 사분면 8×8, 왼쪽 위 = 모서리, 오른쪽 아래 = 중심) ×100:")
    for i in range(8): print("   " + " ".join(f"{sym[i, j]*100:4.0f}" for j in range(8)))
# 회귀 비교: 전역 분율 1개 vs 고리 8개 vs 칸 256개 (2상 분율·어둡기·경계 3종)
def cvr(Z, m=None, alphas=np.logspace(-3, 5, 40)):
    m = np.ones(len(y), bool) if m is None else m; o = np.zeros(len(y))
    for k in range(5):
        t, v = (fold != k) & m, (fold == k) & m
        mu, sd = Z[t].mean(0), Z[t].std(0) + 1e-9
        o[v] = RidgeCV(alphas=alphas).fit((Z[t] - mu) / sd, y[t]).predict((Z[v] - mu) / sd)
    return np.sqrt(((o - y)[m] ** 2).mean())
glob = np.column_stack([F.mean((1, 2)) for F in M[:3]])
rings = np.column_stack([F[:, ring == r].mean(1) for F in M[:3] for r in range(8)])
print("\n=== 회귀 (5폴드 RMSE) [전체 | 고품질만] ===")
print(f"  전역 평균 3개            : {cvr(glob):.3f} | {cvr(glob, hq):.3f}")
print(f"  고리별 평균 3×8          : {cvr(rings):.3f} | {cvr(rings, hq):.3f}")
for g2 in (2, 4, 8):
    Z = np.column_stack([F.reshape(len(y), g2, G // g2, g2, G // g2).mean((2, 4)).reshape(len(y), -1) for F in M[:3]])
    # 위치 무관 대조군: 칸을 이미지마다 무작위로 섞으면(위치 정보 파괴) 같은 정보량의 '순서 없는' 특징
    print(f"  {g2}×{g2} 칸별 평균 3×{g2*g2}       : {cvr(Z):.3f} | {cvr(Z, hq):.3f}")
np.save("analysis/loc_maps.npy", np.stack(M))
