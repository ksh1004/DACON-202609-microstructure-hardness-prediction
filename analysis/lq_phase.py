"""모드별로 '2상 분율' 측정이 경도를 얼마나 설명하는가. 임계값·평활 규모를 훑어 최대 상관을 고품질/저품질에서 비교.
입력 3종: (a) 디노이즈+조명보정 이미지, (b) 원본을 가우시안 σ2로 평활, (c) 원본을 σ4로 평활 (디노이저 없이 면적 평균만)."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from scipy import ndimage as ndi
from PIL import Image
hq = sig < 10
DN = np.load("outputs/denoised/unet_cm7_am22_l1_i10000_boot_fflowpass48_low.npz")["train"].astype(np.float32)
RAW = np.stack([np.asarray(Image.open(f"data/train/{i}.png"), np.float32) for i in ids])
def rel_dark(x, close=5):
    c = ndi.grey_closing(x, size=close) if close else x
    m = np.quantile(c, .7); return (m - c) / m
ths = [0.02, 0.03, 0.04, 0.05, 0.065, 0.08, 0.10, 0.12, 0.15, 0.20]
def scan(name, imgs, close=5):
    R = np.stack([rel_dark(x, close) for x in imgs])
    fr = np.stack([(R > t).mean((1, 2)) for t in ths], 1)             # 임계별 어두운 면적 분율
    qs = np.stack([np.quantile(R.reshape(len(R), -1), q, axis=1) for q in (.5, .75, .9, .95)], 1)  # 상대 어둡기 분위
    c = lambda v, m: np.corrcoef(v[m], y[m])[0, 1]
    print(f"\n[{name}] 임계값별 어두운 분율과 경도의 상관")
    print("   임계    " + " ".join(f"{t:6.3f}" for t in ths))
    print("   고품질  " + " ".join(f"{c(fr[:, j], hq):+6.3f}" for j in range(len(ths))) + f" | 최대 {max(c(fr[:, j], hq) for j in range(len(ths))):.3f}")
    print("   저품질  " + " ".join(f"{c(fr[:, j], ~hq):+6.3f}" for j in range(len(ths))) + f" | 최대 {max(c(fr[:, j], ~hq) for j in range(len(ths))):.3f}")
    print("   분율 중앙값 고품질 " + " ".join(f"{np.median(fr[hq, j]):6.3f}" for j in range(len(ths))))
    print("   분율 중앙값 저품질 " + " ".join(f"{np.median(fr[~hq, j]):6.3f}" for j in range(len(ths))))
    print("   상대 어둡기 분위(50/75/90/95%)와의 상관: 고품질 " + " ".join(f"{c(qs[:, j], hq):+.3f}" for j in range(4)) + " | 저품질 " + " ".join(f"{c(qs[:, j], ~hq):+.3f}" for j in range(4)))
    # 임계 10개 + 분위 4개를 모두 쓴 모드 내 회귀 (5폴드) — 측정 가능한 상분율 정보의 상한에 가까운 값
    from sklearn.linear_model import RidgeCV
    Z = np.column_stack([fr, qs]); out = {}
    for nm, m in (("고품질", hq), ("저품질", ~hq)):
        o = np.zeros(len(y))
        for k in range(5):
            t, v = (fold != k) & m, (fold == k) & m; mu, sd = Z[t].mean(0), Z[t].std(0) + 1e-9
            o[v] = RidgeCV(alphas=np.logspace(-3, 3, 30)).fit((Z[t] - mu) / sd, y[t]).predict((Z[v] - mu) / sd)
        out[nm] = 1 - ((o - y)[m] ** 2).mean() / y[m].var()
    print(f"   상분율 특징 14개만으로 모드 내 회귀 R²: 고품질 {out['고품질']:.3f} / 저품질 {out['저품질']:.3f}")
scan("디노이즈+조명보정", DN)
scan("원본 σ2 평활", [ndi.gaussian_filter(x, 2) for x in RAW])
scan("원본 σ4 평활 (닫기 없음)", [ndi.gaussian_filter(x, 4) for x in RAW], close=0)
# 저품질만: 조명 얼룩 제거(σ48 저역 제거) 후 σ3 평활
ff = [x - ndi.gaussian_filter(x, 48) + x.mean() for x in RAW]
scan("원본 조명 제거 + σ3 평활", [ndi.gaussian_filter(x, 3) for x in ff], close=0)
