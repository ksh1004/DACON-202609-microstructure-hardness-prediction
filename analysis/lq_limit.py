"""저품질 모드의 정보 한계 측정: 고품질 train 이미지를 합성 열화(대비↓·블러·노이즈 → 디노이즈 → σ48 조명보정)한 사본으로
'같은 영역끼리' 학습·평가. hetero3 특징(4보기), 층 11~17. 비교:
 (1) 깨끗한 것으로 학습 → 깨끗한 것 평가 / (2) 합성 저품질로 학습 → 합성 저품질 평가 / (3) 교차 / (4) 실제 저품질끼리 (같은 표본 수)"""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from sklearn.linear_model import RidgeCV
P = "outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Pphase2_Sunet_cm7_am22_l1_i10000_boot_fflowpass48_low/emb.npz"
C = np.load(P)["train"].astype(np.float64).reshape(len(y), 7, 10, 1024)[:, 1:5, :6].reshape(len(y), -1)   # 실제 이미지(고·저품질 모두) hetero3
Sx = np.load("outputs/synlow/cm10_x2_s0/emb.npz")["train"].astype(np.float64).reshape(-1, 7, 6, 1024)[:, 1:5].reshape(-1, 4 * 6 * 1024)
meta = pd.read_csv("outputs/synlow/cm10_x2_s0/meta.csv"); idx = {i: k for k, i in enumerate(ids)}
src = meta.ID.map(idx).values; cp = meta["copy"].values
hq = sig < 10; H = np.where(hq)[0]; Lq = np.where(~hq)[0]
S0 = np.zeros_like(C); S1 = np.zeros_like(C); S0[src[cp == 0]] = Sx[cp == 0]; S1[src[cp == 1]] = Sx[cp == 1]
print(f"고품질 {len(H)}장 (y 표준편차 {y[H].std():.2f}), 저품질 {len(Lq)}장 (y 표준편차 {y[Lq].std():.2f}) | 사본 열화: 대비 {meta.contrast.min():.2f}~{meta.contrast.max():.2f}, 블러 {meta.blur.min():.2f}~{meta.blur.max():.2f}, 노이즈 {meta.noise.min():.0f}~{meta.noise.max():.0f}")
def fit_pred(Xt, yt, Xv):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-8
    return RidgeCV(alphas=np.logspace(-1, 7, 50)).fit(np.clip((Xt - mu) / sd, -6, 6), yt).predict(np.clip((Xv - mu) / sd, -6, 6))
rm = lambda o, m: np.sqrt(((o - y)[m] ** 2).mean())
def run(train_sets, test_set, pool, n_sub=None, seed=0, own_scale=False):
    """pool: 대상 이미지 인덱스. train_sets: 학습 행으로 쓸 특징 행렬 목록(각각 같은 라벨). test_set: 평가 특징 행렬."""
    o = np.zeros(len(y)); rng = np.random.default_rng(seed)
    for k in range(5):
        t = np.array([i for i in pool if fold[i] != k]); v = np.array([i for i in pool if fold[i] == k])
        if n_sub: t = rng.choice(t, n_sub, replace=False)
        Xt = np.vstack([A[t] for A in train_sets]); yt = np.tile(y[t], len(train_sets))
        o[v] = fit_pred(Xt, yt, test_set[v])
    m = np.zeros(len(y), bool); m[pool] = True; return rm(o, m), o
print("\n== 고품질 이미지 %d장 안에서 (폴드 train 약 %d장) ==" % (len(H), int(len(H) * .8)))
e1, o1 = run([C], C, H); print(f"(1) 깨끗 → 깨끗              RMSE {e1:.3f}  R² {1 - e1**2 / y[H].var():.3f}")
e2, o2 = run([S0], S0, H); print(f"(2) 합성저품질 → 합성저품질   RMSE {e2:.3f}  R² {1 - e2**2 / y[H].var():.3f}   (사본 1개로 학습·평가)")
e2b, o2b = run([S0, S1], S0, H); e2c, o2c = run([S0, S1], S1, H); print(f"    사본 2개로 학습           RMSE {e2b:.3f} / {e2c:.3f} (사본0 / 사본1 평가), 두 사본 예측 평균 {rm((o2b + o2c) / 2, hq):.3f}")
e3, _ = run([C], S0, H); print(f"(3) 깨끗 → 합성저품질         RMSE {e3:.3f}")
e4, _ = run([S0], C, H); print(f"(4) 합성저품질 → 깨끗         RMSE {e4:.3f}")
e5, _ = run([C, S0], S0, H); print(f"(5) 깨끗+합성저품질 → 합성저품질 RMSE {e5:.3f}")
n = int(len(Lq) * .8) - 2
print(f"\n== 같은 학습 표본 수 {n}장으로 맞춘 비교 ==")
a = np.mean([run([C], C, H, n, s)[0] for s in range(3)]); b = np.mean([run([S0], S0, H, n, s)[0] for s in range(3)]); c = run([C], C, Lq)[0]
print(f"고품질 깨끗(영역 내)        RMSE {a:.3f}  R² {1 - a**2 / y[H].var():.3f}")
print(f"고품질의 합성 저품질(영역 내) RMSE {b:.3f}  R² {1 - b**2 / y[H].var():.3f}")
print(f"실제 저품질(영역 내)        RMSE {c:.3f}  R² {1 - c**2 / y[Lq].var():.3f}")
d = np.mean([run([C], C, list(H) + list(Lq), None, s)[1] for s in range(1)], 0)
print(f"\n참고: 전체 500장 학습 시 고품질 {rm(d, hq):.3f}, 저품질 {rm(d, ~hq):.3f} (이 특징 구성 기준)")
