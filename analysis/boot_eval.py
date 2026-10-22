"""블록 부트스트랩 표본 잡음 벌점 Ridge: min_w Σ_i (y_i − w·z_i)² + (λ/R) Σ_ir (w·d_ir)² + α|w|²,  d_ir = 표본 r의 특징 − 같은 이미지 표본들의 평균.
λ=0 은 기존 Ridge. 표준화·묶음 배율은 원래 특징(8보기 평균)의 폴드 train 통계."""
import sys; sys.path.insert(0, "analysis")
from view_eval import *
import torch
from view_stack import scale_fn
R = 6
X0 = np.load(D + "boot_orig_train.npy").astype(np.float32)
XR = [np.load(D + f"boot_r{r}_train.npy").astype(np.float32) for r in range(R)]
ref = avg(S(ALL)); print("검산: 부트스트랩 스크립트의 원본 특징 vs 기존 8보기 평균 RMS차 %.4f (표준편차 대비 %.4f)" % (np.sqrt(((X0 - ref) ** 2).mean()), np.sqrt(((X0 - ref) ** 2).mean()) / ref.std()))
dv = np.stack(XR) - np.mean(XR, 0)
nz = (dv ** 2).mean((0, 1)); tot = X0.var(0)
names = ["평균", "std", "het2", "het4", "het8", "het4내부std", "2상평균", "2상std", "기지평균", "기지std", "기여량"]
r = (nz / np.maximum(tot, 1e-12)).reshape(NL, NB, C)
print("블록별 표본 잡음 분산 ÷ 이미지 간 분산: " + " | ".join(f"{n} {r[:, b].mean():.2f}" for b, n in enumerate(names)))
F_ = pd.read_csv("outputs/features/train_features.csv").set_index("ID").loc[ids]; gc = F_.gr_count.values
e = (dv ** 2).mean((0, 2)) ; print("이미지별 표본 잡음 크기와 결정립 수의 상관(Spearman): %.3f" % pd.Series(e).corr(pd.Series(gc), method="spearman"))
dev = "cuda"
def run(lams, alphas=(1e-4, 1e-3, 1e-2, 1e-1), blocks_pen=None):
    out = {(l, a): np.zeros(len(y)) for l in lams for a in alphas}
    for f in range(5):
        t, v = np.where(fold != f)[0], np.where(fold == f)[0]
        ci, mu, sd, sc = scale_fn(X0, fold != f)
        tf = lambda A: (torch.clamp((torch.from_numpy(A[:, ci]).to(dev).double() - torch.from_numpy(mu).to(dev)) / torch.from_numpy(sd).to(dev), -6, 6) * torch.from_numpy(sc).to(dev)).float()
        Zt, Zv = tf(X0[t]), tf(X0[v]); c = Zt.mean(0, keepdim=True); ym = y[t].mean()
        Zr = [tf(A[t]) for A in XR]; Zm = torch.stack(Zr).mean(0)
        Dv = torch.cat([z - Zm for z in Zr], 0)
        if blocks_pen is not None:   # 벌점을 줄 블록만 남김
            bl = (ci // C) % NB; keep = torch.from_numpy(np.isin(bl, blocks_pen)).to(dev); Dv = Dv * keep
        Rows = torch.cat([Zt - c, Dv], 0); G = (Rows @ Rows.T).double(); n = len(t)
        Ke = ((Zv - c) @ Rows.T).double(); yt = torch.from_numpy(y[t] - ym).to(dev)
        tgt = torch.cat([yt, torch.zeros(G.shape[0] - n, dtype=torch.float64, device=dev)])
        for lam in lams:
            s = torch.ones(G.shape[0], dtype=torch.float64, device=dev); s[n:] = np.sqrt(lam / R)
            Gl = G * s[:, None] * s[None, :]; dg = Gl.diagonal()[:n].mean()
            for a in alphas:
                if lam == 0: p = Ke[:, :n] @ torch.linalg.solve(Gl[:n, :n] + a * dg * torch.eye(n, dtype=torch.float64, device=dev), yt)
                else: p = (Ke * s[None, :]) @ torch.linalg.solve(Gl + a * dg * torch.eye(G.shape[0], dtype=torch.float64, device=dev), tgt)
                out[(lam, a)][v] = p.cpu().numpy() + ym
        del G, Gl, Rows, Ke; torch.cuda.empty_cache()
    return out
rm = lambda o: np.sqrt(((o - y) ** 2).mean())
lams = (0, 0.1, 0.3, 1, 3, 10, 30)
for nm, bp in (("모든 블록에 벌점", None), ("평균·std·상풀링·기여량 블록만 (불균일성 블록 제외)", (0, 1, 6, 7, 8, 9, 10)), ("불균일성 블록만", (2, 3, 4, 5))):
    O = run(lams, blocks_pen=bp)
    print(f"\n== {nm} : 행 = λ, 열 = α(상대) {[a for (l, a) in O if l == 0]} ==")
    for l in lams: print(f"  λ={l:<5g} " + "  ".join(f"{rm(O[(l, a)]):.3f}" for (ll, a) in O if ll == l))
    base = O[(0, 1e-3)]; best = min(O, key=lambda k: rm(O[k]))
    rep("   λ=0 (기존)", base); rep(f"   최저 λ={best[0]:g} α={best[1]:g}", O[best], base); rep("   λ=1 α=1e-3", O[(1, 1e-3)], base)
    sys.stdout.flush()
