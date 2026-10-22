"""보기 쌓기 Ridge (보기 잡음 공분산 벌점): 학습 행 = x̄_i + sqrt(λ)·(x_ik − x̄_i), 모든 보기 k. 평가 = 보기 평균 특징 x̄.
λ=0 은 특징 평균 Ridge(기존), λ=1 은 단순 증강 쌓기, λ>1 은 보기 불변성을 더 강하게.
동등한 원형(primal) 문제: min_w Σ_i (y_i − w·x̄_i)² + (λ/V) Σ_ik (w·(x_ik − x̄_i))² + α|w|²  (라벨이 보기와 무관하므로 교차항 소거)
→ 쌍대(dual)로 GPU에서 풂. 표준화·묶음 배율은 multi_ridge와 같게(보기 평균 특징의 폴드 train 통계)."""
import sys, time; sys.path.insert(0, "analysis")
from blend_eval import *
import torch
stem = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"
D = f"outputs/probe2_views/{stem}/"
NL, NB, C = 4, 11, 1024
GR0 = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05))
dev = "cuda"
_V = {}
def view(s, k, sp="train"):
    if (sp, s, k) not in _V: _V[(sp, s, k)] = np.load(D + f"{sp}_s{s}_k{k}.npy")
    return _V[(sp, s, k)]
def scale_fn(Xbar, t, GR=GR0):
    """보기 평균 특징의 폴드 train 통계 → (열 선택, 평균, 배율) : z = clip((x - mu)/sd, ±6) * s"""
    A = Xbar.reshape(len(Xbar), NL, NB, C)
    idx_all, mu_all, sd_all, s_all, ref = [], [], [], [], None
    cols = np.arange(NL * NB * C).reshape(NL, NB, C)
    for blocks, g in GR:
        ci = cols[:, list(blocks)].reshape(-1)
        Z = Xbar[:, ci].astype(np.float64); mu, sd = Z[t].mean(0), Z[t].std(0); keep = sd > 1e-10
        if ref is None: ref, s = keep.sum(), 1.0
        else: s = np.sqrt(g * ref / keep.sum())
        idx_all.append(ci[keep]); mu_all.append(mu[keep]); sd_all.append(sd[keep]); s_all.append(np.full(keep.sum(), s))
    return np.concatenate(idx_all), np.concatenate(mu_all), np.concatenate(sd_all), np.concatenate(s_all)
def stack_cv(views, lams, alphas=(1e-4, 1e-3, 1e-2, 1e-1), GR=GR0, eval_views=None, verbose=False):
    """반환: {(lam, alpha): oof}. alpha는 평균 대각 대비 상대값."""
    Xv = [view(s, k).astype(np.float32) for s, k in views]
    Xbar = np.mean(Xv, 0)
    Xe = Xbar if eval_views is None else np.mean([view(s, k).astype(np.float32) for s, k in eval_views], 0)
    out = {(l, a): np.zeros(len(y)) for l in lams for a in alphas}
    V = len(views)
    for f in range(5):
        t, v = np.where(fold != f)[0], np.where(fold == f)[0]
        ci, mu, sd, sc = scale_fn(Xbar, fold != f, GR)
        tf = lambda X: torch.clamp((torch.from_numpy(X[:, ci]).to(dev).double() - torch.from_numpy(mu).to(dev)) / torch.from_numpy(sd).to(dev), -6, 6) * torch.from_numpy(sc).to(dev)
        Zb = tf(Xbar[t]).float(); Ze = tf(Xe[v]).float()
        Zk = [tf(X[t]).float() for X in Xv]
        Zm = torch.stack(Zk).mean(0)          # 절단 후 보기 평균 (표준화 공간)
        c = Zm.mean(0, keepdim=True); ym = y[t].mean()
        Rows = torch.cat([Zm - c] + [z - Zm for z in Zk], 0)        # (n + V·n)×D : 평균 행 + 보기별 편차 행
        del Zk
        G = (Rows @ Rows.T).double(); n = len(t)
        Ke = ((Ze - c) @ Rows.T).double()                            # m×(n+V·n)
        Kem = Ke[:, :n]
        yt = torch.from_numpy(y[t] - ym).to(dev)
        for lam in lams:
            # 원형 문제: Σ_i (y_i − w·z̄_i)² + (λ/V) Σ_ik (w·d_ik)² + α|w|²  ⇔ 행 [z̄_i (목표 y_i), sqrt(λ/V)·d_ik (목표 0)]
            sclv = torch.ones(G.shape[0], dtype=torch.float64, device=dev); sclv[n:] = np.sqrt(lam / V)
            Gl = G * sclv[:, None] * sclv[None, :]
            tgt = torch.cat([yt, torch.zeros(G.shape[0] - n, dtype=torch.float64, device=dev)])
            dg = Gl.diagonal()[:n].mean()
            for a in alphas:
                if lam == 0:
                    sol = torch.linalg.solve(Gl[:n, :n] + a * dg * torch.eye(n, dtype=torch.float64, device=dev), yt)
                    p = Kem @ sol
                else:
                    sol = torch.linalg.solve(Gl + a * dg * torch.eye(G.shape[0], dtype=torch.float64, device=dev), tgt)
                    p = (Ke * sclv[None, :]) @ sol
                out[(lam, a)][v] = p.cpu().numpy() + ym
        del G, Gl, Rows, Ke; torch.cuda.empty_cache()
    return out
if __name__ == "__main__":
    A_, B_, ALL = [0, 1, 4, 5], [2, 3, 6, 7], list(range(8))
    S = lambda ks, ss=(0,): [(s, k) for s in ss for k in ks]
    rm = lambda o: np.sqrt(((o - y) ** 2).mean())
    lams = (0, 0.25, 1, 4, 16)
    for nm, views in (("B 4보기", S(B_)), ("D4 8보기", S(ALL)), ("B × 이동4 = 16보기", S(B_, (0, 1, 2, 3))), ("D4 8 × 이동4 = 32보기", S(ALL, (0, 1, 2, 3)))):
        t0 = time.time(); R = stack_cv(views, lams)
        print(f"\n== {nm} ({time.time()-t0:.0f}s) : 행 = λ, 열 = α(상대) {[a for (l, a) in R if l == 0]} ==")
        for l in lams: print(f"  λ={l:<5g} " + "  ".join(f"{rm(R[(l, a)]):.3f}" for (ll, a) in R if ll == l))
        base = R[(0, 1e-3)]; best = min(R, key=lambda k: rm(R[k]))
        rep("   λ=0 α=1e-3 (특징 평균)", base); rep(f"   최저 λ={best[0]:g} α={best[1]:g}", R[best], base); rep("   λ=1 α=1e-3", R[(1, 1e-3)], base)
        np.savez(f"analysis/vstack_{len(views)}.npz", **{f"l{l}_a{a}": R[(l, a)] for (l, a) in R})
        sys.stdout.flush()
