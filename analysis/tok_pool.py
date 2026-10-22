"""토큰 수준 2차·분포 풀링 시험 (tok_cache.py의 PCA 축소 토큰 사용, D4 8보기 평균).
지금의 풀링(채널별 평균·표준편차·영역 불균일성·상 조건부 평균)은 채널 사이의 결합 정보를 거의 쓰지 않는다. 채널 간 공분산(행렬 제곱근/로그 정규화),
상 조건부 공분산, 토큰 군집 히스토그램, 채널별 3·4차 모멘트를 추가 묶음으로 넣었을 때 계열 단독·3계열 블렌드가 좋아지는지 본다.
사용: python tok_pool.py dinoL | dn121"""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from joint_kernel import krr
import torch
tag = sys.argv[1]; Z = np.load(f"outputs/tok_cache/{tag}.npz"); TOK = torch.from_numpy(Z["tok"]).cuda(); DM = torch.from_numpy(Z["dm"]).cuda().float()
NL, NV, N, T, K = TOK.shape; print(f"[{tag}] 층 {NL} 보기 {NV} 이미지 {N} 토큰 {T} 차원 {K}")
iu = torch.triu_indices(K, K).cuda()
def mat_fn(C, mode):
    if mode == "raw": return C
    ev, Q = torch.linalg.eigh(C.double()); ev = ev.clamp_min(1e-8)
    f = ev.sqrt() if mode == "sqrt" else (ev + 1e-3 * ev.mean(-1, keepdim=True)).log()
    return (Q * f.unsqueeze(-2) @ Q.transpose(-1, -2)).float()
def wcov(z, w=None):
    """z (B,T,K) float, w (B,T) 가중 → 가중 공분산 (B,K,K)"""
    if w is None: w = torch.ones(z.shape[:2], device=z.device)
    s = w.sum(1, keepdim=True).clamp_min(1e-3); mu = (z * w.unsqueeze(-1)).sum(1) / s; zc = z - mu.unsqueeze(1)
    return torch.einsum("btk,btl,bt->bkl", zc, zc, w) / s.unsqueeze(-1)
@torch.no_grad()
def feats(kind):
    out = []
    for l in range(NL):
        acc = 0
        for v in range(NV):
            parts = []
            for i in range(0, N, 100):
                z = TOK[l, v, i:i + 100].float(); d = DM[v, i:i + 100]
                if kind.startswith("cov_"): f = mat_fn(wcov(z), kind[4:])[:, iu[0], iu[1]]
                elif kind.startswith("pcov_"): f = torch.cat([mat_fn(wcov(z, w), kind[5:])[:, iu[0], iu[1]] for w in (d, 1 - d)], 1)
                elif kind == "mom34":
                    zc = z - z.mean(1, keepdim=True); sd = zc.std(1).clamp_min(1e-6); f = torch.cat([(zc ** 3).mean(1) / sd ** 3, (zc ** 4).mean(1) / sd ** 4], 1)
                elif kind == "meanstd": f = torch.cat([z.mean(1), z.std(1)], 1)
                elif kind.startswith("bow"):
                    Cb = CENT[l]; d2 = (z ** 2).sum(-1, keepdim=True) - 2 * z @ Cb.T + (Cb ** 2).sum(-1); a_ = torch.softmax(-d2 / TAU[l], -1)
                    f = torch.cat([a_.mean(1), (a_ * d.unsqueeze(-1)).mean(1)], 1) if kind.endswith("p") else a_.mean(1)
                parts.append(f)
            acc = acc + torch.cat(parts) / NV
        out.append(acc)
    return torch.cat(out, 1).cpu().numpy().astype(np.float64)
def kmeans(l, k, iters=25, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed); x = TOK[l, 0].float().reshape(-1, K); x = x[torch.randperm(len(x), generator=g, device="cuda")[:120000]]
    c = x[torch.randperm(len(x), generator=g, device="cuda")[:k]].clone()
    for _ in range(iters):
        a_ = torch.cdist(x, c).argmin(1)
        for j in range(k):
            m = a_ == j
            if m.any(): c[j] = x[m].mean(0)
    return c, torch.cdist(x, c).min(1).values.pow(2).median().item()
def kern(M):
    Ks = []
    for k in range(5):
        t = fold != k; mu, sd = M[t].mean(0), M[t].std(0); keep = sd > 1e-10; Zs = np.clip((M[:, keep] - mu[keep]) / sd[keep], -6, 6); Ks.append(Zs @ Zs.T / keep.sum())
    return np.array(Ks)
rm = lambda o: np.sqrt(((o - y) ** 2).mean())
J = dict(np.load("analysis/jk_cache.npz")); C2 = dict(np.load("analysis/cj_cache.npz"))
KD, KT = J["D16"], J["T16"]; KF = sum(C2[n] for n in ("densenet121-tv_in1k", "resnet50-a1_in1k", "resnet50-a2_in1k", "densenetblur121d-ra_in1k")) / 4
oD, oT, oF = krr(KD), krr(KT), krr(KF); base = .45 * oD + .25 * oT + .30 * oF
KB, ob = (KD, oD) if tag.startswith("dino") else (C2["densenet121-tv_in1k"], krr(C2["densenet121-tv_in1k"]))
print(f"기준 계열 단독 {rm(ob):.3f} | 3계열 블렌드 {rm(base):.3f}")
kinds = ["meanstd", "cov_raw", "cov_sqrt", "cov_log", "pcov_sqrt", "mom34"]
CENT, TAU = {}, {}
for kk in (32, 96):
    for l in range(NL): CENT[l], med = kmeans(l, kk); TAU[l] = med
    for kind in (f"bow{kk}", f"bow{kk}p"):
        M = feats("bow" if kind[-1] != "p" else "bowp"); kinds.append((kind, M))
for kd in kinds:
    kind, M = (kd, feats(kd)) if isinstance(kd, str) else kd
    Kx = kern(M); solo = krr(Kx); line = f"{kind:10s} ({M.shape[1]:5d}열) 단독 {rm(solo):.3f} |"
    for g in (0.1, 0.25, 0.5):
        o = krr(KB + g * Kx); d = (o - y) ** 2 - (ob - y) ** 2
        if tag.startswith("dino"): b = .45 * o + .25 * oT + .30 * oF
        else: b = .45 * oD + .25 * oT + .30 * krr(KF + g * Kx / 4)
        db = (b - y) ** 2 - (base - y) ** 2
        line += f" +×{g}: {rm(o):.3f} (Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f}) 블렌드 {rm(b):.3f} ({db.mean():+.2f}±{db.std() / np.sqrt(500):.2f}, {sum(db[fold == f].mean() < 0 for f in range(5))}/5) |"
    print(line, flush=True)
