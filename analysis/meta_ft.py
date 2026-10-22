"""고정 CNN 특징의 지도 적응 (메타 Ridge 미세조정) — 시험용.
고정 특징 + Ridge는 train을 거의 보간하므로 보통의 미세조정(LP-FT)은 손실이 0에 가까워 배울 것이 없다(DINOv2에서 확인).
대신 "일부 이미지로 맞춘 Ridge가 나머지 이미지를 얼마나 잘 맞히는가"(내부 교차검증 오차)를 손실로 삼아 특징 추출기를 조정한다:
  매 스텝: 폴드 train 이미지 전체의 풀링 특징 F 계산 → 내부 5분할로 [Ridge(지지 집합) → 질의 집합 예측] → MSE → dL/dF → 블록에 역전파(2회 통과).
조정 대상은 stride-8 단계의 마지막 블록(DenseNet의 denseblock2)뿐이고, 그 앞단 출력은 8보기 모두 미리 계산해 둔다.
평가: 조정된 블록으로 8보기 평균 특징을 다시 뽑아 폴드 train으로 Ridge → 폴드 밖 100장 오차 (스텝 0 = 고정 특징 기준).
사용: python meta_ft.py --fold 0 [--steps 150 --lr 2e-5 --tune block|bn]"""
import argparse, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="densenet121.tv_in1k"); ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--steps", type=int, default=150); ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--eval-every", type=int, default=25)
ap.add_argument("--tune", choices=["block", "bn"], default="block"); ap.add_argument("--views", type=int, default=1, help="스텝당 평균할 무작위 보기 수")
ap.add_argument("--chunk", type=int, default=32); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--l2sp", type=float, default=0.0)
ap.add_argument("--lam-mult", type=float, default=1.0, help="내부 Ridge의 λ 배율 (스텝 0의 LOO 선택값 기준)")
ap.add_argument("--tag", default="")
a = ap.parse_args(); sys.argv = sys.argv[:1]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np, torch, torch.nn.functional as F, timm
from probe_models import dark_map
from train_cnn import d4
from blend_eval import *
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = "cuda"
S48 = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"; SIZE = 512
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05))
m = timm.create_model(a.model, pretrained=True).cuda().eval(); cfg = m.pretrained_cfg
names = [n for n, _ in m.features.named_children()]; i2 = names.index("denseblock2")
stem = torch.nn.Sequential(*[getattr(m.features, n) for n in names[:i2]]).eval(); blk = m.features.denseblock2.eval()
for p in m.parameters(): p.requires_grad_(False)
tun = [p for n, p in blk.named_parameters() if a.tune == "block" or "norm" in n]
for p in tun: p.requires_grad_(True)
init = [p.detach().clone() for p in tun]
print(f"[{a.model} fold {a.fold}] 조정 파라미터 {sum(p.numel() for p in tun) / 1e3:.0f}k / 블록 전체 {sum(p.numel() for p in blk.parameters()) / 1e3:.0f}k", flush=True)
mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1); std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
X = torch.from_numpy(np.load(ROOT / f"outputs/denoised/{S48}.npz")["train"]).unsqueeze(1); N = len(X)
H, DM = [], []; t0 = time.time()
with torch.no_grad():
    for k in range(8):
        hs, ds = [], []
        for i in range(0, N, 32):
            xk = d4(X[i:i + 32].cuda().float().div_(255), k)
            xi = (F.interpolate(xk, size=(SIZE, SIZE), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1) - mean) / std
            with torch.autocast("cuda", dtype=torch.bfloat16): h = stem(xi)
            hs.append(h.half()); ds.append(dark_map(xk, h.shape[-1]).half())
        H.append(torch.cat(hs)); DM.append(torch.cat(ds))
print(f"앞단 출력 캐시 {tuple(H[0].shape)} × 8보기 ({time.time() - t0:.0f}s, GPU {torch.cuda.memory_allocated() / 2 ** 30:.1f}GB)", flush=True)
def sstd(x, dim): return (x.var(dim) + 1e-8).sqrt()      # 역전파가 안전한 표준편차
def feat(z, dm):
    """phase2m 풀링(hetero3 6블록 + 상 조건부 4블록 + 기여량 1블록)과 같은 구성. sqrt 안에 1e-8을 더한 것만 다르다."""
    z = z.float(); zf = z.flatten(2); fs = [zf.mean(2), sstd(zf, 2)]
    for g in (2, 4, 8): fs.append(sstd(F.adaptive_avg_pool2d(z, g).flatten(2), 2))
    m1, m2 = F.adaptive_avg_pool2d(z, 4), F.adaptive_avg_pool2d(z * z, 4); fs.append(sstd(((m2 - m1 * m1).clamp_min(0) + 1e-8).sqrt().flatten(2), 2))
    w = dm.float().flatten(1).unsqueeze(1); ph = []
    for ww in (w, 1 - w):
        s = ww.sum(2, keepdim=True).clamp_min(1e-3); mu = (zf * ww).sum(2, keepdim=True) / s
        ph.append(mu.squeeze(2)); ph.append(((((zf - mu) ** 2 * ww).sum(2, keepdim=True) / s) + 1e-8).sqrt().squeeze(2))
    fs += [ph[0], ph[2], ph[1], ph[3], (zf * w).mean(2)]
    return torch.cat(fs, 1)
def fwd(idx, k):
    with torch.autocast("cuda", dtype=torch.bfloat16): z = blk(H[k][idx].float())
    return feat(z, DM[k][idx])
@torch.no_grad()
def feats8():
    out = torch.zeros(N, 11 * 512, device=dev)
    for k in range(8):
        for i in range(0, N, 64): out[i:i + 64] += fwd(slice(i, i + 64), k) / 8
    return out.double()
def design(Fm, t):
    """폴드 train(t) 통계로 표준화·절단한 뒤 묶음별 상대 분산 배율 → 커널 K (n×n). Fm: (n, 11·C) double"""
    C = Fm.shape[1] // 11; K = 0
    for blocks, g in GR:
        cols = torch.cat([torch.arange(b * C, (b + 1) * C, device=Fm.device) for b in blocks]); B = Fm[:, cols]
        mu, sd = B[t].mean(0).detach(), B[t].std(0).detach(); keep = sd > 1e-6
        Z = ((B[:, keep] - mu[keep]) / sd[keep]).clamp(-6, 6); K = K + g * (Z @ Z.T) / keep.sum()
    return K
def ridge_pred(K, yv, s, q, lam):
    """지지 집합 s로 맞춘 커널 Ridge의 질의 집합 q 예측 (절편 = 지지 평균, 커널은 지지 평균으로 중심화)"""
    Kss, Kqs = K[s][:, s], K[q][:, s]; cm = Kss.mean(0, keepdim=True); mm = Kss.mean()
    Kc = Kss - cm - cm.T + mm; Kq = Kqs - cm - Kqs.mean(1, keepdim=True) + mm
    al = torch.linalg.solve(Kc + lam * torch.eye(len(s), device=K.device, dtype=K.dtype), yv[s] - yv[s].mean())
    return Kq @ al + yv[s].mean()
yt = torch.tensor(y, device=dev, dtype=torch.float64)
init_state = {k: v.detach().clone() for k, v in blk.state_dict().items()}
def run_fold(FOLD):
    global t0
    blk.load_state_dict(init_state)
    tr_idx = np.where(fold != FOLD)[0]; va_idx = np.where(fold == FOLD)[0]
    TR, VA = torch.tensor(tr_idx, device=dev), torch.tensor(va_idx, device=dev)
    LAMS = np.logspace(-6, 3, 60)
    def pick_lam(K):
        """폴드 train 내부 LOO로 λ(대각 평균 대비) 선택"""
        Kt = K[TR][:, TR]; cm = Kt.mean(0, keepdim=True); Kc = Kt - cm - cm.T + Kt.mean(); ev, Q = torch.linalg.eigh(Kc); ev = ev.clamp_min(0)
        yc = yt[TR] - yt[TR].mean(); Qy = Q.T @ yc; sc = Kc.diagonal().mean(); best = None
        for l in LAMS:
            lam = l * sc; al = Q @ (Qy / (ev + lam)); d = (Q ** 2) @ (1 / (ev + lam)) - 1 / (lam * len(TR)); e = ((al / d) ** 2).mean().item()
            if best is None or e < best[0]: best = (e, l)
        return best[1], best[0] ** 0.5, sc.item()
    hist = []; LAM0 = []
    def evaluate(step):
        Fm = feats8(); K = design(Fm, TR); l, loo, sc = pick_lam(K); p = ridge_pred(K, yt, TR, VA, l * sc)
        if step == 0: LAM0.append(l)
        pf = ridge_pred(K, yt, TR, VA, LAM0[0] * sc); rf = ((pf - yt[VA]) ** 2).mean().sqrt().item()
        r = ((p - yt[VA]) ** 2).mean().sqrt().item(); drift = (sum(((p_ - i_) ** 2).sum() for p_, i_ in zip(tun, init)) / sum((i_ ** 2).sum() for i_ in init)).sqrt().item()
        hist.append((step, r, loo, p.cpu().numpy(), pf.cpu().numpy())); print(f"  스텝 {step:4d}: 폴드 밖 RMSE {r:.3f} (λ 고정 {rf:.3f}) | train 내부 LOO {loo:.3f} | λ/대각 {l:.2e} | 가중치 이동 {drift:.4f} | {time.time() - t0:.0f}s", flush=True)
        return l
    if a.model == "densenet121.tv_in1k":   # 검산: 저장된 고정 특징(probe_models, phase2m 8보기)과 같은지
        ref = np.load(ROOT / f"outputs/probe2/cnn_densenet121.tv_in1k_s2_512_Pphase2m_T8_S{S48}/emb.npz")["train"].astype(np.float64); cur = feats8().cpu().numpy()
        print("  저장 특징과의 상관 %.6f, 상대 오차 %.4f" % (np.corrcoef(ref.ravel(), cur.ravel())[0, 1], np.abs(ref - cur).mean() / np.abs(ref).mean()), flush=True)
    t0 = time.time(); lam_rel = evaluate(0) * a.lam_mult
    opt = torch.optim.AdamW(tun, lr=a.lr, weight_decay=0.0); n = len(tr_idx)
    for step in range(1, a.steps + 1):
        ks = rng.choice(8, a.views, replace=False)
        with torch.no_grad():
            Fm = torch.zeros(n, 11 * 512, device=dev)
            for k in ks:
                for i in range(0, n, 64): Fm[i:i + 64] += fwd(TR[i:i + 64], int(k)) / a.views
        Fm = Fm.double().requires_grad_(True)
        allt = torch.arange(n, device=dev); K = design(Fm, allt); sc = K.diagonal().mean().detach()
        perm = torch.tensor(rng.permutation(n), device=dev); loss = 0
        for j in range(5):
            q = perm[j::5]; msk = torch.ones(n, dtype=torch.bool, device=dev); msk[q] = False; s = allt[msk]
            loss = loss + ((ridge_pred(K, yt[TR], s, q, lam_rel * sc) - yt[TR][q]) ** 2).sum() / n
        g, = torch.autograd.grad(loss, Fm); g = g.float()
        opt.zero_grad(set_to_none=True)
        for k in ks:
            for i in range(0, n, a.chunk):
                (fwd(TR[i:i + a.chunk], int(k)) * g[i:i + a.chunk] / a.views).sum().backward()
        if a.l2sp > 0:
            for p_, i_ in zip(tun, init): p_.grad.add_(2 * a.l2sp * (p_.detach() - i_))
        torch.nn.utils.clip_grad_norm_(tun, 1.0 * sum(p.numel() for p in tun) ** 0.5)     # 폭주 방지용(사실상 비활성)
        opt.step()
        if step % 5 == 0: print(f"    step {step} 내부 CV RMSE {loss.item() ** 0.5:.3f}", flush=True)
        if step % a.eval_every == 0: evaluate(step)
    return va_idx, hist
folds = range(5) if a.fold < 0 else [a.fold]; OOFS = {}; OOFF = {}
for FOLD in folds:
    print(f"== 폴드 {FOLD} ==", flush=True); va_idx, hist = run_fold(FOLD)
    for h in hist:
        OOFS.setdefault(h[0], np.full(N, np.nan))[va_idx] = h[3]; OOFF.setdefault(h[0], np.full(N, np.nan))[va_idx] = h[4]
tag = f"{a.model.replace('.', '-')}_f{a.fold}_{a.tune}_lr{a.lr:g}_v{a.views}{a.tag}"
np.savez(ROOT / f"analysis/metaft_{tag}.npz", **{f"oof{k}": v for k, v in OOFS.items()}, **{f"oofF{k}": v for k, v in OOFF.items()})
msk = ~np.isnan(OOFS[0]); b = OOFS[0][msk]; yy = y[msk]; ff = fold[msk]
print(f"— 요약 ({msk.sum()}장) — 기준(스텝 0) RMSE {np.sqrt(((b - yy) ** 2).mean()):.3f}")
for k in sorted(OOFS):
    if k == 0: continue
    for nm, O in (("λ 재선택", OOFS), ("λ 고정", OOFF)):
        o = O[k][msk]; d = (o - yy) ** 2 - (b - yy) ** 2
        print(f"  스텝 {k:3d} [{nm}] RMSE {np.sqrt(((o - yy) ** 2).mean()):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(len(d)):.2f} | 폴드 개선 {sum(d[ff == q].mean() < 0 for q in np.unique(ff))}/{len(np.unique(ff))} | 기준과 평균 {np.sqrt((((o + b) / 2 - yy) ** 2).mean()):.3f}")
print("DONE", tag)
