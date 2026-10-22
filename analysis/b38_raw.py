"""디노이즈 전 원본 이미지 기반 예전 구성원을 v28a에 더했을 때 (중첩 비음수 가중, P 고정) — 디노이저가 지운 정보가 남아 있는지."""
import sys, io, contextlib; sys.path.insert(0, "analysis"); sys.argv = ["b34.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b34.py", encoding="utf-8").read())
sys.path.insert(0, "src"); from make_submission import V7_CNN
W = PRE2["v28 + 재조정 0.5"]; o28 = bl(W, 0, 1.03)
P2 = "outputs/probe2/"
new = {"raw_het3_448": (oof(P2 + "vit_large_patch14_dinov2.lvd142m_448_L13-15-17-19_Phetero3"), None),
       "raw_cnn9": (np.mean([oof(d) for d in V7_CNN], 0), None),
       "raw_dinoL12": (oof(P2 + "vit_large_patch14_dinov2.lvd142m_518_L1-3-5-7-9-11-13-15-17-19-21-23_T8"), None)}
for n, (o_, _) in new.items():
    rc = np.corrcoef(o_ - y, o28 - y)[0, 1]
    o, Wf = nested_fixP({**W, n: 0.0}) if False else (None, None)
    G[n] = (o_, np.zeros(1000)); b30.G[n] = G[n]
    o, Wf = nested_fixP({**W, n: 0.05})
    print(f"  {n:13s} 단독 {rm(o_):.3f} | v28a와 잔차상관 {rc:.3f} | 중첩 가중 적합 OOF {rm(o):.3f} (v28a 구성원만 {rm(o28_nested):.3f}) | 전체 적합 비중 {Wf[n]:.3f}" if False else
          f"  {n:13s} 단독 {rm(o_):.3f} | v28a와 잔차상관 {rc:.3f} | 중첩 가중 적합 OOF {rm(o):.3f} | 전체 적합 비중 {Wf[n]:.3f}", flush=True)
o_base, _ = nested_fixP(W); print(f"  (비교) v28a 구성원만 중첩 가중 적합 OOF {rm(o_base):.3f}")
