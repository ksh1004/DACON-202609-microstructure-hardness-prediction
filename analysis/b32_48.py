"""σ48 한 보정만으로 v26 후보 미리 보기 (나머지 σ가 끝나기 전)."""
import sys, io, contextlib; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()):
    from b30 import *
f48 = lambda n: (oof(E + n), tp(E + n))
H = dict(P=G["P"], A=f48("dnpp0.25_ff48_a0.01"), Dj3=f48("mv16j3_drt10b_i10000_ff48_a0.01"), T=f48("tips16_dnppm0.05_ff48_a0.01"), Tx=f48("tips16_drt10b_i10000_ff48_a0.01"),
         J=f48("cnnfj4_ff48_a0.01"), Jx=f48("cnnfj4_drt10b_i10000_ff48_a0.01"), J4=f48("mv16j4_ft1b_ff48_a0.01"), DXF=f48("mv16dxf_ft1b_ff48_a0.01"), DF=f48("mv16df_ft1b_ff48_a0.01"),
         Fo=f48("mv16f_ft1b_ff48_a0.01"), Tf=f48("tips16_ftft1b_ff48_a0.01"))
avg = lambda *ks: tuple(sum(H[k][i] for k in ks) / len(ks) for i in (0, 1))
H["Tm"], H["Jm"], H["Tm3"] = avg("T", "Tx"), avg("J", "Jx"), avg("T", "Tx", "Tf")
print("σ48 구성원: " + " | ".join(f"{n} {rm(H[n][0]):.3f}" for n in H))
blo = lambda w, i, k=1.03: c + k * (sum(v * H[n][i] for n, v in w.items()) - c); b0 = blo(W25, 0)
for nm, w in [("v25a 구성(σ48)", W25)] + [(f"D→{d}" + (", T→Tm3" if t else ""), {**{k: v for k, v in W25.items() if k not in ("Dj3",) + (("Tm",) if t else ())}, d: W25["Dj3"], **({"Tm3": W25["Tm"]} if t else {})}) for d in ("J4", "DXF", "DF", "Fo") for t in (0, 1)]:
    o = blo(w, 0); d = (o - y) ** 2 - (b0 - y) ** 2
    print(f"  {nm:20s} OOF {rm(o):.3f} pw13 {rm(o, w=w13):.3f} Public꼴 {rm(o, w=w166):.3f} | 고 {rm(o, ~Lq):.3f} 저 {rm(o, Lq):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f} {sum(d[fold == f].mean() < 0 for f in range(5))}/5 가운데 {d[mid].sum() / 500:+.2f}")
