"""v28a vs v29 후보: Public형·Private형(std 18) 지표와 배율 k별 비교."""
import sys, io, contextlib; sys.path.insert(0, "analysis"); sys.argv = ["b39_da.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b39_da.py", encoding="utf-8").read())
w18 = wg(18.0)
for nm in ("v28a (기준)", "v29b: FoL→(FoL+DAo)/2", "v29b: FoL→(FoL+DAo)/2 + 재조정 0.5", "v29a: FoL→DAo + 재조정 0.5"):
    for k in (1.03, 1.06):
        o = bl(P3[nm], 0, k)
        print(f"  {nm:36s} ×{k:.2f} | OOF {rm(o):.3f} | Public형(16.6) {rm(o, w=w166):.3f} | Private형(18) {rm(o, w=w18):.3f} | 가운데 {rm(o, mid):.3f} 양끝 {rm(o, ~mid):.3f}")
