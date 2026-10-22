import sys; sys.path.insert(0, "analysis")
from blend_eval import *
a = oof(E + "dnpp0.25_ff48"); b = oof(E + "dnpp0.25_ff48_reg4")
print("잔차상관 %.3f" % np.corrcoef(a - y, b - y)[0, 1]); rep("기존", a)
for w in (0.2, 0.35, 0.5): rep(f"기존 {1-w:.2f} + reg4 {w}", (1 - w) * a + w * b, a)
