import sys; sys.path.insert(0, "analysis")
from blend_eval import *
a = oof(E + "dnpp0.25_ff48"); b = oof(E + "dnpp0.25_rawg2"); lq = sig >= 10
rep("기존 (디노이즈 σ48)", a); rep("거친 보기 (원본 가우시안 σ2)", b, a)
print("잔차상관 전체 %.3f, 저품질 %.3f, 고품질 %.3f" % (np.corrcoef(a - y, b - y)[0, 1], np.corrcoef((a - y)[lq], (b - y)[lq])[0, 1], np.corrcoef((a - y)[~lq], (b - y)[~lq])[0, 1]))
for w in (0.2, 0.35, 0.5): rep(f"전체 혼합 {w}", (1 - w) * a + w * b, a)
for w in (0.3, 0.5, 0.7):
    o = a.copy(); o[lq] = (1 - w) * a[lq] + w * b[lq]; rep(f"저품질만 혼합 {w}", o, a)
