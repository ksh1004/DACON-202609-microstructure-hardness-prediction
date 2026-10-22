"""DINOv2-B 16보기(층 3·5·7·9·11) vs L 16보기: 단독 성능과 예측 블렌드 (잡음 없는 조건의 다양성 확인)."""
import sys; sys.path.insert(0, "analysis")
import view_eval as VE
from view_eval import *
L16 = avg(S(ALL, (0, 1))); oL = cvoof(L16); rep("L 16보기", oL)
B16 = np.mean([np.load(D + f"train_s{s}_k{k}_mbasep14.npy").astype(np.float32) for s in (0, 1) for k in range(8)], 0)
VE.NL, VE.C = 5, 768
oB = VE.cvoof(B16); rep("B 16보기 (층 3-11)", oB, oL)
B4 = B16.reshape(len(y), 5, 11, 768)[:, 1:].reshape(len(y), -1); VE.NL = 4
oB4 = VE.cvoof(B4); rep("B 16보기 (층 5-11)", oB4, oL)
print("잔차상관 L-B %.3f" % np.corrcoef(oL - y, oB - y)[0, 1])
for w in (0.1, 0.2, 0.3, 0.5): rep(f"예측 블렌드 L {1-w:.1f} + B {w:.1f}", (1 - w) * oL + w * oB, oL)
