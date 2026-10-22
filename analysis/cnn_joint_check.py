"""고정 CNN 결합 커널 이득의 성격 점검 (σ48, 3계열 블렌드 D .45 T .25 F .30 기준): 경도 구간별 분해, Public 꼴, 모드별, 예측 폭."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from joint_kernel import krr
J = dict(np.load("analysis/jk_cache.npz")); C = dict(np.load("analysis/cj_cache.npz"))
oD, oT = krr(J["D16"]), krr(J["T16"]); bl = lambda oF: .45 * oD + .25 * oT + .30 * oF
cur = ("densenet121-tv_in1k", "resnet50-a1_in1k", "resnet50-a2_in1k"); solo = {n: krr(C[n]) for n in cur + ("densenetblur121d-ra_in1k",)}
base = bl(np.mean([solo[n] for n in cur], 0))
wg = lambda s: np.exp(-0.5 * ((y - y.mean()) / s) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2); w166, w18 = wg(16.6), wg(18.0)
rm = lambda o, m=None, w=None: np.sqrt(np.average(((o - y) ** 2) if m is None else ((o - y)[m] ** 2), weights=w))
Lq = sig >= 10; q = np.quantile(y, [.1, .9]); mid = (y >= q[0]) & (y < q[1])
def sl(p): c_ = np.cov(y, p); return c_[0, 1] / c_[1, 1]
def show(nm, oF):
    o = bl(oF); d = (o - y) ** 2 - (base - y) ** 2
    print(f"{nm:34s} F {rm(oF):.3f} | 블렌드 OOF {rm(o):.3f} Public꼴 {rm(o, w=w166):.3f} Private꼴 {rm(o, w=w18):.3f} pw13 {rm(o, w=w13):.3f} | 고 {rm(o, ~Lq):.3f} 저 {rm(o, Lq):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f} 폴드 {sum(d[fold == k].mean() < 0 for k in range(5))}/5 | 가운데80% 기여 {d[mid].sum() / 500:+.2f} 양끝 {d[~mid].sum() / 500:+.2f} | 기울기 {sl(o):.3f} | 예측 RMS차 {np.sqrt(((o - base) ** 2).mean()):.2f}")
show("현재: 3개 예측 평균", np.mean([solo[n] for n in cur], 0))
show("3개 결합 커널", krr(sum(C[n] for n in cur) / 3))
show("4개 결합 (+ densenetblur121d)", krr(sum(C[n] for n in cur + ("densenetblur121d-ra_in1k",)) / 4))
show("DenseNet 2 + ResNet-50 a1", krr(sum(C[n] for n in ("densenet121-tv_in1k", "densenetblur121d-ra_in1k", "resnet50-a1_in1k")) / 3))
print("— 결합 커널 안에서 모델 가중을 바꾸면 (3개) —")
for w in ((.5, .25, .25), (.4, .4, .2), (.6, .2, .2)): show(f"  dn121 {w[0]} / a1 {w[1]} / a2 {w[2]}", krr(sum(v * C[n] for n, v in zip(cur, w))))
print("— 결합 CNN을 블렌드에서 더 크게 —")
jF = krr(sum(C[n] for n in cur) / 3)
for wd, wt, wf in ((.45, .25, .30), (.40, .25, .35), (.40, .20, .40), (.35, .25, .40)):
    o = wd * oD + wt * oT + wf * jF; print(f"  D {wd} T {wt} F {wf}: OOF {rm(o):.3f} Public꼴 {rm(o, w=w166):.3f} pw13 {rm(o, w=w13):.3f}")
