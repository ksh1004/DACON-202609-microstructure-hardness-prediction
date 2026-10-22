import sys; sys.path.insert(0, "analysis")
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from b30 import *
F = pd.read_csv("outputs/features/train_features.csv").set_index("ID").loc[pd.read_csv("data/train.csv").ID]
T = pd.read_csv("outputs/features/test_features.csv")
o, _ = show("v25a", W25); r = o - y
print(f"{'σ구간':8s} {'n':>4s} {'정답평균':>7s} {'std':>5s} {'잔차평균':>7s} {'RMSE':>6s} {'blur_dn':>8s} {'contrast':>8s} {'q_std':>6s} | test n")
for lo, hi in ((0, 5), (5, 7), (7, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 99)):
    m = (sig >= lo) & (sig < hi); mt = (T.q_noise_sigma >= lo) & (T.q_noise_sigma < hi)
    print(f"{lo:2d}~{hi:2d}   {m.sum():4d} {y[m].mean():7.1f} {y[m].std():5.1f} {r[m].mean():+7.2f} {np.sqrt((r[m] ** 2).mean()):6.2f} {F.q_blur_dn.values[m].mean():8.3f} {F.q_contrast.values[m].mean():8.3f} {F.q_std.values[m].mean():6.2f} | {mt.sum()}")
m = (sig >= 10) & (sig < 14)
print("σ10~14 상관: 잔차 vs 정답", np.corrcoef(r[m], y[m])[0, 1].round(3), "| 기울기(예측~정답)", np.polyfit(y[m], o[m], 1)[0].round(3), "| 다른 저품질 기울기", np.polyfit(y[sig >= 14], o[sig >= 14], 1)[0].round(3), "| 고품질", np.polyfit(y[sig < 10], o[sig < 10], 1)[0].round(3))
