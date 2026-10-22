"""라벨 잡음 상한: 서로 다른 모델 계열의 OOF 잔차 공분산. 라벨 잡음 ε은 모든 모델 잔차에 공통으로 들어가므로
Cov(r_A, r_B) >= Var(ε) (모델 오차끼리 음의 상관이 아니면). 가장 이질적인 쌍의 공분산이 상한에 가깝다."""
import sys, glob, os; sys.path.insert(0, "analysis")
from blend_eval import *
co, _ = grp(CNN); qo, _ = grp([E + f"dnpp0.25_ff{s}" for s in FF]); po = oof(E + "mr_v14_pp_sc")
v15 = np.load("analysis/v15a_oof.npy")
M = {"v15a": v15, "DINO+pp": qo, "P(측정)": po, "CNN9": co}
# 다른 계열 후보: 수작업 특징 Ridge, 자기지도, 원본 DINO 등 oof.csv가 있는 폴더 탐색
cands = {}
for pat in ("outputs/probe/*/oof.csv", "outputs/emb_reg/*/oof.csv", "outputs/cnn/*/oof.csv", "outputs/*/oof.csv", "outputs/probe2/*/oof.csv"):
    for f in glob.glob(pat):
        d = os.path.dirname(f).replace("\\", "/")
        try:
            df = pd.read_csv(f)
            col = "pred" if "pred" in df.columns else [c for c in df.columns if c not in ("ID", "fold", "hardness")][0]
            p = df.set_index("ID").loc[ids, col].values
            if np.isfinite(p).all(): cands[d] = p
        except Exception as e: pass
print("OOF 폴더 수:", len(cands))
r15 = v15 - y
rows = []
for d, p in cands.items():
    r = p - y
    rows.append((d.replace("outputs/", "")[:70], np.sqrt((r ** 2).mean()), np.corrcoef(r, r15)[0, 1], np.cov(r, r15)[0, 1]))
t = pd.DataFrame(rows, columns=["model", "rmse", "rescorr_v15a", "cov_v15a"]).sort_values("cov_v15a")
print("\n== v15a 잔차와 공분산이 가장 작은 모델 (rmse<15) ==")
print(t[t.rmse < 15].head(25).to_string(index=False))
print("\n== 주요 블록 간 잔차 공분산의 제곱근(=공통 성분 크기) ==")
ks = list(M)
for i in ks:
    print(f"{i:10s}", " ".join(f"{np.sqrt(max(np.cov(M[i]-y, M[j]-y)[0,1], 0)):6.2f}" for j in ks))
# 모드별
for nm, m in (("고품질", sig < 10), ("저품질", sig >= 10)):
    print(f"\n[{nm}] 공통성분 sqrt(Cov): CNN9-v15a {np.sqrt(np.cov((co-y)[m], (v15-y)[m])[0,1]):.2f}, CNN9-P {np.sqrt(np.cov((co-y)[m], (po-y)[m])[0,1]):.2f}, v15a RMSE {np.sqrt(((v15-y)[m]**2).mean()):.2f}")
