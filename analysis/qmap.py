"""모드 정렬 시험: 저품질(σ≥10) 이미지의 측정 특징을 열별 분위 대응으로 고품질 분포에 맞춘다 (train의 두 모드 분포만 사용, 라벨 미사용)."""
import numpy as np, pandas as pd, sys
fc = "outputs/features_cache/"; S = sys.argv[1]; PRES = [p.format(S=S) for p in sys.argv[2:]]
sg = {sp: pd.read_csv(f"outputs/features/{sp}_features.csv").set_index("ID").q_noise_sigma for sp in ("train", "test")}
for pre in PRES:
    d = {sp: pd.read_csv(f"{fc}{pre}_{sp}.csv").set_index("ID") for sp in ("train", "test")}
    lqt = (sg["train"].loc[d["train"].index] >= 10).values
    out = {sp: d[sp].copy() for sp in d}
    for c in d["train"].columns:
        if not np.issubdtype(d["train"][c].dtype, np.number) or c.startswith("q_"): continue
        a = np.sort(d["train"][c].values[lqt]); b = np.sort(d["train"][c].values[~lqt])
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 50 or len(b) < 50 or a[0] == a[-1]: continue
        qa = (np.arange(len(a)) + .5) / len(a); qb = (np.arange(len(b)) + .5) / len(b)
        for sp in d:
            lq = (sg[sp].loc[d[sp].index] >= 10).values; v = d[sp][c].values[lq]
            out[sp].loc[lq, c] = np.interp(np.interp(v, a, qa), qb, b)
    for sp in d: out[sp].reset_index().to_csv(f"{fc}{pre}_qm_{sp}.csv", index=False)
print("ok")
