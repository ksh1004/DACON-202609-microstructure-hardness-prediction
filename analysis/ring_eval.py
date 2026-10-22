"""H1 검증: 위치 부류별 토큰 평균(고리·4×4 궤도) 블록을 직접 넣으면 (a) 기존에 더해 이득인가 (b) 편향 제거로 잃은 것을 되찾는가."""
import sys; sys.path.insert(0, "analysis")
import view_eval as VE
from view_eval import *
P = lambda s, k: np.load(D + f"train_s{s}_k{k}_pb.npy").astype(np.float32)
raw8 = np.mean([view(0, k).astype(np.float32) for k in range(8)], 0).reshape(len(y), NL, 11, C)
pb8 = np.mean([P(0, k) for k in range(8)], 0).reshape(len(y), NL, 11, C)
ring = np.load(D + "ring_train.npy").reshape(len(y), NL, 13, C)  # 0-4 고리 평균, 5-7 궤도 평균, 8-12 고리 std
def run(name, base, GR, ref=None):
    X = np.concatenate([base, ring], 2); VE.NB = 24
    o = VE.cvoof(X.reshape(len(y), -1), GR); rep(name, o, ref); sys.stdout.flush(); return o
H, PH, M = (0, 1, 2, 3, 4, 5), (6, 7, 8, 9), (10,)
RM, OB, RS = tuple(range(11, 16)), tuple(range(16, 19)), tuple(range(19, 24))
b = run("기존 8보기", raw8, ((H, 1.0), (PH, 0.25), (M, 0.05)))
for g in (0.1, 0.25, 0.5, 1.0):
    run(f"기존 + 고리 평균 G={g}", raw8, ((H, 1.0), (PH, 0.25), (M, 0.05), (RM, g)), b)
run("기존 + 4×4 궤도 평균 G=0.25", raw8, ((H, 1.0), (PH, 0.25), (M, 0.05), (OB, 0.25)), b)
run("기존 + 고리 평균 0.25 + 고리 std 0.25", raw8, ((H, 1.0), (PH, 0.25), (M, 0.05), (RM, 0.25), (RS, 0.25)), b)
run("기존 + 고리 std 0.25", raw8, ((H, 1.0), (PH, 0.25), (M, 0.05), (RS, 0.25)), b)
p = run("편향 제거 8보기", pb8, ((H, 1.0), (PH, 0.25), (M, 0.05)), b)
for g in (0.25, 1.0):
    run(f"편향 제거 + 고리 평균 G={g}", pb8, ((H, 1.0), (PH, 0.25), (M, 0.05), (RM, g)), b)
run("편향 제거 + 고리 평균 0.5 + 궤도 0.5", pb8, ((H, 1.0), (PH, 0.25), (M, 0.05), (RM, 0.5), (OB, 0.5)), b)
n = run("불균일성 없이 (평균·std + 상풀링 + 기여량)", raw8, (((0, 1), 1.0), (PH, 0.75), (M, 0.15)), b)
run("불균일성 없이 + 고리 평균 1.0", raw8, (((0, 1), 1.0), (PH, 0.75), (M, 0.15), (RM, 1.0)), b)
run("불균일성 없이 + 고리 평균 1 + 궤도 1 + 고리 std 1", raw8, (((0, 1), 1.0), (PH, 0.75), (M, 0.15), (RM, 1.0), (OB, 1.0), (RS, 1.0)), b)
