"""편향 제거 후 성능이 떨어진 이유 탐색.
H2: 편향이 있으면 std_q(b+s) ≈ const + var(s)/(2 std b) → 분산 형태. 편향 제거 특징의 불균일성 블록(2~5)을 보기별로 제곱한 뒤 평균해 비교.
또한 블록별로 '기존 것'과 '편향 제거 것'을 섞어 어느 블록에서 손해가 나는지 분해."""
import sys; sys.path.insert(0, "analysis")
from view_eval import *
P = lambda s, k: np.load(D + f"train_s{s}_k{k}_pb.npy").astype(np.float32)
def avg_t(loader, tf=None):
    a = np.zeros((len(y), NL * NB * C), np.float32)
    for k in range(8):
        x = loader(0, k).astype(np.float32).reshape(len(y), NL, NB, C).copy()
        if tf is not None: x = tf(x)
        a += x.reshape(len(y), -1)
    return a / 8
raw8, pb8 = avg_t(view), avg_t(P)
o_raw, o_pb = cvoof(raw8), cvoof(pb8)
rep("8보기 기존", o_raw); rep("8보기 편향 제거", o_pb, o_raw)
def sq(x): x[:, :, 2:6] = x[:, :, 2:6] ** 2; return x
rep("편향 제거 + 불균일성 블록 제곱(분산)", cvoof(avg_t(P, sq)), o_raw)
rep("기존 + 불균일성 블록 제곱", cvoof(avg_t(view, sq)), o_raw)
# 블록 교체 분해: 편향 제거 특징을 기준으로 특정 블록만 기존 것으로 되돌림
R4, P4 = raw8.reshape(len(y), NL, NB, C), pb8.reshape(len(y), NL, NB, C)
names = ["평균", "std", "het2", "het4", "het8", "het4내부std", "2상평균", "2상std", "기지평균", "기지std", "기여량"]
for blocks, nm in (((1,), "std"), ((2,), "het2"), ((3,), "het4"), ((4,), "het8"), ((5,), "het4내부std"), ((2, 3, 4, 5), "het 전부"), ((6, 7, 8, 9, 10), "상풀링·기여량"), ((1, 2, 3, 4, 5), "std+het 전부")):
    X = P4.copy(); X[:, :, list(blocks)] = R4[:, :, list(blocks)]
    rep(f"편향 제거 기준, [{nm}]만 기존으로", cvoof(X.reshape(len(y), -1)), o_raw); sys.stdout.flush()
# 둘 다 넣기: 기존 het + 편향 제거 het (별도 묶음)
X = np.concatenate([R4, P4[:, :, 2:6]], 2)  # NB=15: 0..10 기존, 11..14 편향 제거 het
import view_eval as VE
VE.NB = 15
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05), ((11, 12, 13, 14), 0.5))
rep("기존 + 편향 제거 het 블록 추가(G=0.5)", VE.cvoof(X.reshape(len(y), -1), GR), o_raw)
