"""노이즈 균일화 디노이즈: 모든 이미지를 같은 노이즈 수준 σ_c 에서 출발시켜 디노이즈.
원본 추정 노이즈 σ_i < σ_c 인 이미지는 N(0, σ_c² − σ_i²) 를 더한 뒤(고정 시드) 부트스트랩 디노이저(D4 평균)에 넣는다 → 잔여 노이즈·질감이 이미지 간에 균일해짐.
K개의 서로 다른 노이즈 실현을 각각 디노이즈해 저장(특징 평균용). 저품질(σ_i ≥ 10)은 기존처럼 σ48 저역 조명 보정.
이미지별 고정 규칙(라벨·다른 이미지 미사용). 사용: python eqnoise.py --sc 12 --K 2 [--splits train test]
산출물: outputs/denoised/eqn<σ_c>_k<k>_fflowpass48_low.npz"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_cnn import DATA, OUT, load_split  # noqa: E402
from train_denoiser import UNet, denoise  # noqa: E402
from flatfield_cache import correct  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sc", type=float, default=12.0)
ap.add_argument("--K", type=int, default=2)
ap.add_argument("--splits", nargs="+", default=["train"])
ap.add_argument("--ff", type=float, default=48.0)
a = ap.parse_args()
net = UNet().cuda(); net.load_state_dict(torch.load(ROOT / "weights/denoiser_cm7_am22_l1_i10000_boot.pt", map_location="cuda"))
ids = {"train": pd.read_csv(DATA / "train.csv").ID, "test": pd.read_csv(DATA / "sample_submission.csv").ID}
old = np.load(OUT / "denoised/unet_cm7_am22_l1_i10000_boot_fflowpass48_low.npz")
for k in range(a.K):
    out = {}
    for sp in ("train", "test"):
        if sp not in a.splits:
            out[sp] = old[sp]; continue   # 자리 채움(미사용)
        sig = pd.read_csv(OUT / f"features/{sp}_features.csv").set_index("ID").loc[ids[sp], "q_noise_sigma"].values
        X = load_split(sp, ids[sp]).float()                               # N,1,H,W (0~255)
        g = torch.Generator().manual_seed(1000 * k + (0 if sp == "train" else 1))
        add = torch.from_numpy(np.sqrt(np.clip(a.sc ** 2 - sig ** 2, 0, None))).float().view(-1, 1, 1, 1)
        Xn = (X + torch.randn(X.shape, generator=g) * add).clamp(0, 255)
        Dn = denoise(net, Xn).squeeze(1).numpy()
        Dn = np.stack([correct(x, "lowpass", a.ff) if s >= 10 else x for x, s in zip(Dn, sig)]).astype(np.uint8)
        out[sp] = Dn
        print(f"[eqn{a.sc:g} k{k}] {sp}: 노이즈를 더한 이미지 {(add.view(-1) > 0).sum().item()}/{len(sig)}장", flush=True)
    p = OUT / "denoised" / f"eqn{a.sc:g}_k{k}_fflowpass{a.ff:g}_low.npz"
    np.savez_compressed(p, **out); print("->", p.relative_to(ROOT))
