"""방향별 디노이즈 특징: D4 방향 k마다 디노이저 출력 model(d4(x,k))을 평균하지 않고 그대로(σ≥10은 조명 보정 σ48) DINOv2 보기 k로 사용.
'평균 이미지의 특징' 대신 '특징의 평균'. 산출물: outputs/probe2_views/<stem>/<split>_s0_k<k>.npy (view_embed와 같은 형식)
사용: python perview.py --weights weights/denoiser_drt10b_i10000.pt --name unet_drt10b_i10000_pv   |  --drunet --name drunet_pv"""
import argparse, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np, pandas as pd, torch
from PIL import Image
from train_cnn import d4
from flatfield_cache import correct
from view_embed import embed
from probe_models import build
ap = argparse.ArgumentParser(); ap.add_argument("--weights"); ap.add_argument("--drunet", action="store_true"); ap.add_argument("--name", required=True)
ap.add_argument("--splits", nargs="+", default=["train"]); ap.add_argument("--ff", type=float, default=48)
a = ap.parse_args(); torch.backends.cudnn.benchmark = True
if a.drunet:
    from drunet_net import load; net = load(ROOT / "weights/pretrained_denoiser/drunet_gray.pth")
else:
    from train_denoiser import UNet; net = UNet((48, 96, 192, 384)).cuda(); net.load_state_dict(torch.load(ROOT / a.weights, map_location="cuda")); net.eval()
m, mean, std = build("vit_large_patch14_dinov2.lvd142m", 448); d = ROOT / "outputs/probe2_views" / f"{a.name}_fflowpass{int(a.ff)}_low"; d.mkdir(parents=True, exist_ok=True); t0 = time.time()
for sp in a.splits:
    ids = pd.read_csv(ROOT / "data" / ("train.csv" if sp == "train" else "sample_submission.csv")).ID
    sg = pd.read_csv(ROOT / f"outputs/features/{sp}_features.csv").set_index("ID").loc[ids, "q_noise_sigma"].values
    X = torch.from_numpy(np.stack([np.array(Image.open(ROOT / f"data/{sp}/{i}.png").convert("L")) for i in ids])).unsqueeze(1)
    for k in range(8):
        p = d / f"{sp}_s0_k{k}.npy"
        if p.exists(): continue
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 16):
                xb = d4(X[i:i + 16].cuda().float().div(255), k)
                if a.drunet:
                    nm = torch.tensor(sg[i:i + 16] / 255, dtype=torch.float32, device="cuda").view(-1, 1, 1, 1).expand(-1, 1, *xb.shape[-2:]); z = net(torch.cat([xb, nm], 1))
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16): z = net(xb).float()
                outs.append((z.clamp(0, 1) * 255).round().byte().squeeze(1).cpu().numpy())
        Z = np.concatenate(outs); Z = np.stack([correct(im, "lowpass", a.ff) if s >= 10 else im for im, s in zip(Z, sg)])
        np.save(p, embed(m, mean, std, torch.from_numpy(Z).unsqueeze(1), 448, [11, 13, 15, 17], 8, (0, 0), 0))
        print(f"[{sp} k{k}] {time.time() - t0:.0f}s", flush=True)
print("DONE")
