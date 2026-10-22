"""CNN 회귀 학습 (K-Fold) — 노트북(RTX 5060, Windows)과 Colab(T4, Linux) 겸용.

예) python src/train_cnn.py --model resnet50.a1_in1k --epochs 30 --exp r50_base
산출물: outputs/cnn/<exp>/{oof.csv, test_pred.csv, log.json}

설계 메모
- 데이터 500장(+test 1000장)은 uint8 텐서로 GPU에 상주 → DataLoader/워커 없음(Windows spawn 오버헤드 회피)
- 증강은 GPU 배치 연산: D4(반전·90°회전), 가우시안 노이즈, 블러, 약한 밝기/대비 (EDA 근거: 방향·품질은 경도와 무관)
  스케일/전단/탄성 변형은 결정립 크기·정렬도(경도 신호)를 바꾸므로 사용하지 않음
- 정밀도: bf16 지원 GPU(Ampere+)는 bf16, 아니면(T4) fp16+GradScaler
- 메모리 포맷: 기본 NCHW. RTX 5060/Windows에서 channels_last + 학습모드 BN이 ~25배 느림 (T4에서는 --channels-last 권장)
- 폴드: outputs/folds.csv 고정 (경도 10분위 층화) → 모든 모델이 공유해 스태킹 시 누수 방지
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torchvision.transforms.functional import gaussian_blur

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"


# ----------------------------------------------------------------------------- 데이터
def load_split(split: str, ids) -> torch.Tensor:
    arr = np.stack([np.array(Image.open(DATA / split / f"{i}.png")) for i in ids])
    return torch.from_numpy(arr).unsqueeze(1)  # N,1,H,W uint8


def get_folds(train: pd.DataFrame, n_folds=5, seed=42) -> np.ndarray:
    path = OUT / "folds.csv"
    if path.exists():
        f = pd.read_csv(path)
        assert (f.ID.values == train.ID.values).all()
        return f.fold.values
    bins = pd.qcut(train.hardness, 10, labels=False)
    fold = np.zeros(len(train), dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(train, bins)):
        fold[va] = k
    OUT.mkdir(exist_ok=True)
    pd.DataFrame({"ID": train.ID, "fold": fold}).to_csv(path, index=False)
    return fold


# ----------------------------------------------------------------------------- 증강
def d4(x: torch.Tensor, k: int) -> torch.Tensor:
    if k & 4:
        x = x.transpose(-1, -2)
    return torch.rot90(x, k & 3, dims=(-2, -1))


def random_crop(x: torch.Tensor, c: int) -> torch.Tensor:
    """샘플마다 다른 위치에서 c×c 부분 영역을 자른다 (확대·축소 없음 → 결정립 크기 보존).

    경도는 정상(stationary) 미세구조의 전역 통계이므로 부분 영역도 같은 정답을 공유한다고 보고 학습한다 (타일 학습).
    """
    B, _, H, W = x.shape
    if c >= H:
        return x
    ys = torch.randint(0, H - c + 1, (B,)).tolist()
    xs = torch.randint(0, W - c + 1, (B,)).tolist()
    return torch.stack([x[i, :, ys[i]:ys[i] + c, xs[i]:xs[i] + c] for i in range(B)])


def resize(x: torch.Tensor, a) -> torch.Tensor:
    """입력 업샘플링: 원본 256 기준 a.img_size/256 배율 (크롭이어도 같은 배율 → 전체 이미지 추론과 스케일 일치)."""
    scale = a.img_size / 256
    if scale == 1:
        return x
    size = int(round(x.shape[-1] * scale))
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def rotate_arbitrary(x: torch.Tensor, p: float) -> torch.Tensor:
    """확률 p로 임의 각도(0~360°) 회전. 모서리는 reflection 패딩, bilinear 보간(약한 평활화 동반)."""
    m = torch.rand(x.shape[0], device=x.device) < p
    if not m.any():
        return x
    ang = torch.rand(int(m.sum()), device=x.device) * (2 * math.pi)
    theta = torch.zeros(len(ang), 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 1] = torch.cos(ang), -torch.sin(ang)
    theta[:, 1, 0], theta[:, 1, 1] = torch.sin(ang), torch.cos(ang)
    xm = x[m]
    grid = F.affine_grid(theta, list(xm.shape), align_corners=False)
    x = x.clone()
    x[m] = F.grid_sample(xm, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    return x


def augment(x: torch.Tensor, a) -> torch.Tensor:
    """x: B,1,H,W float [0,1] (GPU)."""
    B = x.shape[0]
    if not a.no_d4:
        t = torch.randint(0, 8, (B,), device=x.device)
        out = torch.empty_like(x)
        for k in range(8):
            m = t == k
            if m.any():
                out[m] = d4(x[m], k)
        x = out
    if a.rot_p > 0:
        x = rotate_arbitrary(x, a.rot_p)
    if a.blur_p > 0:
        m = torch.rand(B, device=x.device) < a.blur_p
        if m.any():
            s = random.uniform(0.3, 1.0)
            x[m] = gaussian_blur(x[m], kernel_size=5, sigma=s)
    if a.contrast > 0:
        c = 1 + (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * a.contrast
        b = (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * a.brightness
        mu = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mu) * c + mu + b
    if a.noise > 0:
        sig = torch.rand(B, 1, 1, 1, device=x.device) * (a.noise / 255.0)
        x = x + torch.randn_like(x) * sig
    return x.clamp(0, 1)


# ----------------------------------------------------------------------------- 모델
def load_init_weights(model: torch.nn.Module, path: str) -> None:
    """외부 사전학습 가중치(예: NASA MicroNet, MIT 라이선스) 로드.

    - 키 이름이 같으면 이름으로(torchvision ResNet → timm ResNet),
      다르면 순서+모양 일치로(efficientnet_pytorch → timm tf_efficientnet) 매핑한다.
    - 분류기(head)는 제외하고, 3채널 첫 합성곱은 채널 합으로 1채널에 맞춘다 (timm의 in_chans=1 처리와 동일).
    """
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    head = ("_fc.", "fc.", "classifier.", "head.")
    src = [(k, v) for k, v in sd.items() if not k.startswith(head)]
    msd = model.state_dict()
    dst = [k for k in msd if not k.startswith(head)]
    if all(k in msd for k, _ in src):
        pairs = [(k, v) for k, v in src]
    else:
        assert len(src) == len(dst), f"파라미터 수 불일치: {len(src)} vs {len(dst)}"
        pairs = [(d, v) for (_, v), d in zip(src, dst)]
    new = {}
    for k, v in pairs:
        t = msd[k]
        if v.ndim == 4 and v.shape[1] == 3 and t.shape[1] == 1:
            v = v.sum(dim=1, keepdim=True)
        assert tuple(v.shape) == tuple(t.shape), f"모양 불일치 {k}: {tuple(v.shape)} vs {tuple(t.shape)}"
        new[k] = v
    missing = [k for k in msd if k not in new]
    model.load_state_dict(new, strict=False)
    print(f"  init weights <- {Path(path).name}: {len(new)} tensors (미로드 {len(missing)}개: {missing[:3]})", flush=True)


class HeteroHeadNet(torch.nn.Module):
    """백본 마지막 특징맵(B,C,H,W) → [채널별 평균·표준편차·2×2/4×4 영역평균의 영역 간 표준편차] → LayerNorm → 선형.
    전역 평균 풀링이 버리는 '위치별 미세구조 차이(불균일성)'를 회귀에 쓰게 한다 (DINOv2 hetero 풀링과 같은 요약)."""

    def __init__(self, backbone: torch.nn.Module, drop: float):
        super().__init__()
        self.backbone = backbone
        c = backbone.num_features
        self.norm = torch.nn.LayerNorm(4 * c)
        self.drop = torch.nn.Dropout(drop)
        self.fc = torch.nn.Linear(4 * c, 1)

    def set_grad_checkpointing(self, enable: bool = True):
        self.backbone.set_grad_checkpointing(enable)

    def forward(self, x):
        z = self.backbone.forward_features(x).float()
        f = [z.mean((2, 3)), z.flatten(2).std(2)]
        for g in (2, 4):
            f.append(F.adaptive_avg_pool2d(z, g).flatten(2).std(2))
        return self.fc(self.drop(self.norm(torch.cat(f, 1))))


def build_model(a, k: int | None = None) -> torch.nn.Module:
    if getattr(a, "head", "gap") == "hetero":
        a2 = argparse.Namespace(**{**vars(a), "head": "gap"})
        bb = build_model(a2, k)
        bb.reset_classifier(0)
        return HeteroHeadNet(bb, a.drop)
    kw = dict(pretrained=not a.scratch and not a.init_weights, num_classes=1, in_chans=1, drop_rate=a.drop)
    if "vit" in a.model or "deit" in a.model:
        kw["img_size"] = 256
    try:
        m = timm.create_model(a.model, drop_path_rate=a.drop_path, **kw)
    except TypeError:
        m = timm.create_model(a.model, **kw)
    if a.init_weights:
        # "{k}" 자리표시자 → 폴드별 사전학습 가중치 (예: 폴드 k의 학습 이미지로만 자기지도 사전학습한 백본)
        path = a.init_weights.replace("{k}", str(k)) if k is not None else a.init_weights
        load_init_weights(m, path)
    return m


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 1e-6 + (base - 1e-6) * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def predict(model, X: torch.Tensor, mean, std, a, tta=True, bs=64) -> np.ndarray:
    model.eval()
    preds = []
    ks = range(8) if tta else [0]
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(a.device).float().div_(255)
        p = 0
        for k in ks:
            xi = (resize(d4(xb, k), a) - mean) / std
            if a.channels_last:
                xi = xi.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=a.amp_dtype, enabled=a.amp_dtype != torch.float32):
                p = p + model(xi).float().squeeze(1)
        preds.append((p / len(ks)).cpu())
    return torch.cat(preds).numpy()


def train_fold(k, Xtr, ytr, Xva, yva, Xte, a, attempt=0):
    seed = a.seed * 100 + k + 7919 * attempt
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    ymu, ysd = ytr.mean(), ytr.std()
    mean = (Xtr.float().mean() / 255).item()
    std = (Xtr.float().std() / 255).item()
    model = build_model(a, k).to(a.device)
    if a.grad_ckpt:
        model.set_grad_checkpointing(True)  # 큰 모델 × 512 입력에서 VRAM 절약 (결과 동일, 속도 약간 감소)
    if a.channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    scaler = torch.amp.GradScaler(enabled=a.amp_dtype == torch.float16)
    Xtr_g = Xtr.to(a.device)
    yt = torch.tensor((ytr - ymu) / ysd, dtype=torch.float32, device=a.device)
    n = len(Xtr_g)
    spe = n // a.batch  # drop_last
    total, warm = spe * a.epochs, spe * a.warmup
    step, hist = 0, []
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(n, device=a.device)
        tl = 0.0
        for b in range(spe):
            idx = perm[b * a.batch:(b + 1) * a.batch]
            x = augment(Xtr_g[idx].float().div_(255), a)
            if a.crop:
                x = random_crop(x, a.crop)
            x = (resize(x, a) - mean) / std
            if a.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, warm, a.lr)
            with torch.autocast("cuda", dtype=a.amp_dtype, enabled=a.amp_dtype != torch.float32):
                out = model(x).float().squeeze(1)
            loss = F.mse_loss(out, yt[idx]) if a.loss == "mse" else F.huber_loss(out, yt[idx], delta=a.huber_delta)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            tl += loss.item()
            step += 1
        if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
            pv = predict(model, Xva, mean, std, a, tta=False) * ysd + ymu
            r = float(np.sqrt(np.mean((pv - yva) ** 2)))
            hist.append({"epoch": ep + 1, "train_loss": tl / spe, "val_rmse": r})
            print(f"  fold{k} ep{ep + 1:3d} loss {tl / spe:.4f} val_rmse(noTTA) {r:.3f}", flush=True)
    pv = predict(model, Xva, mean, std, a, tta=True) * ysd + ymu
    pt = predict(model, Xte, mean, std, a, tta=True) * ysd + ymu
    if a.save_weights:
        wdir = OUT / "cnn" / a.exp / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "ymu": ymu, "ysd": ysd},
                   wdir / f"fold{k}.pt")
    del model, opt, Xtr_g
    torch.cuda.empty_cache()
    return pv, pt, hist


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50.a1_in1k")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--head", choices=["gap", "hetero"], default="gap",
                    help="gap: timm 기본 전역평균 헤드 / hetero: 평균·표준편차·영역 불균일성 풀링 헤드")
    ap.add_argument("--loss", choices=["mse", "huber"], default="mse")
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=8.0, help="추가 가우시안 노이즈 최대 σ (0~255 스케일)")
    ap.add_argument("--blur-p", type=float, default=0.25)
    ap.add_argument("--contrast", type=float, default=0.1)
    ap.add_argument("--brightness", type=float, default=0.03)
    ap.add_argument("--rot-p", type=float, default=0.0, help="임의 각도 회전 확률 (reflection 패딩)")
    ap.add_argument("--no-d4", action="store_true", help="D4(반전·90°회전) 증강 끄기 (ablation용)")
    ap.add_argument("--warmup", type=int, default=1, help="선형 warmup 에폭 수")
    ap.add_argument("--retry-loss", type=float, default=0.8, help="최종 train loss가 이보다 크면 학습 실패로 보고 재시도")
    ap.add_argument("--max-retry", type=int, default=2)
    ap.add_argument("--grad-ckpt", action="store_true", help="그래디언트 체크포인팅 (VRAM 절약)")
    ap.add_argument("--crop", type=int, default=0, help="학습 시 무작위 부분영역 크기 (예: 128). 추론은 전체 이미지")
    ap.add_argument("--init-weights", default=None,
                    help="외부 사전학습 가중치 경로 (예: weights/micronet/resnet50_imagenet-micronet_weights.pth)")
    ap.add_argument("--img-size", type=int, default=256,
                    help="입력 확대 크기 (예: 512). stride-4 stem에서 1~2px 결정립계 손실을 줄이기 위한 업샘플링")
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="일부 폴드만 실행 (디버그)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--scratch", action="store_true", help="사전학습 없이 학습")
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--save-weights", action="store_true")
    ap.add_argument("--src", default=None, help="입력 이미지 캐시 npz (예: outputs/denoised/unet_*.npz). 없으면 원본 PNG")
    return ap


def finalize_args(a):
    """파싱된 인자에 실행 환경 설정을 채운다 (train_full.py와 공유)."""
    a.exp = a.exp or a.model.replace(".", "_")
    a.device = "cuda"
    if a.dtype == "auto":
        a.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    else:
        a.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    torch.backends.cudnn.benchmark = True
    return a


def main():
    a = finalize_args(build_parser().parse_args())

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "sample_submission.csv")
    fold = get_folds(train)
    if a.src:  # 디노이즈 등 이미지별 전처리 캐시 (train/test: N×H×W uint8, ID 순서 = train.csv / sample_submission.csv)
        z = np.load(ROOT / a.src)
        X, Xte = torch.from_numpy(z["train"]).unsqueeze(1), torch.from_numpy(z["test"]).unsqueeze(1)
    else:
        X = load_split("train", train.ID)
        Xte = load_split("test", test.ID)
    y = train.hardness.values.astype(np.float64)

    out_dir = OUT / "cnn" / a.exp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{a.exp}] {a.model} | {torch.cuda.get_device_name(0)} | amp={a.amp_dtype} | cl={a.channels_last}")
    oof = np.zeros(len(train))
    tpred = np.zeros((5, len(test)))
    logs, t0 = {}, time.time()
    for k in (a.folds if a.folds is not None else range(5)):
        tf = time.time()
        tr, va = fold != k, fold == k
        # 간헐적 학습 실패(손실이 줄지 않고 평균 예측에 머묾) 감지 → 시드 바꿔 재시도.
        # 판단은 '학습 손실'로만 한다 (검증 점수를 쓰면 검증 데이터로 모델을 고르는 셈이 됨)
        for attempt in range(a.max_retry + 1):
            pv, pt, hist = train_fold(k, X[tr], y[tr], X[va], y[va], Xte, a, attempt)
            final_loss = hist[-1]["train_loss"]
            if final_loss <= a.retry_loss or attempt == a.max_retry:
                break
            print(f"  fold{k} 학습 실패 감지: 최종 train loss {final_loss:.3f} > {a.retry_loss} → 재시도 {attempt + 1}",
                  flush=True)
        oof[va] = pv
        tpred[k] = pt
        r = float(np.sqrt(np.mean((pv - y[va]) ** 2)))
        logs[f"fold{k}"] = {"rmse_tta": r, "sec": time.time() - tf, "hist": hist, "retries": attempt}
        print(f"  fold{k} RMSE(TTA) {r:.3f}  ({time.time() - tf:.0f}s)", flush=True)

    done = [int(k[4:]) for k in logs]
    m = np.isin(fold, done)
    rmse = float(np.sqrt(np.mean((oof[m] - y[m]) ** 2)))
    summary = {"exp": a.exp, "model": a.model, "oof_rmse": rmse, "minutes": (time.time() - t0) / 60,
               "device": torch.cuda.get_device_name(0), "args": {k: str(v) for k, v in vars(a).items()},
               "folds": logs}
    fpath = OUT / "features" / "train_features.csv"
    if fpath.exists():
        q = pd.read_csv(fpath).set_index("ID").loc[train.ID, "q_noise_sigma"].values
        for g, gm in [("low_noise", q <= 10), ("high_noise", q > 10)]:
            mm = m & gm
            summary[f"rmse_{g}"] = float(np.sqrt(np.mean((oof[mm] - y[mm]) ** 2)))
    pd.DataFrame({"ID": train.ID, "fold": fold, "hardness": y, "pred": oof}).to_csv(out_dir / "oof.csv", index=False)
    pd.DataFrame({"ID": test.ID, "hardness": tpred[done].mean(0)}).to_csv(out_dir / "test_pred.csv", index=False)
    with open(out_dir / "log.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[{a.exp}] OOF RMSE {rmse:.3f} | low {summary.get('rmse_low_noise', float('nan')):.3f} "
          f"high {summary.get('rmse_high_noise', float('nan')):.3f} | {summary['minutes']:.1f} min")


if __name__ == "__main__":
    main()
