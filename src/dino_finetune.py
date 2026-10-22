"""DINOv2 지도 미세조정 (LP-FT): 고정 특징 Ridge 해로 헤드를 초기화한 뒤 낮은 학습률로 전체 조정.

1) 폴드 k 학습 이미지의 고정 다층 특징(518 입력, 층별 평균·표준편차, D4 4방향 평균)으로 RidgeCV 적합
2) 헤드 = [특징 표준화(고정 버퍼) → Linear(Ridge 계수)] → 시작 성능 = 고정 특징 프로브와 동일
3) 백본(층별 학습률 감쇠) + 헤드를 낮은 학습률로 미세조정, 증강은 CNN과 동일(경도 보존 변형만)
4) 검증·test 예측: D4 4방향 TTA

예) python src/dino_finetune.py --model vit_base_patch14_dinov2.lvd142m --layers 0 1 2 3 4 5 6 7 8 9 10 11 --folds 0 1 --exp dinoB_ft
산출물(5폴드 완료 시): outputs/emb_reg/<exp>/{oof.csv, test_pred.csv}
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_models import pool  # noqa: E402
from train_cnn import DATA, OUT, augment, d4, get_folds, load_split  # noqa: E402

PUB_STD = 16.56
SIZE = 518  # --size로 변경


class DinoReg(nn.Module):
    def __init__(self, name, layers, pool_mode="meanstd"):
        super().__init__()
        self.bb = timm.create_model(name, pretrained=True, num_classes=0, img_size=SIZE, dynamic_img_size=True)
        self.layers = layers
        self.pool_mode = pool_mode
        cfg = self.bb.pretrained_cfg
        self.register_buffer("im_mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        self.register_buffer("f_mu", torch.zeros(1))
        self.register_buffer("f_sd", torch.ones(1))
        self.head = None

    def feats(self, x):  # x: B,1,256,256 in [0,1]
        x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        x = (x.repeat(1, 3, 1, 1) - self.im_mean) / self.im_std
        fs = self.bb.forward_intermediates(x, indices=self.layers, intermediates_only=True, stop_early=True)
        if self.pool_mode != "meanstd":  # probe_models.py와 같은 풀링 (hetero3 등)
            return torch.cat([pool(z, self.pool_mode) for z in fs], 1)
        return torch.cat([torch.cat([z.float().mean((2, 3)), z.float().std((2, 3))], 1) for z in fs], 1)

    def forward(self, x):
        return self.head((self.feats(x) - self.f_mu) / self.f_sd).squeeze(1)


@torch.no_grad()
def tta_feats(model, X, bs=16):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].cuda().float().div_(255)
        acc = 0
        for k in (0, 1, 4, 5):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                acc = acc + model.feats(d4(xb, k))
        out.append((acc / 4).float())
    return torch.cat(out)


@torch.no_grad()
def tta_pred(model, X, bs=16):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].cuda().float().div_(255)
        acc = 0
        for k in (0, 1, 4, 5):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                acc = acc + model(d4(xb, k)).float()
        out.append((acc / 4).cpu())
    return torch.cat(out).numpy()


def param_groups(model, lr_bb, lr_head, decay):
    blocks = model.bb.blocks
    n = len(blocks)
    groups = [{"params": list(model.head.parameters()), "lr": lr_head, "weight_decay": 0.0}]
    for i, blk in enumerate(blocks):
        groups.append({"params": list(blk.parameters()), "lr": lr_bb * decay ** (n - 1 - i), "weight_decay": 0.05})
    rest = [p for nme, p in model.bb.named_parameters() if not nme.startswith("blocks.")]
    groups.append({"params": rest, "lr": lr_bb * decay ** n, "weight_decay": 0.05})
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-bb", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--llrd", type=float, default=0.85, help="층별 학습률 감쇠 (아래층일수록 작게)")
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--noise", type=float, default=16.0)
    ap.add_argument("--rot-p", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=518, help="입력 해상도 (hetero3는 448이 최적)")
    ap.add_argument("--pool", choices=["meanstd", "hetero", "hetero3"], default="meanstd")
    ap.add_argument("--src", default=None, help="입력 이미지 캐시 npz (디노이즈·조명보정 등). 없으면 원본 PNG")
    ap.add_argument("--blur-p", type=float, default=0.25)
    ap.add_argument("--contrast-aug", type=float, default=0.1)
    a = ap.parse_args()
    global SIZE
    SIZE = a.size
    a.no_d4, a.contrast, a.brightness = False, a.contrast_aug, 0.03
    torch.backends.cudnn.benchmark = True
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "sample_submission.csv")
    fold = get_folds(train)
    y = train.hardness.values.astype(np.float64)
    if a.src:
        z = np.load(OUT.parent / a.src)
        X, Xte = torch.from_numpy(z["train"]).unsqueeze(1), torch.from_numpy(z["test"]).unsqueeze(1)
    else:
        X, Xte = load_split("train", train.ID), load_split("test", test.ID)
    out = OUT / "emb_reg" / a.exp
    out.mkdir(parents=True, exist_ok=True)
    oof, tp = np.zeros(len(y)), np.zeros(len(test))
    done, t0 = [], time.time()
    for k in (a.folds if a.folds is not None else range(5)):
        torch.manual_seed(42 + k)
        tr, va = fold != k, fold == k
        model = DinoReg(a.model, a.layers, a.pool).cuda()
        model.bb.set_grad_checkpointing(True)
        # 1) LP: 고정 특징 Ridge → 헤드 초기화
        Ftr = tta_feats(model, X[tr])
        mu, sd = Ftr.mean(0, keepdim=True), Ftr.std(0, keepdim=True) + 1e-6
        rg = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(((Ftr - mu) / sd).cpu().numpy(), y[tr])
        model.f_mu, model.f_sd = mu, sd
        model.head = nn.Linear(Ftr.shape[1], 1).cuda()
        with torch.no_grad():
            model.head.weight.copy_(torch.tensor(rg.coef_, dtype=torch.float32).view(1, -1))
            model.head.bias.fill_(float(rg.intercept_))
        p0 = tta_pred(model, X[va])
        r0 = np.sqrt(np.mean((p0 - y[va]) ** 2))
        del Ftr
        # 2) FT
        opt = torch.optim.AdamW(param_groups(model, a.lr_bb, a.lr_head, a.llrd))
        base = [g["lr"] for g in opt.param_groups]
        Xtr_g = X[tr].cuda()
        yt = torch.tensor(y[tr], dtype=torch.float32, device="cuda")
        ysd = float(y[tr].std())
        n = len(Xtr_g)
        spe = n // a.batch
        total = spe * a.epochs
        step = 0
        for ep in range(a.epochs):
            model.train()
            perm = torch.randperm(n, device="cuda")
            tl = 0.0
            for b in range(spe):
                idx = perm[b * a.batch:(b + 1) * a.batch]
                x = augment(Xtr_g[idx].float().div_(255), a)
                f = min(1.0, (step + 1) / spe) * 0.5 * (1 + math.cos(math.pi * step / total))
                for g, bl in zip(opt.param_groups, base):
                    g["lr"] = bl * f
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(x).float()
                loss = F.mse_loss(pred / ysd, yt[idx] / ysd)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl += loss.item()
                step += 1
            print(f"  fold{k} ep{ep + 1} train loss {tl / spe:.4f}", flush=True)
        oof[va] = tta_pred(model, X[va])
        pt = tta_pred(model, Xte)
        tp += pt / 5
        np.savez(out / f"fold{k}.npz", va=np.where(va)[0], pred=oof[va], test=pt)  # 폴드별 저장 (나눠 실행 후 병합)
        r1 = np.sqrt(np.mean((oof[va] - y[va]) ** 2))
        print(f"  fold{k} RMSE: 고정특징(LP) {r0:.3f} → 미세조정(FT) {r1:.3f} ({time.time() - t0:.0f}s)", flush=True)
        done.append(k)
        del model, opt, Xtr_g
        torch.cuda.empty_cache()
    m = np.isin(fold, done)
    w = np.exp(-0.5 * ((y - y.mean()) / PUB_STD) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
    w /= w.mean()
    print(f"[{a.exp}] folds {done}: FT RMSE {np.sqrt(np.mean((oof[m] - y[m]) ** 2)):.3f} | "
          f"재가중 {np.sqrt(np.mean(w[m] * (oof[m] - y[m]) ** 2)):.3f} | {(time.time() - t0) / 60:.1f} min")
    files = [out / f"fold{j}.npz" for j in range(5)]
    if all(f.exists() for f in files):  # 다른 실행에서 끝낸 폴드까지 합쳐 최종 산출물 생성
        oof_all, tp_all = np.zeros(len(y)), np.zeros(len(test))
        for f in files:
            z = np.load(f)
            oof_all[z["va"]] = z["pred"]
            tp_all += z["test"] / 5
        pd.DataFrame({"ID": train.ID, "fold": fold, "hardness": y, "pred": oof_all}).to_csv(out / "oof.csv", index=False)
        pd.DataFrame({"ID": test.ID, "hardness": tp_all}).to_csv(out / "test_pred.csv", index=False)
        print(f"[{a.exp}] 5폴드 병합 OOF RMSE {np.sqrt(np.mean((oof_all - y) ** 2)):.3f}")


if __name__ == "__main__":
    main()
