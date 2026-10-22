"""GPU 학습 처리량 벤치마크 — 노트북(RTX 5060)과 Colab(T4)에서 동일하게 실행해 비교한다.

실행: python bench/gpu_benchmark.py [--out bench/result_<tag>.csv] [--batch 32]
측정 항목
  - 후보 백본별 학습 처리량 (256x256, AMP fp16, AdamW, 회귀 헤드 1개)
    * 기본은 NCHW. RTX 5060(Windows, torch 2.11)에서 channels_last + 학습모드 BatchNorm이 ~25배 느린 문제가 있어
      --channels-last 는 옵션으로만 제공 (T4/Linux에서는 켜는 편이 유리할 수 있음)
  - 피크 GPU 메모리
  - 보정용: ResNet-50 추론 (224x224, batch 256, AMP) → NVIDIA 공개 T4 수치(1,198 img/s)와 비교
  - 측정 중 GPU 전력/SM 클럭/온도 (nvidia-smi)
"""
from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path

import timm
import torch

MODELS = [
    "resnet18", "resnet34", "resnet50",
    "efficientnet_b0", "efficientnet_b2", "tf_efficientnetv2_s",
    "regnety_016", "regnety_032",
    "convnext_nano", "convnext_tiny",
    "vit_small_patch16_224",
]


def gpu_stats() -> str:
    try:
        q = "power.draw,power.limit,clocks.sm,temperature.gpu"
        return subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                                       text=True, timeout=5).strip()
    except Exception:
        return "n/a"


def bench_train(name: str, batch: int, size: int, iters: int, warmup: int, dtype, cl: bool) -> dict:
    kw = {"img_size": size} if name.startswith("vit") else {}
    fmt = torch.channels_last if cl else torch.contiguous_format
    m = timm.create_model(name, pretrained=False, num_classes=1, **kw).cuda().to(memory_format=fmt)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    x = torch.randn(batch, 3, size, size, device="cuda").to(memory_format=fmt)
    y = torch.randn(batch, 1, device="cuda")
    torch.cuda.reset_peak_memory_stats()

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            loss = torch.nn.functional.mse_loss(m(x).float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t = time.perf_counter()
    stats = ""
    for i in range(iters):
        step()
        if i == iters // 2:
            stats = gpu_stats()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    res = {
        "model": name,
        "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2),
        "train_img_s": round(batch * iters / dt, 1),
        "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "gpu_power_limit_clock_temp": stats,
    }
    del m, opt
    torch.cuda.empty_cache()
    return res


def bench_infer_rn50(batch=256, iters=30, warmup=10) -> float:
    m = timm.create_model("resnet50", pretrained=False).cuda().eval().to(memory_format=torch.channels_last)
    x = torch.randn(batch, 3, 224, 224, device="cuda").to(memory_format=torch.channels_last)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(warmup):
            m(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            m(x)
        torch.cuda.synchronize()
    del m
    torch.cuda.empty_cache()
    return round(batch * iters / (time.perf_counter() - t), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16",
                    help="T4(Turing)는 bf16 텐서코어가 없으므로 fp16, RTX 5060(Blackwell)은 bf16이 더 빠름")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True

    dev = torch.cuda.get_device_name(0)
    print(f"device={dev} | torch={torch.__version__} | cuda={torch.version.cuda} | timm={timm.__version__} | "
          f"python={platform.python_version()} | os={platform.system()}")
    print(f"idle: {gpu_stats()}")
    rn50 = bench_infer_rn50()
    print(f"[calib] ResNet-50 inference 224px bs256 AMP: {rn50} img/s (NVIDIA 공개 T4 수치: 1198 img/s)")

    rows = []
    for name in args.models:
        try:
            dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
            r = bench_train(name, args.batch, args.size, args.iters, 15, dt, args.channels_last)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            r = {"model": name, "train_img_s": "OOM"}
        r.update(device=dev, batch=args.batch, size=args.size, dtype=args.dtype, channels_last=args.channels_last,
                 rn50_infer_img_s=rn50)
        rows.append(r)
        print(r)

    out = Path(args.out or f"bench/result_{dev.replace(' ', '_')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("saved ->", out)


if __name__ == "__main__":
    main()
