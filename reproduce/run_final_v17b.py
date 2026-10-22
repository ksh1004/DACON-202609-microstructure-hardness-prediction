"""최종 제출(v17b) 재현 드라이버 — 패키지 루트에서 실행 (모든 경로는 이 파일 기준 상대경로).

  python run_final.py --mode infer   # 저장된 디노이저·CNN 가중치로 추론 (Ridge·특징은 매번 새로 계산, 약 90분)
  python run_final.py --mode train   # 디노이저·CNN까지 처음부터 학습 (RTX 5060 Ti 16GB 기준 약 3~4시간)

단계별 산출물이 이미 있으면 건너뛴다(중단 후 재개 가능). --force 로 전부 다시 계산.
최종 산출물: submission/submission.csv

파이프라인 (test 이미지는 추론에만 사용, 정답·test 통계로 학습하는 단계 없음):
  1. 원본 특징 캐시 (q_noise_sigma: 디노이저 정답 선택 σ<7, 조명 보정 대상 σ≥10)
  2. 디노이저: 1차 U-Net (train σ<7 이미지 + 합성 가우시안 노이즈) → 부트스트랩 U-Net (1차 출력을 정답으로) → 전체 이미지 디노이즈
  3. 저품질 모드(σ≥10)만 조명 보정: 가우시안 저역(σ 32/48/64/96) 제거 4종
  4. DINOv2-L (고정, 448 입력) 9~21층 패치 토큰 풀링 — 보정 4종 각각, 보기 묶음 두 가지:
     4a. 4보기(D4 중 4방향) 평균: hetero3 풀링 + 상 조건부 풀링(2상/기지 가중 평균·std)
     4b. 16보기(D4 8방향 × 격자 이동 2) 평균: 위 + 혼합 법칙 기여량(2상 분율 × 2상 평균 토큰)
     풀링 통계는 이미지를 넣는 방향·격자 위치(보기)에 따라 흔들리므로, 서로 다른 보기 묶음으로 만든 두 모델을 평균한다 (추론 시 TTA, 학습 아님)
  5. 11~17층 Ridge, 전체 500장 재적합 (4종 × 2): 5a. hetero3 + 상 조건부 풀링(상대 분산 0.25) / 5b. 5a + 혼합 기여량(0.05)
  6. σ48 보정 이미지에서 상분율 특징(phase_features) + 수작업 특징(ph_/pore/gr/sp) + σ7~10 평활 밝기 CDF(smooth_cdf_features)
  7. σ48의 4보기 DINO 특징(hetero3 + 상 조건부 풀링 0.25) + 상분율(0.4) + 수작업(0.15) + 평활 CDF(0.2) → 한 Ridge (전체 재적합)
  8. CNN 9모델 (ConvNeXt-T / EfficientNet-B2 / ResNet-50 × 시드 42·43·44, 원본 512 입력, 5폴드 평균)
  9. 최종 = 0.15 × CNN 평균 + 0.275 × DINO(4보기) 4종 평균 + 0.425 × DINO(16보기) 4종 평균 + 0.15 × DINO+측정특징
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
DN1, BOOT = "cm7_am22_l1_i6000", "cm7_am22_l1_i10000_boot"
FF = (32, 48, 64, 96)
DINO = "vit_large_patch14_dinov2.lvd142m"
LAYERS = ["9", "11", "13", "15", "17", "19", "21"]
CNN_RECIPES = {
    "convnext_tiny": ["--model", "convnext_tiny.fb_in22k_ft_in1k", "--lr", "3e-4"],
    "effnet_b2": ["--model", "efficientnet_b2.ra_in1k", "--lr", "1e-3", "--warmup", "3"],
    "resnet50": ["--model", "resnet50.a1_in1k", "--lr", "1e-3"],
}
CNN_COMMON = ["--epochs", "30", "--eval-every", "10", "--rot-p", "0.5", "--noise", "16", "--img-size", "512", "--grad-ckpt", "--save-weights"]
SEEDS = (42, 43, 44)
FORCE = False


def step(name: str, outputs: list[Path], cmds: list[list[str]]):
    if not FORCE and outputs and all(p.exists() for p in outputs):
        print(f"[skip] {name}")
        return
    print(f"[run ] {name}", flush=True)
    t = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for c in cmds:
        subprocess.run([PY, "-W", "ignore", *c], cwd=ROOT, env=env, check=True)
    print(f"[done] {name} ({time.time() - t:.0f}s)", flush=True)


def main():
    global FORCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["infer", "train"], default="infer")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    FORCE = a.force
    O = ROOT / "outputs"
    dn = lambda tag: O / "denoised" / f"unet_{tag}.npz"

    step("1. 원본 특징", [O / "features/train_features.csv", O / "features/test_features.csv"], [["src/raw_features.py"]])

    if a.mode == "train":
        step("2a. 1차 디노이저 학습", [dn(DN1)], [["src/train_denoiser.py", "--iters", "6000"]])
        step("2b. 부트스트랩 디노이저 학습", [dn(BOOT)],
             [["src/train_denoiser.py", "--iters", "10000", "--target-src", f"outputs/denoised/unet_{DN1}.npz"]])
    else:
        step("2. 부트스트랩 디노이저 추론", [dn(BOOT)],
             [["src/denoise_with.py", "--weights", f"weights/denoiser_{BOOT}.pt", "--tag", BOOT]])

    lay = "-".join(LAYERS)
    emb4 = lambda src: f"outputs/probe2/{DINO}_448_L{lay}_Pphase2_Sunet_{src}"            # 4보기, 층당 10블록
    emb16 = lambda src: f"outputs/probe2/{DINO}_448_L{lay}_Pphase2m_T8_Sunet_{src}_sh2"   # 16보기, 층당 11블록
    # multi_ridge 묶음: 11~17층(층 위치 1~4)의 hetero3 블록(0~5, 기준) + 상 조건부 풀링 블록(6~9, 상대 분산 0.25) [+ 혼합 기여량 블록(10, 0.05)]
    dino_groups = lambda e, nb: ["--group", f"emb:{e}:1:1,2,3,4:1024:{nb}:0,1,2,3,4,5", "--group", f"emb:{e}:0.25:1,2,3,4:1024:{nb}:6,7,8,9"]
    mix_group = lambda e: ["--group", f"emb:{e}:0.05:1,2,3,4:1024:11:10"]
    probe = ["src/probe_models.py", "--model", DINO, "--size", "448", "--layers", *LAYERS, "--bs", "8"]
    for s in FF:
        src = f"{BOOT}_fflowpass{s}_low"
        step(f"3. 조명 보정 σ{s}", [dn(src)],
             [["src/flatfield_cache.py", "--src", f"outputs/denoised/unet_{BOOT}.npz", "--mode", "lowpass", "--sigma", str(s), "--only-low"]])
        step(f"4a. DINOv2 특징 σ{s} (4보기)", [ROOT / emb4(src) / "emb.npz"],
             [[*probe, "--pool", "phase2", "--src", f"outputs/denoised/unet_{src}.npz"]])
        step(f"4b. DINOv2 특징 σ{s} (16보기)", [ROOT / emb16(src) / "emb.npz"],
             [[*probe, "--pool", "phase2m", "--tta", "8", "--shifts", "2", "--src", f"outputs/denoised/unet_{src}.npz"]])
        step(f"5a. DINO Ridge σ{s} (4보기)", [O / "emb_reg" / f"dnpp0.25_ff{s}" / "test_pred_full.csv"],
             [["src/multi_ridge.py", "--name", f"dnpp0.25_ff{s}", *dino_groups(emb4(src), 10)]])
        step(f"5b. DINO Ridge σ{s} (16보기)", [O / "emb_reg" / f"mv16_dnppm0.05_ff{s}" / "test_pred_full.csv"],
             [["src/multi_ridge.py", "--name", f"mv16_dnppm0.05_ff{s}", *dino_groups(emb16(src), 11), *mix_group(emb16(src))]])

    s48 = f"unet_{BOOT}_fflowpass48_low"
    fc = O / "features_cache"
    step("6. 상분율·수작업·평활 CDF 특징 (σ48)",
         [fc / f"phase_{s48}_test.csv", fc / f"{s48}_test.csv", fc / f"smcdf3_{s48}_test.csv"],
         [["src/phase_features.py", "--src", f"outputs/denoised/{s48}.npz"],
          ["src/features_from_cache.py", "--src", f"outputs/denoised/{s48}.npz"],
          ["src/smooth_cdf_features.py", "--src", f"outputs/denoised/{s48}.npz"]])
    step("7. DINO + 측정 특징 Ridge", [O / "emb_reg/mr_v14_pp_sc/test_pred_full.csv"],
         [["src/multi_ridge.py", "--name", "mr_v14_pp_sc", *dino_groups(emb4(f"{BOOT}_fflowpass48_low"), 10),
           "--group", f"csv:outputs/features_cache/phase_{s48}:0.381:pf_",
           "--group", f"csv:outputs/features_cache/{s48}:0.019:ph_",
           "--group", f"csv:outputs/features_cache/{s48}:0.15:(pore|gr|sp)_",
           "--group", f"csv:outputs/features_cache/smcdf3_{s48}:0.2:sc"]])

    cnn_dirs = []
    for m, rec in CNN_RECIPES.items():
        for sd in SEEDS:
            exp = f"s512_{m}_s{sd}"
            d = O / "cnn" / exp
            cnn_dirs.append(f"outputs/cnn/{exp}")
            if a.mode == "train":
                step(f"8. CNN 학습 {exp}", [d / "test_pred.csv", d / "weights/fold4.pt"],
                     [["src/train_cnn.py", *rec, *CNN_COMMON, "--seed", str(sd), "--exp", exp]])
            else:
                step(f"8. CNN 추론 {exp}", [d / "test_pred_reproduced.csv"], [["src/cnn_predict.py", "--exp", exp]])

    step("9. 최종 블렌드", [ROOT / "submission/submission.csv"],
         [["src/final_blend.py", "--out", "submission/submission.csv", "--weights", "0.15", "0.275", "0.425", "0.15",
           "--group", "cnn", *cnn_dirs,
           "--group", "dino4", *[f"outputs/emb_reg/dnpp0.25_ff{s}" for s in FF],
           "--group", "dino16", *[f"outputs/emb_reg/mv16_dnppm0.05_ff{s}" for s in FF],
           "--group", "phase", "outputs/emb_reg/mr_v14_pp_sc"]])


if __name__ == "__main__":
    main()
