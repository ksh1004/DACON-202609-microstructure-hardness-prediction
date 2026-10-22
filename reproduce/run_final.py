"""최종 제출(v24a_dinoj_lowreg_k103) 재현 드라이버 — 패키지 루트에서 실행 (모든 경로는 이 파일 기준 상대경로).

  python run_final.py --mode infer   # 저장된 디노이저 가중치로 추론 (Ridge·특징은 매번 새로 계산, 약 3시간 15분)
  python run_final.py --mode train   # 디노이저까지 처음부터 학습 (RTX 5060 Ti 16GB 기준 +35분)

단계별 산출물이 이미 있으면 건너뛴다(중단 후 재개 가능). --force 로 전부 다시 계산.
최종 산출물: submission/submission.csv

파이프라인 (test 이미지는 추론에만 사용, 정답·test 통계로 학습하는 단계 없음):
  1. 원본 특징 캐시 (q_noise_sigma: 디노이저 정답 선택 σ<7, 조명 보정 대상 σ≥10)
  2. 디노이저: 1차 U-Net (train σ<7 이미지 + 합성 가우시안 노이즈) → 부트스트랩 U-Net (1차 출력을 정답으로) → 전체 이미지 디노이즈
     2c. 두 번째 디노이즈 경로: 공개 사전학습 DRUNet(gray, KAIR, MIT)에 이미지별 추정 σ를 넣어 디노이즈 (D4 평균) — 결정립계를 지어내지 않는 충실한 복원
  3. 저품질 모드(σ≥10)만 조명 보정: 가우시안 저역(σ 32/48/64/96) 제거 4종 (두 디노이즈 경로 모두)
  4. 고정(frozen) 사전학습 모델의 특징맵 풀링 — 보정 4종 각각. 풀링은 공통: 평균·표준편차·영역 불균일성(hetero3) + 상 조건부 풀링
     (2상/기지 가중 평균·std) [+ 혼합 법칙 기여량 = 2상 분율 × 2상 평균]. 풀링 통계가 이미지를 넣는 방향·격자 위치(보기)에 따라
     흔들리므로 여러 보기의 통계를 평균한다 (추론 시 TTA, 학습 아님).
     4a. DINOv2-L (448 입력, 9~21층) 4보기          4b. DINOv2-L 16보기(D4 8방향 × 격자 이동 2) + 기여량
     4e. DINOv2-L 16보기 + 기여량 — DRUNet 경로 이미지
     4c. TIPSv2-So400m (448 입력, 8~14층) 16보기 + 기여량
     4d. ImageNet CNN 4종(DenseNet-121·흐림 풀링 DenseNet-121 denseblock2, ResNet-50 A1·A2 layer2)의 stride-8 특징맵 (512 입력) 8보기 + 기여량
  5. 모델·보정별 Ridge, 전체 500장 재적합: hetero3(기준) + 상 조건부 풀링(상대 분산 0.25) [+ 기여량(0.05)]
     벌점은 구성원별 LOO 선택 대신 작은 고정값(--alpha-rel 0.01): 덜 정규화한 구성원끼리 오차 상관이 낮아 블렌드가 좋아진다
     DINOv2 16보기(5b)는 U-Net 경로와 DRUNet 경로의 특징을 한 Ridge에 결합한다
     고정 CNN(5d)은 채널 수가 적어 모델별로 따로 맞추지 않고 4개 모델의 특징을 한 Ridge에 결합한다 (모델 간 같은 가중)
  6. σ48 보정 이미지에서 상분율 특징(phase_features) + 수작업 특징(ph_/pore/gr/sp) + σ7~10 평활 밝기 CDF(smooth_cdf_features)
  7. σ48의 4보기 DINO 특징(hetero3 + 상 조건부 풀링 0.25) + 상분율(0.4) + 수작업(0.15) + 평활 CDF(0.2) → 한 Ridge (전체 재적합)
  8. 최종 = 0.1765 × DINO+측정특징 + 0.8235 × [0.45 × DINOv2(4보기 0.275 : 두 경로 16보기 0.425) + 0.25 × TIPSv2 + 0.30 × 고정 CNN 4종 결합]
     → 예측 폭 ×1.03 (train 경도 평균 기준; 구성원을 평균할수록 예측이 가운데로 몰리는 것을 되돌리는 고정 배율)
학습하는 모델은 자체 디노이저(U-Net)뿐이고 (DRUNet은 공개 가중치 그대로 추론만), 경도 회귀는 모두 고정 특징 + Ridge다 (미세조정 CNN은 2026-10-14에 제외: 교차검증·Public 모두 빼는 쪽이 좋았음).
사전학습 모델은 모두 대회 시작일(2026-09-29) 이전에 공개된 오픈소스 가중치다 (README 5절).
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
AR = "0.01"   # Ridge 고정 벌점 (multi_ridge --alpha-rel)
DINO = "vit_large_patch14_dinov2.lvd142m"
TIPS, TIPS_LAYERS = "vit_so400m_patch14_reg1_tipsv2.webli", ["8", "10", "12", "14"]
CNNF = (("densenet121.tv_in1k@2", "dn121"), ("resnet50.a1_in1k@2", "r50a1"), ("resnet50.a2_in1k@2", "r50a2"), ("densenetblur121d.ra_in1k@2", "dnb121"))   # timm 모델@features_only 단계(stride 8), Ridge 이름
LAYERS = ["9", "11", "13", "15", "17", "19", "21"]
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
    step("2c. DRUNet 두 번째 디노이즈 경로", [O / "denoised/drunet.npz"], [["src/drunet_denoise.py"]])

    lay = "-".join(LAYERS)
    emb4 = lambda src: f"outputs/probe2/{DINO}_448_L{lay}_Pphase2_Sunet_{src}"            # 4보기, 층당 10블록
    emb16 = lambda src: f"outputs/probe2/{DINO}_448_L{lay}_Pphase2m_T8_Sunet_{src}_sh2"   # 16보기, 층당 11블록
    emb16d = lambda s: f"outputs/probe2/{DINO}_448_L{lay}_Pphase2m_T8_Sdrunet_fflowpass{s}_low_sh2"   # DRUNet 경로 16보기
    # multi_ridge 묶음: 11~17층(층 위치 1~4)의 hetero3 블록(0~5, 기준) + 상 조건부 풀링 블록(6~9, 상대 분산 0.25) [+ 혼합 기여량 블록(10, 0.05)]
    dino_groups = lambda e, nb: ["--group", f"emb:{e}:1:1,2,3,4:1024:{nb}:0,1,2,3,4,5", "--group", f"emb:{e}:0.25:1,2,3,4:1024:{nb}:6,7,8,9"]
    mix_group = lambda e: ["--group", f"emb:{e}:0.05:1,2,3,4:1024:11:10"]
    probe = ["src/probe_models.py", "--model", DINO, "--size", "448", "--layers", *LAYERS, "--bs", "8"]
    embT = lambda src: f"outputs/probe2/{TIPS}_448_L{'-'.join(TIPS_LAYERS)}_Pphase2m_T8_Sunet_{src}_sh2"          # TIPSv2 16보기, 4층 × 11블록 × 1152
    embC = lambda m, src: f"outputs/probe2/cnn_{m.replace('@', '_s')}_512_Pphase2m_T8_Sunet_{src}"                # 고정 CNN 8보기, 1단계 × 11블록 × 512
    one_groups = lambda e, layers, c: ["--group", f"emb:{e}:1:{layers}:{c}:11:0,1,2,3,4,5", "--group", f"emb:{e}:0.25:{layers}:{c}:11:6,7,8,9",
                                       "--group", f"emb:{e}:0.05:{layers}:{c}:11:10"]
    for s in FF:
        src = f"{BOOT}_fflowpass{s}_low"
        step(f"3. 조명 보정 σ{s}", [dn(src)],
             [["src/flatfield_cache.py", "--src", f"outputs/denoised/unet_{BOOT}.npz", "--mode", "lowpass", "--sigma", str(s), "--only-low"]])
        step(f"3b. 조명 보정 σ{s} (DRUNet 경로)", [O / "denoised" / f"drunet_fflowpass{s}_low.npz"],
             [["src/flatfield_cache.py", "--src", "outputs/denoised/drunet.npz", "--mode", "lowpass", "--sigma", str(s), "--only-low"]])
        step(f"4e. DINOv2 특징 σ{s} (16보기, DRUNet 경로)", [ROOT / emb16d(s) / "emb.npz"],
             [[*probe, "--pool", "phase2m", "--tta", "8", "--shifts", "2", "--src", f"outputs/denoised/drunet_fflowpass{s}_low.npz"]])
        step(f"4a. DINOv2 특징 σ{s} (4보기)", [ROOT / emb4(src) / "emb.npz"],
             [[*probe, "--pool", "phase2", "--src", f"outputs/denoised/unet_{src}.npz"]])
        step(f"4b. DINOv2 특징 σ{s} (16보기)", [ROOT / emb16(src) / "emb.npz"],
             [[*probe, "--pool", "phase2m", "--tta", "8", "--shifts", "2", "--src", f"outputs/denoised/unet_{src}.npz"]])
        step(f"5a. DINO Ridge σ{s} (4보기)", [O / "emb_reg" / f"dnpp0.25_ff{s}_a{AR}" / "test_pred_full.csv"],
             [["src/multi_ridge.py", "--alpha-rel", AR, "--name", f"dnpp0.25_ff{s}_a{AR}", *dino_groups(emb4(src), 10)]])
        step(f"5b. DINO Ridge σ{s} (16보기, 두 경로 결합)", [O / "emb_reg" / f"mv16j_ff{s}_a{AR}" / "test_pred_full.csv"],
             [["src/multi_ridge.py", "--alpha-rel", AR, "--name", f"mv16j_ff{s}_a{AR}", *dino_groups(emb16(src), 11), *mix_group(emb16(src)),
               *dino_groups(emb16d(s), 11), *mix_group(emb16d(s))]])
        step(f"4c. TIPSv2 특징 σ{s} (16보기)", [ROOT / embT(src) / "emb.npz"],
             [["src/probe_models.py", "--model", TIPS, "--size", "448", "--layers", *TIPS_LAYERS, "--pool", "phase2m", "--tta", "8", "--shifts", "2",
               "--bs", "8", "--src", f"outputs/denoised/unet_{src}.npz"]])
        step(f"5c. TIPSv2 Ridge σ{s}", [O / "emb_reg" / f"tips16_dnppm0.05_ff{s}_a{AR}" / "test_pred_full.csv"],
             [["src/multi_ridge.py", "--alpha-rel", AR, "--name", f"tips16_dnppm0.05_ff{s}_a{AR}", *one_groups(embT(src), "0,1,2,3", 1152)]])
        for m, n in CNNF:
            step(f"4d. 고정 CNN 특징 {n} σ{s} (8보기)", [ROOT / embC(m, src) / "emb.npz"],
                 [["src/probe_models.py", "--model", f"cnn:{m}", "--size", "512", "--pool", "phase2m", "--tta", "8", "--bs", "16",
                   "--src", f"outputs/denoised/unet_{src}.npz"]])
        step(f"5d. 고정 CNN 결합 Ridge σ{s} (4개 모델)", [O / "emb_reg" / f"cnnfj4_ff{s}_a{AR}" / "test_pred_full.csv"],
             [["src/multi_ridge.py", "--alpha-rel", AR, "--name", f"cnnfj4_ff{s}_a{AR}", *[g for m, _ in CNNF for g in one_groups(embC(m, src), "0", 512)]]])

    s48 = f"unet_{BOOT}_fflowpass48_low"
    fc = O / "features_cache"
    step("6. 상분율·수작업·평활 CDF 특징 (σ48)",
         [fc / f"phase_{s48}_test.csv", fc / f"{s48}_test.csv", fc / f"smcdf3_{s48}_test.csv"],
         [["src/phase_features.py", "--src", f"outputs/denoised/{s48}.npz"],
          ["src/features_from_cache.py", "--src", f"outputs/denoised/{s48}.npz"],
          ["src/smooth_cdf_features.py", "--src", f"outputs/denoised/{s48}.npz"]])
    step("7. DINO + 측정 특징 Ridge", [O / f"emb_reg/mr_v14_pp_sc_a{AR}/test_pred_full.csv"],
         [["src/multi_ridge.py", "--alpha-rel", AR, "--name", f"mr_v14_pp_sc_a{AR}", *dino_groups(emb4(f"{BOOT}_fflowpass48_low"), 10),
           "--group", f"csv:outputs/features_cache/phase_{s48}:0.381:pf_",
           "--group", f"csv:outputs/features_cache/{s48}:0.019:ph_",
           "--group", f"csv:outputs/features_cache/{s48}:0.15:(pore|gr|sp)_",
           "--group", f"csv:outputs/features_cache/smcdf3_{s48}:0.2:sc"]])

    # 가중치(합으로 정규화): 측정 블록 0.15 : DINOv2 4보기 0.12375 : 두 경로 16보기 0.19125 : TIPSv2 0.175 : 고정 CNN 결합 0.21, 예측 폭 ×1.03
    step("8. 최종 블렌드", [ROOT / "submission/submission.csv"],
         [["src/final_blend.py", "--out", "submission/submission.csv", "--weights", "0.15", "0.12375", "0.19125", "0.175", "0.21",
           "--scale", "1.03",
           "--group", "phase", f"outputs/emb_reg/mr_v14_pp_sc_a{AR}",
           "--group", "dino4", *[f"outputs/emb_reg/dnpp0.25_ff{s}_a{AR}" for s in FF],
           "--group", "dinoj", *[f"outputs/emb_reg/mv16j_ff{s}_a{AR}" for s in FF],
           "--group", "tips16", *[f"outputs/emb_reg/tips16_dnppm0.05_ff{s}_a{AR}" for s in FF],
           "--group", "cnnfj4", *[f"outputs/emb_reg/cnnfj4_ff{s}_a{AR}" for s in FF]]])


if __name__ == "__main__":
    main()
