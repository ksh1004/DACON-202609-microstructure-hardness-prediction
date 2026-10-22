"""최종 제출(v26) 재현 드라이버 — 패키지 루트에서 실행 (모든 경로는 이 파일 기준 상대경로).

  python run_final.py --mode infer   # 저장된 디노이저 가중치로 추론 (Ridge·특징은 매번 새로 계산)
  python run_final.py --mode train   # 디노이저까지 처음부터 학습 (RTX 5060 Ti 16GB 기준 약 +1시간 40분)

단계별 산출물이 이미 있으면 건너뛴다(중단 후 재개 가능). --force 로 전부 다시 계산.
최종 산출물: submission/submission.csv

파이프라인 (test 이미지는 추론에만 사용, 정답·test 통계로 학습하는 단계 없음):
  1. 원본 특징 캐시 (q_noise_sigma: 디노이저 정답 선택, 조명 보정 대상 σ≥10, DRUNet 노이즈 지도)
  2. 디노이즈 경로 4개 (모두 D4 8방향 평균):
     U  자체 U-Net: 1차 (train σ<7 이미지 + 합성 가우시안 노이즈) → 부트스트랩 (1차 출력을 정답으로)
     D  공개 사전학습 DRUNet(gray, KAIR, MIT)에 이미지별 추정 σ — 결정립계를 지어내지 않는 충실한 복원
     X  자체 U-Net 증류(DRT): DRUNet이 정리한 고품질 train 이미지(σ<10, 291장)를 정답으로 학습 → 그 출력을 정답으로 한 번 더
     F  DRUNet 미세조정: 공개 DRUNet을 같은 정답(σ<10, 291장) + 합성 노이즈로 미세조정 → 그 출력을 정답으로 한 번 더
  3. 저품질 모드(σ≥10)만 조명 보정: 가우시안 저역(σ 32/48/64/96) 제거 4종 (경로마다)
  4. 고정(frozen) 사전학습 모델의 특징맵 풀링 — 보정 4종 각각. 풀링: 평균·표준편차·영역 불균일성(hetero3) + 상 조건부 풀링
     (2상/기지 가중 평균·std) [+ 혼합 법칙 기여량 = 2상 분율 × 2상 평균], 여러 보기(D4 방향 × 격자 이동) 평균 (추론 시 TTA, 학습 아님)
     DINOv2-L (448, 9~21층): U 4보기 / DINO 구성원 경로 16보기      TIPSv2-So400m (448, 8~14층) 16보기: U·X·F
     ImageNet CNN 4종(DenseNet-121·흐림 풀링 DenseNet-121 denseblock2, ResNet-50 A1·A2 layer2) stride-8 특징맵 (512) 8보기: U·X
  5. 모델·경로·보정별 Ridge(고정 벌점 --alpha-rel 0.01), 전체 500장 재적합. DINOv2 16보기는 DPATHS 경로 특징을 한 Ridge에,
     고정 CNN은 4개 모델 특징을 한 Ridge에.
  6~7. σ48 U 경로: 4보기 DINO + 상분율·수작업·평활 CDF → 한 Ridge (측정 블록)
  8. 최종 = 0.1765 × 측정 블록 + 0.1456 × DINOv2 4보기(U) + 0.2250 × DINOv2 16보기(DPATHS) + 0.2059 × TIPSv2(U·X·F 예측 평균)
     + 0.2471 × 고정 CNN 결합(U·X 예측 평균) → 예측 폭 ×1.03 (train 경도 평균 기준 고정 배율)
학습하는 모델은 디노이저뿐이고 (자체 U-Net 4개, 공개 DRUNet 미세조정 2개), 경도 회귀는 모두 고정 특징 + Ridge다.
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
DRT1, DRT = "drt10_i10000", "drt10b_i10000"     # 자체 U-Net 증류 1차·2차
FT1, FT = "ft1", "ft1b"                          # DRUNet 미세조정 1차·2차
DPATHS = "F"                                     # DINOv2 16보기 구성원의 경로 (F = 미세조정 단독, DXF = DRUNet·DRT·미세조정 결합)
FF = (32, 48, 64, 96)
AR = "0.01"   # Ridge 고정 벌점 (multi_ridge --alpha-rel)
DINO = "vit_large_patch14_dinov2.lvd142m"
TIPS, TIPS_LAYERS = "vit_so400m_patch14_reg1_tipsv2.webli", ["8", "10", "12", "14"]
CNNF = (("densenet121.tv_in1k@2", "dn121"), ("resnet50.a1_in1k@2", "r50a1"), ("resnet50.a2_in1k@2", "r50a2"), ("densenetblur121d.ra_in1k@2", "dnb121"))   # timm 모델@features_only 단계(stride 8)
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
    D = O / "denoised"
    # 경로 이름 → 디노이즈 캐시 파일 이름(확장자 제외)
    stem = {"U": f"unet_{BOOT}", "D": "drunet", "X": f"unet_{DRT}", "F": f"drunetft_{FT}"}

    step("1. 원본 특징", [O / "features/train_features.csv", O / "features/test_features.csv"], [["src/raw_features.py"]])

    step("2c. DRUNet 경로 (D)", [D / "drunet.npz"], [["src/drunet_denoise.py"]])
    if a.mode == "train":
        step("2a. 1차 디노이저 학습", [D / f"unet_{DN1}.npz"], [["src/train_denoiser.py", "--iters", "6000"]])
        step("2b. 부트스트랩 디노이저 학습 (U)", [D / f"unet_{BOOT}.npz"],
             [["src/train_denoiser.py", "--iters", "10000", "--target-src", f"outputs/denoised/unet_{DN1}.npz"]])
        step("2d. 증류 디노이저 1차 학습", [D / f"unet_{DRT1}.npz"],
             [["src/train_denoiser.py", "--iters", "10000", "--clean-max", "10", "--target-src", "outputs/denoised/drunet.npz", "--tag", DRT1]])
        step("2d. 증류 디노이저 2차 학습 (X)", [D / f"unet_{DRT}.npz"],
             [["src/train_denoiser.py", "--iters", "10000", "--clean-max", "10", "--target-src", f"outputs/denoised/unet_{DRT1}.npz", "--tag", DRT]])
        step("2e. DRUNet 미세조정 1차", [D / f"drunetft_{FT1}.npz"], [["src/drunet_finetune.py", "--tag", FT1]])
        step("2e. DRUNet 미세조정 2차 (F)", [D / f"drunetft_{FT}.npz"],
             [["src/drunet_finetune.py", "--tag", FT, "--target-src", f"outputs/denoised/drunetft_{FT1}.npz"]])
    else:
        step("2. 부트스트랩 디노이저 추론 (U)", [D / f"unet_{BOOT}.npz"],
             [["src/denoise_with.py", "--weights", f"weights/denoiser_{BOOT}.pt", "--tag", BOOT]])
        step("2d. 증류 디노이저 추론 (X)", [D / f"unet_{DRT}.npz"],
             [["src/denoise_with.py", "--weights", f"weights/denoiser_{DRT}.pt", "--tag", DRT]])
        step("2e. 미세조정 DRUNet 추론 (F)", [D / f"drunetft_{FT}.npz"],
             [["src/drunet_denoise.py", "--weights", f"weights/pretrained_denoiser/drunetft_{FT}.pt", "--tag", FT]])

    lay = "-".join(LAYERS)
    ff = lambda p, s: f"{stem[p]}_fflowpass{s}_low"
    emb4 = lambda s: f"outputs/probe2/{DINO}_448_L{lay}_Pphase2_S{ff('U', s)}"                  # DINOv2 4보기, 층당 10블록
    emb16 = lambda p, s: f"outputs/probe2/{DINO}_448_L{lay}_Pphase2m_T8_S{ff(p, s)}_sh2"         # DINOv2 16보기, 층당 11블록
    embT = lambda p, s: f"outputs/probe2/{TIPS}_448_L{'-'.join(TIPS_LAYERS)}_Pphase2m_T8_S{ff(p, s)}_sh2"   # TIPSv2 16보기
    embC = lambda m, p, s: f"outputs/probe2/cnn_{m.replace('@', '_s')}_512_Pphase2m_T8_S{ff(p, s)}"         # 고정 CNN 8보기
    # multi_ridge 묶음: 11~17층(층 위치 1~4)의 hetero3 블록(0~5, 기준) + 상 조건부 풀링 블록(6~9, 상대 분산 0.25) [+ 혼합 기여량 블록(10, 0.05)]
    dino_groups = lambda e, nb: ["--group", f"emb:{e}:1:1,2,3,4:1024:{nb}:0,1,2,3,4,5", "--group", f"emb:{e}:0.25:1,2,3,4:1024:{nb}:6,7,8,9"]
    mix_group = lambda e: ["--group", f"emb:{e}:0.05:1,2,3,4:1024:11:10"]
    one_groups = lambda e, layers, c: ["--group", f"emb:{e}:1:{layers}:{c}:11:0,1,2,3,4,5", "--group", f"emb:{e}:0.25:{layers}:{c}:11:6,7,8,9",
                                       "--group", f"emb:{e}:0.05:{layers}:{c}:11:10"]
    probe = ["src/probe_models.py", "--model", DINO, "--size", "448", "--layers", *LAYERS, "--bs", "8"]
    dname = {"F": f"mv16f_{FT}", "DXF": f"mv16dxf_{FT}"}[DPATHS]
    tname = {"U": "tips16_dnppm0.05", "X": f"tips16_{DRT}", "F": f"tips16_ft{FT}"}
    cname = {"U": "cnnfj4", "X": f"cnnfj4_{DRT}"}
    for s in FF:
        for p in ("U", "D", "X", "F"):
            if p == "D" and "D" not in DPATHS:
                continue
            step(f"3. 조명 보정 σ{s} ({p})", [D / f"{ff(p, s)}.npz"],
                 [["src/flatfield_cache.py", "--src", f"outputs/denoised/{stem[p]}.npz", "--mode", "lowpass", "--sigma", str(s), "--only-low"]])
        step(f"4a. DINOv2 4보기 σ{s} (U)", [ROOT / emb4(s) / "emb.npz"], [[*probe, "--pool", "phase2", "--src", f"outputs/denoised/{ff('U', s)}.npz"]])
        step(f"5a. DINOv2 4보기 Ridge σ{s}", [O / "emb_reg" / f"dnpp0.25_ff{s}_a{AR}" / "test_pred_full.csv"],
             [["src/multi_ridge.py", "--alpha-rel", AR, "--name", f"dnpp0.25_ff{s}_a{AR}", *dino_groups(emb4(s), 10)]])
        for p in DPATHS:
            step(f"4b. DINOv2 16보기 σ{s} ({p})", [ROOT / emb16(p, s) / "emb.npz"],
                 [[*probe, "--pool", "phase2m", "--tta", "8", "--shifts", "2", "--src", f"outputs/denoised/{ff(p, s)}.npz"]])
        step(f"5b. DINOv2 16보기 Ridge σ{s} ({DPATHS} 결합)", [O / "emb_reg" / f"{dname}_ff{s}_a{AR}" / "test_pred_full.csv"],
             [["src/multi_ridge.py", "--alpha-rel", AR, "--name", f"{dname}_ff{s}_a{AR}",
               *[g for p in DPATHS for g in (*dino_groups(emb16(p, s), 11), *mix_group(emb16(p, s)))]]])
        for p in ("U", "X", "F"):
            step(f"4c. TIPSv2 16보기 σ{s} ({p})", [ROOT / embT(p, s) / "emb.npz"],
                 [["src/probe_models.py", "--model", TIPS, "--size", "448", "--layers", *TIPS_LAYERS, "--pool", "phase2m", "--tta", "8", "--shifts", "2",
                   "--bs", "8", "--src", f"outputs/denoised/{ff(p, s)}.npz"]])
            step(f"5c. TIPSv2 Ridge σ{s} ({p})", [O / "emb_reg" / f"{tname[p]}_ff{s}_a{AR}" / "test_pred_full.csv"],
                 [["src/multi_ridge.py", "--alpha-rel", AR, "--name", f"{tname[p]}_ff{s}_a{AR}", *one_groups(embT(p, s), "0,1,2,3", 1152)]])
        for p in ("U", "X"):
            for m, n in CNNF:
                step(f"4d. 고정 CNN {n} σ{s} ({p})", [ROOT / embC(m, p, s) / "emb.npz"],
                     [["src/probe_models.py", "--model", f"cnn:{m}", "--size", "512", "--pool", "phase2m", "--tta", "8", "--bs", "16",
                       "--src", f"outputs/denoised/{ff(p, s)}.npz"]])
            step(f"5d. 고정 CNN 결합 Ridge σ{s} ({p})", [O / "emb_reg" / f"{cname[p]}_ff{s}_a{AR}" / "test_pred_full.csv"],
                 [["src/multi_ridge.py", "--alpha-rel", AR, "--name", f"{cname[p]}_ff{s}_a{AR}", *[g for m, _ in CNNF for g in one_groups(embC(m, p, s), "0", 512)]]])

    s48 = ff("U", 48)
    fc = O / "features_cache"
    step("6. 상분율·수작업·평활 CDF 특징 (σ48, U)",
         [fc / f"phase_{s48}_test.csv", fc / f"{s48}_test.csv", fc / f"smcdf3_{s48}_test.csv"],
         [["src/phase_features.py", "--src", f"outputs/denoised/{s48}.npz"],
          ["src/features_from_cache.py", "--src", f"outputs/denoised/{s48}.npz"],
          ["src/smooth_cdf_features.py", "--src", f"outputs/denoised/{s48}.npz"]])
    step("7. DINO + 측정 특징 Ridge", [O / f"emb_reg/mr_v14_pp_sc_a{AR}/test_pred_full.csv"],
         [["src/multi_ridge.py", "--alpha-rel", AR, "--name", f"mr_v14_pp_sc_a{AR}", *dino_groups(emb4(48), 10),
           "--group", f"csv:outputs/features_cache/phase_{s48}:0.381:pf_",
           "--group", f"csv:outputs/features_cache/{s48}:0.019:ph_",
           "--group", f"csv:outputs/features_cache/{s48}:0.15:(pore|gr|sp)_",
           "--group", f"csv:outputs/features_cache/smcdf3_{s48}:0.2:sc"]])

    # 가중치(합으로 정규화): 측정 블록 0.15 : DINOv2 4보기 0.12375 : DINOv2 16보기 0.19125 : TIPSv2 0.175 : 고정 CNN 결합 0.21, 예측 폭 ×1.03
    # 그룹 안의 폴더는 같은 가중으로 평균 → TIPSv2 = 세 경로 예측 평균, 고정 CNN = 두 경로 예측 평균 (각각 보정 4종 평균)
    step("8. 최종 블렌드", [ROOT / "submission/submission.csv"],
         [["src/final_blend.py", "--out", "submission/submission.csv", "--weights", "0.15", "0.12375", "0.19125", "0.175", "0.21",
           "--scale", "1.03",
           "--group", "phase", f"outputs/emb_reg/mr_v14_pp_sc_a{AR}",
           "--group", "dino4", *[f"outputs/emb_reg/dnpp0.25_ff{s}_a{AR}" for s in FF],
           "--group", "dino16", *[f"outputs/emb_reg/{dname}_ff{s}_a{AR}" for s in FF],
           "--group", "tips", *[f"outputs/emb_reg/{tname[p]}_ff{s}_a{AR}" for p in ("U", "X", "F") for s in FF],
           "--group", "cnn", *[f"outputs/emb_reg/{cname[p]}_ff{s}_a{AR}" for p in ("U", "X") for s in FF]]])


if __name__ == "__main__":
    main()
