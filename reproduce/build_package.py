"""최종 제출 코드 패키지 조립: final_package/ 에 재현에 필요한 코드·가중치만 복사 (실험 산출물과 분리).

실행: python reproduce/build_package.py [--with-data]
  --with-data       : data/ 도 복사 (로컬 재현 테스트용; 주최측 검증 시에는 제공 데이터를 data/ 에 둔다)
결과: final_package/{run_final.py, README.md, requirements.txt, src/, weights/(디노이저 2개), outputs/folds.csv}
(v22a부터 미세조정 CNN을 쓰지 않으므로 CNN 가중치는 넣지 않는다. train_cnn.py는 이미지 로딩·D4 변환 함수 때문에 남긴다.)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "final_package"
SRC = ["raw_features.py", "features.py", "structure_features.py", "train_denoiser.py", "denoise_with.py", "train_cnn.py",
       "flatfield_cache.py", "probe_models.py", "phase_features.py", "features_from_cache.py",
       "smooth_cdf_features.py", "multi_ridge.py", "final_blend.py", "drunet_net.py", "drunet_denoise.py"]
WEIGHTS = ["denoiser_cm7_am22_l1_i6000.pt", "denoiser_cm7_am22_l1_i10000_boot.pt", "pretrained_denoiser/drunet_gray.pth"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-data", action="store_true")
    a = ap.parse_args()
    PKG.mkdir(exist_ok=True)
    (PKG / "src").mkdir(exist_ok=True)
    for f in SRC:
        shutil.copy2(ROOT / "src" / f, PKG / "src" / f)
    shutil.copy2(ROOT / "reproduce/run_final.py", PKG / "run_final.py")
    shutil.copy2(ROOT / "reproduce/README.md", PKG / "README.md")
    shutil.copy2(ROOT / "requirements.txt", PKG / "requirements.txt")
    (PKG / "weights").mkdir(exist_ok=True)
    for w in WEIGHTS:
        (PKG / "weights" / w).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "weights" / w, PKG / "weights" / w)
    (PKG / "outputs").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "outputs/folds.csv", PKG / "outputs/folds.csv")
    if a.with_data:
        shutil.copytree(ROOT / "data", PKG / "data", dirs_exist_ok=True)
    print(f"[build_package] → {PKG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
