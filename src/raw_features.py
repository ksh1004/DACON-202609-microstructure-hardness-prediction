"""원본 이미지 수작업 특징 캐시 생성 (eda/run_eda.py의 get_features와 동일, 그림·분석 없이).

재현 파이프라인에서 필요한 값: q_noise_sigma (디노이저 정답 이미지 선택 σ<7, 조명 보정 대상 저품질 모드 σ≥10).
이미지별 고정 연산이며 정답(경도)·데이터셋 통계를 쓰지 않는다.

실행: python src/raw_features.py
산출물: outputs/features/{train,test}_features.csv
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import extract_features  # noqa: E402

DATA = ROOT / "data"
FEAT_DIR = ROOT / "outputs" / "features"


def _feat_one(split, id_):
    return {"ID": id_, **extract_features(np.array(Image.open(DATA / split / f"{id_}.png")))}


def main():
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    for split, csv in (("train", "train.csv"), ("test", "sample_submission.csv")):
        path = FEAT_DIR / f"{split}_features.csv"
        ids = pd.read_csv(DATA / csv).ID.tolist()
        if path.exists() and len(pd.read_csv(path)) == len(ids):
            print(f"[raw_features] {split}: 캐시 사용 {path.relative_to(ROOT)}")
            continue
        t = time.time()
        # joblib(loky)은 임시폴더 경로에 비ASCII(한글 사용자명)가 있으면 실패 → 표준 ProcessPool 사용
        with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
            rows = list(ex.map(_feat_one, [split] * len(ids), ids, chunksize=8))
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[raw_features] {split}: {len(rows)}장 ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
