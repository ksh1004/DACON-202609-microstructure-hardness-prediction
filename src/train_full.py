"""전체 train 500장으로 최종 모델 학습 → test 예측 (검증 폴드 없음).

K-Fold 모델은 각각 80%(400장)로 학습된다. 최종 제출용으로는 같은 레시피를 100% 데이터에 적용한 모델이 보통 더 좋다.
검증 점수가 없으므로 레시피(에폭·lr·증강)는 반드시 K-Fold CV로 먼저 정한 것을 그대로 쓴다.

예) python src/train_full.py --model convnext_nano.in12k_ft_in1k --lr 3e-4 --rot-p 0.5 --noise 16 --seed 42 --exp full_convnext_s42
산출물: outputs/full/<exp>/{test_pred.csv, log.json, model.pt}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_cnn as T  # noqa: E402


def main():
    # train_cnn의 인자 정의를 그대로 재사용 (--folds, --save-weights 등 일부는 무시됨)
    a = T.finalize_args(T.build_parser().parse_args())

    train = pd.read_csv(T.DATA / "train.csv")
    test = pd.read_csv(T.DATA / "sample_submission.csv")
    X = T.load_split("train", train.ID)
    Xte = T.load_split("test", test.ID)
    y = train.hardness.values.astype(np.float64)
    out = T.OUT / "full" / a.exp
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{a.exp}] FULL-DATA {a.model} | n={len(X)} | amp={a.amp_dtype}")
    t0 = time.time()
    a.save_weights = False
    for attempt in range(a.max_retry + 1):
        # 검증 자리에는 train 일부(8장)를 넣어 함수 형태만 맞춤 — 점수는 사용하지 않음
        _, pt, hist = T.train_fold(0, X, y, X[:8], y[:8], Xte, a, attempt)
        if hist[-1]["train_loss"] <= a.retry_loss or attempt == a.max_retry:
            break
        print(f"  학습 실패 감지 (train loss {hist[-1]['train_loss']:.3f}) → 재시도 {attempt + 1}", flush=True)
    pd.DataFrame({"ID": test.ID, "hardness": pt}).to_csv(out / "test_pred.csv", index=False)
    with open(out / "log.json", "w", encoding="utf-8") as f:
        json.dump({"exp": a.exp, "model": a.model, "minutes": (time.time() - t0) / 60, "retries": attempt,
                   "final_train_loss": hist[-1]["train_loss"], "args": {k: str(v) for k, v in vars(a).items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"[{a.exp}] done {(time.time() - t0) / 60:.1f} min, final train loss {hist[-1]['train_loss']:.3f}")


if __name__ == "__main__":
    main()
