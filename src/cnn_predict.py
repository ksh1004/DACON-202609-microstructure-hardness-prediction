"""저장된 CNN 폴드 가중치로 test 예측만 다시 만든다 (학습 없음, 재현·검증용).

train_cnn.py --save-weights 로 저장된 outputs/cnn/<exp>/weights/fold{k}.pt 와 log.json의 학습 인자로 모델을 재구성하고,
학습 때와 같은 D4 8방향 TTA로 test를 예측해 5개 폴드 평균을 낸다.

예) python src/cnn_predict.py --exp s512_convnext_tiny_s43            # outputs/cnn/<exp>/test_pred_reproduced.csv
    python src/cnn_predict.py --exp s512_convnext_tiny_s43 --check    # 학습 때 저장한 test_pred.csv와 비교
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cnn import DATA, OUT, build_model, build_parser, finalize_args, load_split, predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def restore_args(log_args: dict):
    """log.json의 문자열 인자를 train_cnn 파서로 다시 해석 (기본값·형 변환을 학습 때와 동일하게)"""
    p = build_parser()
    a = p.parse_args([])
    for k, v in log_args.items():
        if not hasattr(a, k) or k in ("device", "amp_dtype"):
            continue
        d = getattr(a, k)
        if v == "None":
            val = None
        elif isinstance(d, bool):
            val = v == "True"
        elif isinstance(d, int) and not isinstance(d, bool):
            val = int(v)
        elif isinstance(d, float):
            val = float(v)
        elif isinstance(d, list) or (d is None and v.startswith("[")):
            val = json.loads(v) if v.startswith("[") else None
        else:
            val = v
        setattr(a, k, val)
    return finalize_args(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--check", action="store_true", help="저장된 test_pred.csv와 차이 출력")
    o = ap.parse_args()
    d = OUT / "cnn" / o.exp
    log = json.loads((d / "log.json").read_text(encoding="utf-8"))
    a = restore_args(log["args"])
    a.scratch, a.init_weights = True, None  # 사전학습 가중치 내려받기 생략 (저장 가중치로 덮어씀)
    test = pd.read_csv(DATA / "sample_submission.csv")
    if a.src:
        Xte = torch.from_numpy(np.load(ROOT / a.src)["test"]).unsqueeze(1)
    else:
        Xte = load_split("test", test.ID)
    preds = []
    for k in range(5):
        ck = torch.load(d / "weights" / f"fold{k}.pt", map_location="cpu", weights_only=False)
        m = build_model(a, k).to(a.device)
        m.load_state_dict(ck["state_dict"])
        preds.append(predict(m, Xte, ck["mean"], ck["std"], a, tta=True) * ck["ysd"] + ck["ymu"])
        del m
        torch.cuda.empty_cache()
    tp = np.mean(preds, 0)
    pd.DataFrame({"ID": test.ID, "hardness": tp}).to_csv(d / "test_pred_reproduced.csv", index=False)
    msg = f"[{o.exp}] test 예측 재생성 → {(d / 'test_pred_reproduced.csv').relative_to(ROOT)}"
    if o.check:
        ref = pd.read_csv(d / "test_pred.csv").set_index("ID").loc[test.ID, "hardness"].values
        msg += f" | 저장본과 최대 차이 {np.abs(ref - tp).max():.4f}, RMS {np.sqrt(np.mean((ref - tp) ** 2)):.4f}"
    print(msg)


if __name__ == "__main__":
    main()
