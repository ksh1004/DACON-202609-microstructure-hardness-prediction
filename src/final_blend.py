"""최종 제출 파일: 그룹별 test 예측 평균 → 고정 가중 평균 (검증 점수 없이 test 예측만 필요).

그룹 폴더의 test 예측 파일 우선순위: test_pred_full.csv(Ridge 전체 500장 재적합) > test_pred_reproduced.csv(저장 가중치 추론)
> test_pred.csv(폴드 평균). oof.csv가 모두 있으면 OOF RMSE도 출력.

예) python src/final_blend.py --out submission.csv --weights 0.1 0.4 0.5 \
      --group cnn outputs/cnn/s512_convnext_tiny_s42 ... --group dino outputs/emb_reg/dn2b_fflowpass48_low_L11-17 ... \
      --group phase outputs/emb_reg/dnph_ff48_pgs_L11-17
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_pred(d: Path, ids) -> tuple[np.ndarray, str]:
    for name in ("test_pred_full.csv", "test_pred_reproduced.csv", "test_pred.csv"):
        if (d / name).exists():
            return pd.read_csv(d / name).set_index("ID").loc[ids, "hardness"].values, name
    raise FileNotFoundError(f"{d}: test 예측 파일 없음")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", action="append", required=True, help="그룹이름 폴더1 폴더2 ...")
    ap.add_argument("--weights", type=float, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="블렌드 예측의 폭 배율 (train 경도 평균 기준). 구성원을 평균할수록 예측이 가운데로 몰리는 것을 되돌린다 (기본 1 = 미적용)")
    a = ap.parse_args()
    assert len(a.weights) == len(a.group)
    te = pd.read_csv(ROOT / "data/sample_submission.csv")
    tr = pd.read_csv(ROOT / "data/train.csv")
    w = np.array(a.weights) / sum(a.weights)
    tp, oof, has_oof = np.zeros(len(te)), np.zeros(len(tr)), True
    for (gname, *dirs), g in zip(a.group, w):
        preds, srcs = zip(*[test_pred(ROOT / d, te.ID) for d in dirs])
        tp += g * np.mean(preds, 0)
        try:
            oof += g * np.mean([pd.read_csv(ROOT / d / "oof.csv").set_index("ID").loc[tr.ID, "pred"].values for d in dirs], 0)
        except FileNotFoundError:
            has_oof = False
        print(f"  {gname:8s} w={g:.3f} n={len(dirs)} test 예측: " + ", ".join(sorted(set(srcs))))
    if a.scale != 1.0:   # 고정 배율: test 통계는 쓰지 않는다 (기준점 = train 경도 평균)
        c = tr.hardness.values.mean()
        tp, oof = c + a.scale * (tp - c), c + a.scale * (oof - c)
        print(f"  예측 폭 ×{a.scale:g} (기준점 = train 평균 {c:.4f})")
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": te.ID, "hardness": tp}).to_csv(out, index=False)
    msg = f"[final_blend] → {out.relative_to(ROOT)} (test mean {tp.mean():.2f}, std {tp.std():.2f})"
    if has_oof:
        msg += f" | OOF RMSE {np.sqrt(np.mean((oof - tr.hardness.values) ** 2)):.3f}"
    print(msg)


if __name__ == "__main__":
    main()
