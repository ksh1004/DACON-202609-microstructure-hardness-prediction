"""저품질 촬영 모드의 톤 정렬 — 디노이즈(+조명보정) 이미지 캐시에서 이미지별 고정 연산 (라벨 미사용).

근거(2026-10-12): 두 촬영 모드의 경도 분포는 같은데(생성 파라미터 분포 동일), 저품질 모드는 어두운 쪽 계조가 압축되어
같은 기준(기지보다 7.5% 어두움)으로 잰 2상 분율 중앙값이 고품질 0.29 / 저품질 0.16 으로 어긋난다.
→ 기지 밝기 대비 상대 어둡기 rel=(m−x)/m 의 화소 분포를, train 저품질 전체 ↔ train 고품질 전체 분위 대응(단조 곡선 T)으로 맞춘다.
곡선은 train 이미지의 두 모드 분포만으로 정하고(--fit), 저품질(q_noise_sigma ≥ 10) 이미지에만 x' = m·(1 − T(rel)) 를 적용한다 (test도 같은 고정 곡선).

실행: python src/tone_align.py --src outputs/denoised/<stem>.npz
산출물: outputs/denoised/<stem>_tm.npz, weights/tone_curve_<stem>.npy (2×K: 저품질 분위값, 고품질 분위값)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_features import matrix_level  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QS = np.r_[np.linspace(0, 0.01, 21)[:-1], np.linspace(0.01, 0.99, 197), np.linspace(0.99, 1, 21)[1:]]


def rel_of(img):
    x = img.astype(np.float64)
    m = matrix_level(x)
    return (m - x) / m, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--strength", type=float, default=1.0, help="정렬 강도 (1 = 완전 분위 대응, 0 = 변화 없음)")
    a = ap.parse_args()
    stem = Path(a.src).stem
    z = np.load(ROOT / a.src)
    sig = {sp: pd.read_csv(ROOT / f"outputs/features/{sp}_features.csv").set_index("ID").loc[
        pd.read_csv(ROOT / "data" / csv).ID, "q_noise_sigma"].values for sp, csv in (("train", "train.csv"), ("test", "sample_submission.csv"))}
    lq = sig["train"] >= 10
    rels = [rel_of(im)[0].ravel() for im in z["train"]]
    qL = np.quantile(np.concatenate([r for r, f in zip(rels, lq) if f]), QS)
    qH = np.quantile(np.concatenate([r for r, f in zip(rels, lq) if not f]), QS)
    (ROOT / "weights").mkdir(exist_ok=True)
    np.save(ROOT / "weights" / f"tone_curve_{stem}.npy", np.stack([qL, qH]))
    print("톤 곡선 (저품질 rel → 고품질 rel): " + ", ".join(f"{l:+.3f}→{h:+.3f}" for l, h in zip(qL[[30, 70, 118, 168, 196, 207]], qH[[30, 70, 118, 168, 196, 207]])))
    out = {}
    for sp in ("train", "test"):
        res = []
        for im, s in zip(z[sp], sig[sp]):
            if s >= 10:
                rel, m = rel_of(im)
                t = np.interp(rel, qL, qH)
                im = np.clip(np.rint(m * (1 - (rel + a.strength * (t - rel)))), 0, 255).astype(np.uint8)
            res.append(im)
        out[sp] = np.stack(res)
    tag = "_tm" + ("" if a.strength == 1.0 else f"{a.strength:g}")
    p = ROOT / "outputs/denoised" / f"{stem}{tag}.npz"
    np.savez_compressed(p, **out)
    print(f"[tone_align] 저장 {p.relative_to(ROOT)} (저품질 train {lq.sum()}장, test {(sig['test'] >= 10).sum()}장 변환)")


if __name__ == "__main__":
    main()
