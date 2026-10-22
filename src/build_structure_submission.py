"""Build the pre-specified 25%-per-block structure-feature comparison candidate.

This is a review candidate, not a demonstrated Public/Private improvement.
No weights are fitted, and no test labels/statistics are used for selection.
Run evaluate_structure.py --predict-test fused first.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['s512_convnext_tiny_s42', 's512_effnet_b2_s42', 's512_resnet50_s42']


def aligned(path, ids, column):
    f = pd.read_csv(path)
    assert f.ID.is_unique and len(f) == len(ids), path
    assert set(f.ID) == set(ids), path
    f = f.set_index('ID').loc[ids]
    assert np.isfinite(f[column].to_numpy()).all(), path
    return f


def main():
    train = pd.read_csv(ROOT / 'data' / 'train.csv')
    ss = pd.read_csv(ROOT / 'data' / 'sample_submission.csv')
    folds = aligned(ROOT / 'outputs' / 'folds.csv', train.ID, 'fold').fold.to_numpy()
    paths, op, tp = [], [], []
    for name in MODELS:
        d = ROOT / 'outputs' / 'cnn' / name
        opath, tpath = d / 'oof.csv', d / 'test_pred.csv'
        o = aligned(opath, train.ID, 'pred')
        assert np.array_equal(o.fold, folds) and np.allclose(o.hardness, train.hardness)
        op.append(o.pred.to_numpy())
        tp.append(aligned(tpath, ss.ID, 'hardness').hardness.to_numpy())
        paths.extend([opath, tpath])
    d = ROOT / 'outputs' / 'structure_v1' / 'evaluation' / 'fused'
    opath, tpath = d / 'split42' / 'oof.csv', d / 'test_pred.csv'
    o = aligned(opath, train.ID, 'pred')
    assert np.array_equal(o.fold, folds) and np.allclose(o.hardness, train.hardness)
    op.append(o.pred.to_numpy())
    tp.append(aligned(tpath, ss.ID, 'hardness').hardness.to_numpy())
    paths.extend([opath, tpath])
    oof = np.mean(op, axis=0)
    ss.hardness = np.mean(tp, axis=0)
    score = float(np.sqrt(np.mean((oof - train.hardness.to_numpy()) ** 2)))
    sub = ROOT / 'submissions' / 'sub_structure_v1_fused.csv'
    ss.to_csv(sub, index=False)
    pd.DataFrame({'ID': train.ID, 'fold': folds, 'hardness': train.hardness, 'pred': oof}).to_csv(
        ROOT / 'outputs' / 'structure_v1' / 'candidate_oof.csv', index=False)
    manifest = {'submission': sub.relative_to(ROOT).as_posix(), 'status': 'review_only',
                'oof_rmse': score, 'public_score': None, 'private_score': None,
                'weights': {**{n: .25 for n in MODELS}, 'fused_structure_ridge': .25},
                'selection_note': 'OOF difference from old v4 is only about -0.0024; no material gain established.',
                'sources_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ROOT / 'outputs' / 'structure_v1' / 'submission_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'file': sub.name, 'rows': len(ss), 'columns': ss.columns.tolist(), 'oof_rmse': score,
                      'pred_min': float(ss.hardness.min()), 'pred_max': float(ss.hardness.max())}), flush=True)


if __name__ == '__main__':
    main()
