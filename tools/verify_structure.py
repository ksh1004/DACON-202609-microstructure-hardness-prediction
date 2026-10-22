"""Verify feature/model alignment, fold-only normalization, and submission provenance."""
from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluate_structure import load_features, feature_sets, folds_for


def main():
    out = ROOT / 'outputs' / 'structure_v1'
    tr = pd.read_csv(ROOT / 'data' / 'train.csv')
    ss = pd.read_csv(ROOT / 'data' / 'sample_submission.csv')
    assert tr.ID.is_unique and ss.ID.is_unique and not set(tr.ID) & set(ss.ID)
    a, b = load_features('train', tr.ID)
    X = feature_sets(a, b)['fused']
    fold = folds_for(tr, 42)
    assert not np.array_equal(fold, folds_for(tr, 142))
    checks = []
    d = out / 'evaluation' / 'fused' / 'split42'
    oof = pd.read_csv(d / 'oof.csv').set_index('ID').loc[tr.ID]
    for k in range(5):
        model = joblib.load(d / f'fold{k}.joblib')
        scaler = model.named_steps['standardscaler']
        expected = X.iloc[fold != k].mean().to_numpy()
        np.testing.assert_allclose(scaler.mean_, expected, rtol=1e-12, atol=1e-12)
        p = model.predict(X.iloc[fold == k])
        np.testing.assert_allclose(p, oof.pred.to_numpy()[fold == k], rtol=1e-12, atol=1e-10)
    checks.append('all five fused feature checkpoints reproduce OOF and use fold-only normalization')
    for split, ids in [('train', tr.ID), ('test', ss.ID)]:
        cache = np.load(out / f'{split}_views.npz')
        assert np.array_equal(cache['ID'], ids.to_numpy())
        assert cache['views'].shape == (len(ids), 2, 256, 256)
        assert cache['views'].dtype == np.uint8
    checks.append('feature-view cache IDs, shapes, and dtypes match source manifests')
    manifest = json.loads((out / 'submission_manifest.json').read_text())
    for rel, sha in manifest['sources_sha256'].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == sha
    sub = pd.read_csv(ROOT / manifest['submission'])
    assert sub.columns.tolist() == ss.columns.tolist()
    assert np.array_equal(sub.ID, ss.ID) and len(sub) == 1000
    assert np.isfinite(sub.hardness.to_numpy()).all()
    cand = pd.read_csv(out / 'candidate_oof.csv').set_index('ID').loc[tr.ID]
    score = float(np.sqrt(np.mean((cand.pred.to_numpy() - tr.hardness.to_numpy()) ** 2)))
    assert abs(score - manifest['oof_rmse']) < 1e-10
    checks.append('submission schema, 1000 IDs, finite predictions, source checksums, and OOF score verified')
    result = {'passed': True, 'checks': checks, 'candidate_oof_rmse': score}
    (out / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
