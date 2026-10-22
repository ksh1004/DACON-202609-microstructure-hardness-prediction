"""Reproduce the confirmed v4 baseline and audit TRAIN OOF failures only.

Stored test predictions are checked solely to identify the submitted artifact.
No model is fitted, no labels are inferred from leaderboard feedback, and
diagnostic feature disagreements are not treated as segmentation ground truth.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CNN = ['s512_convnext_tiny_s42', 's512_effnet_b2_s42', 's512_resnet50_s42']


def aligned(path, ids):
    f = pd.read_csv(path)
    assert len(f) == len(ids) and f.ID.is_unique and set(f.ID) == set(ids), path
    return f.set_index('ID').loc[ids]


def main():
    tr = pd.read_csv(ROOT / 'data/train.csv')
    ids, y = tr.ID, tr.hardness.to_numpy()
    folds = aligned(ROOT / 'outputs/folds.csv', ids).fold.to_numpy()
    dirs = [ROOT / 'outputs/cnn' / n for n in CNN] + [ROOT / 'outputs/feat_ridge']
    predictions = []
    for d in dirs:
        f = aligned(d / 'oof.csv', ids)
        assert np.array_equal(f.fold, folds) and np.allclose(f.hardness, y)
        predictions.append(f.pred.to_numpy())
    P = np.column_stack(predictions)
    assert np.isfinite(P).all()
    p = P.mean(1)
    old = aligned(ROOT / 'outputs/features/train_features.csv', ids)
    new = aligned(ROOT / 'outputs/structure_v1/train_features.csv', ids)
    ratio = np.maximum(old.gr_count, new['v2_gr0.1_count']) / np.maximum(
        np.minimum(old.gr_count, new['v2_gr0.1_count']), 1)
    table = pd.DataFrame({'ID': ids.to_numpy(), 'fold': folds, 'hardness': y, 'pred': p,
                         'error': p - y, 'squared_error': (p - y) ** 2,
                         'noise_sigma': old.q_noise_sigma.to_numpy(),
                         'model_prediction_std': P.std(1),
                         'all_models_same_error_sign': (P > y[:, None]).all(1) | (P < y[:, None]).all(1),
                         'old_grain_count': old.gr_count.to_numpy(),
                         'nlm_grain_count_h004': new['v2_gr0.04_count'].to_numpy(),
                         'nlm_grain_count_h010': new['v2_gr0.1_count'].to_numpy(),
                         'grain_method_ratio': ratio.to_numpy()}).sort_values('squared_error', ascending=False)
    top = table.head(50)
    # Verify that the OOF composition corresponds to the user's named submission.
    subpath = ROOT / 'submissions/sub_v4_big512.csv'
    sub = pd.read_csv(subpath)
    reconstructed = np.mean([aligned(d / 'test_pred.csv', sub.ID).hardness.to_numpy() for d in dirs], axis=0)
    artifact_delta = float(np.max(np.abs(sub.hardness.to_numpy() - reconstructed)))
    assert artifact_delta < 1e-8
    corr = float(pd.Series(np.abs(p-y)).corr(pd.Series(ratio.to_numpy()), method='spearman'))
    summary = {'confirmed_on': '2026-10-05', 'submission': subpath.relative_to(ROOT).as_posix(),
               'public_score_user_confirmed': 12.3168181193,
               'submission_sha256': hashlib.sha256(subpath.read_bytes()).hexdigest(),
               'submission_reconstruction_max_abs_diff': artifact_delta,
               'oof_rmse': float(np.sqrt(np.mean((p-y)**2))),
               'top50_sse_share': float(top.squared_error.sum() / table.squared_error.sum()),
               'top50_high_noise_count': int((top.noise_sigma > 10).sum()),
               'top50_low_noise_count': int((top.noise_sigma <= 10).sum()),
               'top50_all_models_same_sign_count': int(top.all_models_same_error_sign.sum()),
               'top50_median_model_prediction_std': float(top.model_prediction_std.median()),
               'remaining450_median_model_prediction_std': float(table.iloc[50:].model_prediction_std.median()),
               'grain_method_disagreement_abs_error_spearman': corr,
               'interpretation': 'Shared errors persist at low noise. Grain-method disagreement does not rank errors.',
               'limits': 'Post-hoc train OOF diagnostics. No causal attribution, true segmentation masks, or label-noise estimate.'}
    out = ROOT / 'outputs/v4_error_audit'; out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'train_errors.csv', index=False)
    top.to_csv(out / 'top50_train_errors.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
