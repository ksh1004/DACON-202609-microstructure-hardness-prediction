"""Train-only comparison of structure features, with repeated outer splits.

Every scaler and alpha search is fitted within the outer training fold.
Test prediction is a separate command after selecting a configuration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'structure_v1'


def folds_for(train, seed):
    if seed == 42:
        f = pd.read_csv(ROOT / 'outputs' / 'folds.csv').set_index('ID').loc[train.ID]
        return f.fold.to_numpy()
    f = np.zeros(len(train), dtype=int)
    bins = pd.qcut(train.hardness, 10, labels=False)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(train, bins)):
        f[va] = k
    return f


def load_features(split, ids):
    a = pd.read_csv(ROOT / 'outputs' / 'features' / f'{split}_features.csv').set_index('ID').loc[ids]
    a = a[[c for c in a if not c.startswith('nui_')]]
    b = pd.read_csv(OUT / f'{split}_features.csv').set_index('ID').loc[ids]
    assert a.index.is_unique and b.index.is_unique
    return a, b


def feature_sets(a, b):
    g = b[[c for c in b if c.startswith('v2_gr')]]
    return {'baseline': a, 'structure': b, 'grains_added': pd.concat([a, g], axis=1),
            'fused': pd.concat([a, b], axis=1)}


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 142, 242])
    ap.add_argument('--sets', nargs='+', default=['baseline', 'structure', 'grains_added', 'fused'])
    ap.add_argument('--regressor', choices=['ridge', 'svr'], default='ridge')
    ap.add_argument('--predict-test', choices=['baseline', 'structure', 'grains_added', 'fused'])
    args = ap.parse_args()
    out = OUT / ('evaluation' if args.regressor == 'ridge' else 'evaluation_svr')
    out.mkdir(parents=True, exist_ok=True)
    if args.predict_test:
        ss = pd.read_csv(ROOT / 'data' / 'sample_submission.csv')
        a, b = load_features('test', ss.ID)
        X = feature_sets(a, b)[args.predict_test]
        models = [joblib.load(out / args.predict_test / 'split42' / f'fold{k}.joblib') for k in range(5)]
        ss.hardness = np.mean([m.predict(X) for m in models], axis=0)
        assert np.isfinite(ss.hardness).all()
        ss.to_csv(out / args.predict_test / 'test_pred.csv', index=False)
        print('Saved inference only:', args.predict_test, flush=True)
        return
    tr = pd.read_csv(ROOT / 'data' / 'train.csv')
    a, b = load_features('train', tr.ID)
    sets = feature_sets(a, b)
    y, noise = tr.hardness.to_numpy(), a.q_noise_sigma.to_numpy()
    records = []
    t0 = time.time()
    with threadpool_limits(limits=2):
        for seed in args.seeds:
            f = folds_for(tr, seed)
            for name in args.sets:
                X = sets[name]
                assert np.isfinite(X.to_numpy()).all()
                dst = out / name / f'split{seed}'; dst.mkdir(parents=True, exist_ok=True)
                pred = np.full(len(tr), np.nan)
                alphas = []
                for k in range(5):
                    if args.regressor == 'ridge':
                        estimator = make_pipeline(StandardScaler(), Ridge())
                        grid = {'ridge__alpha': np.logspace(-1, 5, 9)}
                    else:
                        estimator = make_pipeline(StandardScaler(), TransformedTargetRegressor(
                            regressor=SVR(epsilon=.1), transformer=StandardScaler()))
                        grid = {'transformedtargetregressor__regressor__C': [1, 3, 10, 30],
                                'transformedtargetregressor__regressor__gamma': ['scale', .001, .01]}
                    m = GridSearchCV(estimator, grid,
                        cv=KFold(4, shuffle=True, random_state=seed + k),
                        scoring='neg_mean_squared_error', n_jobs=1)
                    m.fit(X.iloc[f != k], y[f != k])
                    pred[f == k] = m.predict(X.iloc[f == k])
                    alphas.append(m.best_params_)
                    joblib.dump(m.best_estimator_, dst / f'fold{k}.joblib')
                assert np.isfinite(pred).all()
                pd.DataFrame({'ID': tr.ID, 'fold': f, 'hardness': y, 'pred': pred}).to_csv(dst / 'oof.csv', index=False)
                row = {'name': name, 'regressor': args.regressor, 'split_seed': seed, 'n_features': X.shape[1],
                       'rmse': rmse(y, pred), 'low_noise': rmse(y[noise <= 10], pred[noise <= 10]),
                       'high_noise': rmse(y[noise > 10], pred[noise > 10]),
                       'fold_rmse': [rmse(y[f == k], pred[f == k]) for k in range(5)], 'alpha': alphas}
                records.append(row)
                print(json.dumps(row), flush=True)
                pd.DataFrame(records).to_csv(out / 'results.csv', index=False)
                (out / 'results.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Done in {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
