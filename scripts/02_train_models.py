"""Step 2 - train the Conformer and the three baselines under one protocol (same data, split, losses, budget).
Writes checkpoints and validation/test probabilities to C.OUT_DIR."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.metrics import roc_auc_score

import config as C
from src.data_mimic import load_sets
from src.models import ConformerAge, ResNet1D, PatchTST, n_params
from src import train as T

stats = dict(np.load(f'{C.MIMIC_DIR}/norm_stats_12h.npz'))
d, masks, sets = load_sets(f'{C.MIMIC_DIR}/windows_12h_2h_gina2026.npz', stats)
Dtr, Dva, Dte = (T.to_tensors(sets[k]) for k in ['train', 'val', 'test'])
print('train / val / test windows:', [len(s[2]) for s in sets.values()],
      '| positive weight:', round(float((sets['train'][2] == 0).sum() / sets['train'][2].sum()), 1))

networks = {'Conformer': (ConformerAge, C.CONFORMER_SEED), 'ResNet': (ResNet1D, C.SEED), 'PatchTST': (PatchTST, C.SEED)}
for name, (make, seed) in networks.items():
    out = f'{C.OUT_DIR}/{name.lower()}'
    print(f'\n=== {name} ({n_params(make()):,} parameters)')
    T.train_two_stage(make, Dtr, Dva, out, seed)
    members = T.load_ensemble(make, out)
    pv, pt = T.ensemble_predict(members, Dva), T.ensemble_predict(members, Dte)
    np.save(f'{out}/val_probs.npy', pv); np.save(f'{out}/test_probs.npy', pt)
    print(f'{name} ensemble | val AUC {roc_auc_score(sets["val"][2], pv):.4f} | test AUC {roc_auc_score(sets["test"][2], pt):.4f}')

# CatBoost on raw (unstandardised) values plus temporal summary features and age
out = f'{C.OUT_DIR}/catboost'; os.makedirs(out, exist_ok=True)
F = {k: T.catboost_features(d['X'][m], d['age'][m]) for k, m in masks.items()}
cb = T.train_catboost(F['train'], d['y'][masks['train']], F['val'], d['y'][masks['val']], f'{out}/catboost.cbm')
pv, pt = cb.predict_proba(F['val'])[:, 1], cb.predict_proba(F['test'])[:, 1]
np.save(f'{out}/val_probs.npy', pv); np.save(f'{out}/test_probs.npy', pt)
print(f'CatBoost | best iteration {cb.get_best_iteration()} | test AUC {roc_auc_score(d["y"][masks["test"]], pt):.4f}')
