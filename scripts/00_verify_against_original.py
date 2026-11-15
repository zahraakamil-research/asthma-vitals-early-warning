"""Check that the published data-preparation code rebuilds the dataset used in the paper.
Builds everything in memory (nothing is written) and compares it with the original files of the study.
Run in Colab after cloning this repository; ORIGINAL points to the study's own Drive folder."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from google.colab import auth
from google.cloud import bigquery

import config as C
from src import data_mimic as dm

ORIGINAL = '/content/drive/MyDrive/Path1_Vitals'
o = np.load(f'{ORIGINAL}/windows_12h_2h_sustained_gina2026.npz', allow_pickle=True)
ns = np.load(f'{ORIGINAL}/norm_stats_12h.npz')

auth.authenticate_user()
vh = dm.query_hourly(bigquery.Client(project=C.GCP_PROJECT))
vh = vh[vh.anchor_age >= C.MIN_AGE]
d = dm.build_windows(vh)
d['split'] = dm.patient_split(d['subject'])

print(f'windows: rebuilt {len(d["y"])} | original {len(o["y"])}')
print(f'patients: rebuilt {len(np.unique(d["subject"]))} | original {len(np.unique(o["subject"]))}')
for k in ['y', 'transient', 'flag']:
    print(f'{k:9s} sum: rebuilt {d[k].sum()} | original {o[k].sum()}')
print('sum of all input values equal:', np.isclose(d['X'].astype(np.float64).sum(), o['X'].astype(np.float64).sum(), rtol=1e-7))

# same patients in the same split, and same age per patient
sp_o = dict(zip(o['subject'], o['split'])); sp_d = dict(zip(d['subject'], d['split']))
same = [sp_o.get(s) == v for s, v in sp_d.items()]
print(f'patients in the same split: {sum(same)} of {len(same)}')
ag_o = dict(zip(o['subject'], o['age'])); ag_d = dict(zip(d['subject'], d['age']))
print('age identical for every patient:', all(np.isclose(ag_o.get(s, -1), a) for s, a in ag_d.items()))

# which training subset reproduces the saved standardisation statistics
for name, m in [('train, flag 0, no transient', (d['split'] == 'train') & (d['flag'] == 0) & (d['transient'] == 0)),
                ('train, flag 0', (d['split'] == 'train') & (d['flag'] == 0)),
                ('all train windows', d['split'] == 'train')]:
    st = dm.norm_stats(d, m)
    ok = np.allclose(st['mu'], ns['mu'], atol=1e-3) and np.allclose(st['sd'], ns['sd'], atol=1e-3) and \
        np.isclose(st['a_mu'], float(ns['a_mu']), atol=1e-3) and np.isclose(st['a_sd'], float(ns['a_sd']), atol=1e-3)
    print(f'standardisation from {name:30s}: matches saved file = {ok}')
