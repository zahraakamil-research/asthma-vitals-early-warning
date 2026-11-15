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
d['split'] = dm.assign_split(d['subject'], dm.split_patients(vh))

print(f'windows: rebuilt {len(d["y"])} | original {len(o["y"])}')
print(f'patients: rebuilt {len(np.unique(d["subject"]))} | original {len(np.unique(o["subject"]))}')
for k in ['y', 'transient', 'flag']:
    print(f'{k:9s} sum: rebuilt {d[k].sum()} | original {o[k].sum()}')
print('sum of all input values equal:', np.isclose(d['X'].astype(np.float64).sum(), o['X'].astype(np.float64).sum(), rtol=1e-7))

import collections
print('windows per patient identical:', collections.Counter(d['subject'].tolist()) == collections.Counter(o['subject'].tolist()))
# same patients in the same split, and same age per patient
sp_o = dict(zip(o['subject'], o['split'])); sp_d = dict(zip(d['subject'], d['split']))
same = [sp_o.get(s) == v for s, v in sp_d.items()]
print(f'patients in the same split: {sum(same)} of {len(same)}')
ag_o = dict(zip(o['subject'], o['age'])); ag_d = dict(zip(d['subject'], d['age']))
print('age identical for every patient:', all(np.isclose(ag_o.get(s, -1), a) for s, a in ag_d.items()))

# standardisation constants published in config.py equal the saved file used by the frozen model
print('published standardisation constants equal saved file:',
      np.allclose(C.NORM_STATS['mu'], ns['mu']) and np.allclose(C.NORM_STATS['sd'], ns['sd']) and
      np.isclose(C.NORM_STATS['a_mu'], float(ns['a_mu'])) and np.isclose(C.NORM_STATS['a_sd'], float(ns['a_sd'])))
