"""Step 1 - MIMIC-IV cohort, hourly vitals, 12 h windows with GINA 2026 labels, patient split, standardisation stats.
Run in Google Colab with BigQuery access to MIMIC-IV v3.1 (credentialed PhysioNet account).
Outputs stay on your own Drive; they contain patient-level data and must not be shared."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from google.colab import auth
from google.cloud import bigquery

import config as C
from src import data_mimic as dm

auth.authenticate_user()
client = bigquery.Client(project=C.GCP_PROJECT)
os.makedirs(C.MIMIC_DIR, exist_ok=True)

vh = dm.query_hourly(client)
vh = vh[vh.anchor_age >= C.MIN_AGE]
vh.to_parquet(f'{C.MIMIC_DIR}/vitals_hourly.parquet', index=False)
print('adult asthma ICU patients with vitals:', vh.subject_id.nunique(), '| stays:', vh.stay_id.nunique())

d = dm.build_windows(vh)
d['split'] = dm.assign_split(d['subject'], dm.split_patients(vh))
np.savez_compressed(f'{C.MIMIC_DIR}/windows_12h_2h_gina2026.npz', **d)


print('windows:', len(d['y']), '| patients with >= 1 complete window:', len(np.unique(d['subject'])))
for s in ['train', 'val', 'test']:
    m = (d['split'] == s) & (d['flag'] == 0)
    print(f'{s:5s} patients {len(np.unique(d["subject"][d["split"] == s])):5d} | evaluated windows {m.sum():7d} | '
          f'episodes {d["y"][m].sum():5d} ({100 * d["y"][m].mean():.2f}%)')
dm.check_split(d)
dm.check_labels(d)
