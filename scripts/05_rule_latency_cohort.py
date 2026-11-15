"""Step 5 - instantaneous rule alarm burden on MIMIC-IV test patients (Table 8), inference time (Section 4.10),
cohort characteristics (Table 2) and patient-flow counts (Fig. 2)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch
from google.colab import auth
from google.cloud import bigquery

import config as C
from src.gina_rule import alarm_burden, gina_decision
from src.models import ConformerAge
from src import train as T

d = np.load(f'{C.MIMIC_DIR}/windows_12h_2h_gina2026.npz', allow_pickle=True)
vh = pd.read_parquet(f'{C.MIMIC_DIR}/vitals_hourly.parquet')

# Table 8: alarm burden of the rule on every hour of the test patients
test_ids = set(np.unique(d['subject'][d['split'] == 'test']))
t = vh[vh.subject_id.isin(test_ids)].sort_values(['stay_id', 'hr_idx'])
print('test patients:', t.subject_id.nunique(), '| stays:', t.stay_id.nunique())
print(alarm_burden(t, 'stay_id', 'spo2', 'resp_rate', 'anchor_age').round(2))

# inference time: median and 95th percentile of 30 single decisions after 20 warm-up runs, CPU
members = [m.cpu() for m in T.load_ensemble(ConformerAge, f'{C.OUT_DIR}/conformer')]
x1, a1 = torch.zeros(1, 12, 3), torch.zeros(1)
with torch.no_grad():
    for _ in range(20):
        torch.stack([torch.sigmoid(m(x1, a1)) for m in members]).mean()
    ms = []
    for _ in range(30):
        s = time.perf_counter(); torch.stack([torch.sigmoid(m(x1, a1)) for m in members]).mean()
        ms.append((time.perf_counter() - s) * 1000)
print(f'Conformer ensemble: median {np.median(ms):.2f} ms | p95 {np.percentile(ms, 95):.2f} ms')
ms = []
for _ in range(30):
    s = time.perf_counter(); gina_decision(90.0, 25.0, 80.0, 46); ms.append((time.perf_counter() - s) * 1000)
print(f'GINA rule: median {np.median(ms):.4f} ms')

# Table 2 and Fig. 2: cohort characteristics and patient flow (BigQuery)
auth.authenticate_user(); bq = bigquery.Client(project=C.GCP_PROJECT)
A = f"""SELECT DISTINCT hadm_id FROM `{C.MIMIC_HOSP}.diagnoses_icd`
        WHERE (icd_version = 9 AND icd_code LIKE '493%') OR (icd_version = 10 AND icd_code LIKE 'J45%')"""
cohort = bq.query(f"""SELECT i.subject_id, i.stay_id, i.los, ad.hospital_expire_flag, p.gender
    FROM `{C.MIMIC_ICU}.icustays` i JOIN ({A}) a USING (hadm_id)
    JOIN `{C.MIMIC_HOSP}.admissions` ad USING (hadm_id) JOIN `{C.MIMIC_HOSP}.patients` p ON p.subject_id = i.subject_id""").to_dataframe()
rows = {}
for s in ['train', 'val', 'test']:
    ids = set(np.unique(d['subject'][d['split'] == s])); c = cohort[cohort.subject_id.isin(ids)]
    m = (d['split'] == s) & (d['flag'] == 0); ag = d['age'][d['split'] == s]; xl = d['X'][m][:, -1]
    rows[s] = {'patients': len(ids), 'age, median (IQR)': f'{np.median(ag):.0f} ({np.percentile(ag, 25):.0f}-{np.percentile(ag, 75):.0f})',
               'female, %': f'{(c.drop_duplicates("subject_id").gender == "F").mean() * 100:.1f}',
               'ICU stays': c.stay_id.nunique(), 'ICU length of stay, days': f'{c.los.median():.1f}',
               'in-hospital mortality, %': f'{c.drop_duplicates("stay_id").hospital_expire_flag.mean() * 100:.1f}',
               'evaluated windows': int(m.sum()), 'episodes': int(d['y'][m].sum()),
               **{f'{n}, median': f'{np.median(xl[:, k]):.0f}' for k, n in enumerate(['HR', 'RR', 'SpO2'])}}
print(pd.DataFrame(rows).to_string())
flow = bq.query(f"""WITH a AS ({A}),
 s AS (SELECT i.subject_id, i.stay_id, p.anchor_age FROM `{C.MIMIC_ICU}.icustays` i JOIN a USING (hadm_id)
       JOIN `{C.MIMIC_HOSP}.patients` p USING (subject_id)),
 v AS (SELECT DISTINCT stay_id FROM `{C.MIMIC_ICU}.chartevents` WHERE itemid IN (220045, 220210, 220277)
       AND stay_id IN (SELECT stay_id FROM s))
SELECT (SELECT COUNT(DISTINCT subject_id) FROM `{C.MIMIC_HOSP}.patients`) AS all_patients,
       (SELECT COUNT(DISTINCT subject_id) FROM `{C.MIMIC_ICU}.icustays`) AS icu_patients,
       (SELECT COUNT(DISTINCT s.subject_id) FROM s JOIN v USING (stay_id) WHERE anchor_age >= 18) AS adult_asthma_with_vitals,
       (SELECT COUNT(DISTINCT s.stay_id) FROM s JOIN v USING (stay_id) WHERE anchor_age >= 18) AS their_stays""").to_dataframe()
print(flow.T.to_string()); print('with >= 1 complete window:', len(np.unique(d['subject'])))
