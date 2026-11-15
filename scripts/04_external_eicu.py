"""Step 4 - external test on eICU-CRD Demo v2.0.1 with everything frozen: build setting, standardisation,
Conformer ensemble and thresholds. Prespecified success criterion: AUC >= 0.80 with lower 95% bound >= 0.70.
Also reports the instantaneous rule's alarm burden (all ages)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config as C
from src import data_eicu as DE
from src import evaluate as E
from src import train as T
from src.gina_rule import gina_warning_tier, alarm_burden
from src.models import ConformerAge

DE.download()
pt, ast, adult = DE.cohort()
vh = DE.hourly(adult)
e = DE.build_windows(vh, adult)
os.makedirs(C.EICU_DIR, exist_ok=True)
np.savez_compressed(f'{C.EICU_DIR}/eicu_windows_12h_2h_gina2026.npz', **e)
em = e['flag'] == 0
print(f'evaluated windows {em.sum()} | episodes {e["y"][em].sum()} | stays {len(np.unique(e["stay"][em]))} | '
      f'patients {len(np.unique(e["patient"][em]))}')

# frozen Conformer ensemble and MIMIC-IV training statistics
stats = dict(np.load(f'{C.MIMIC_DIR}/norm_stats_12h.npz'))
x = ((e['X'][em] - stats['mu']) / stats['sd']).astype(np.float32)
a = ((e['age'][em] - stats['a_mu']) / stats['a_sd']).astype(np.float32)
y = e['y'][em]
D = T.to_tensors((x, a, y.astype(np.float32)))
members = T.load_ensemble(ConformerAge, f'{C.OUT_DIR}/conformer')
p = T.ensemble_predict(members, D)
np.save(f'{C.EICU_DIR}/eicu_conformer_probs.npy', p)

# AUC with stay-level bootstrap (2,000 resamples)
auc_ci, _ = E.bootstrap_auc(y, {'Conformer': p}, e['stay'][em])
auc, lo, hi = auc_ci['Conformer']
print(f'eICU AUC {auc:.3f} (95% CI {lo:.3f}-{hi:.3f}) | prespecified criterion met: {auc >= 0.80 and lo >= 0.70}')

seg, onset = E.timeline(e['X'], e['stay'], e['flag'], e['y'])
rule = gina_warning_tier(e['X'][em][:, -1])
for t_name, t in C.THRESHOLDS.items():
    h = E.hour_metrics(y, p, t)
    print(f'{t_name} {t}: ' + ' | '.join(f'{k} {v:.3f}' for k, v in h.items()))
for n, al in [('GINA rule (warning tier) alone', rule), ('Conformer alone (0.47)', p >= 0.47),
              ('Fusion (0.47)', rule | (p >= 0.47)), ('Conformer alone (0.30)', p >= 0.30), ('Fusion (0.30)', rule | (p >= 0.30))]:
    r = E.event_metrics(al, e['flag'], seg, onset); lo_, hi_ = E.wilson_ci(r['detected'], r['episodes'])
    print(f'{n:32s} | detected {r["detected"]}/{r["episodes"]} recall {r["event_recall"]:.3f} (Wilson {lo_:.2f}-{hi_:.2f}) | '
          f'alarm precision {r["alarm_precision"]:.3f} | episode F1 {r["event_f1"]:.3f} | false alarms/day {r["false_alarms_per_day"]:.2f}')

# calibration with the Platt mapping fitted on MIMIC-IV validation data
yv = np.load(f'{C.MIMIC_DIR}/windows_12h_2h_gina2026.npz')
mval = (yv['split'] == 'val') & (yv['flag'] == 0)
platt = E.fit_platt(np.load(f'{C.OUT_DIR}/conformer/val_probs.npy'), yv['y'][mval])
for tag, q in [('raw', p), ('Platt', platt(p))]:
    c = E.calibration(y, q)
    print(f'eICU {tag:5s} | Brier {c["brier"]:.3f} | mean {c["mean_pred"]:.3f} vs observed {c["observed"]:.3f}')

# alarm burden of the instantaneous rule on all asthma stays (all ages), raw hourly values
vr = pd.read_csv(f'{C.EICU_RAW}/vitalPeriodic.csv.gz', usecols=['patientunitstayid', 'observationoffset', 'heartrate', 'respiration', 'sao2'])
vr = vr[vr.patientunitstayid.isin(ast.patientunitstayid)]
vr['hour'] = vr.observationoffset // 60
hh = vr.groupby(['patientunitstayid', 'hour'])[DE.COLS].mean().dropna(subset=['respiration', 'sao2']).reset_index()
hh['age'] = hh.patientunitstayid.map(pt.set_index('patientunitstayid').age_n)
print(alarm_burden(hh, 'patientunitstayid', 'sao2', 'respiration', 'age', 'heartrate').round(2))
