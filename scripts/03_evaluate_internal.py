"""Step 3 - internal test on MIMIC-IV: thresholds from validation, matched comparison (Tables 3-4),
Conformer with confidence intervals (Table 5), ablation (Table 6), threshold sensitivity (Table 7),
detection by episode type and duration, calibration and Platt scaling."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

import config as C
from src.data_mimic import load_sets
from src.gina_rule import gina_warning_tier, gina_severe
from src.models import ConformerAge
from src import evaluate as E
from src import train as T

pd.set_option('display.width', 250)
stats = {k: np.asarray(v, dtype=np.float32) for k, v in C.NORM_STATS.items()}
d, masks, sets = load_sets(f'{C.MIMIC_DIR}/windows_12h_2h_gina2026.npz', stats)
yv, yt = sets['val'][2], sets['test'][2]
te = np.where(d['split'] == 'test')[0]
flag_te, X_te, subj_te = d['flag'][te], d['X'][te], d['subject'][te]
seg, onset = E.timeline(X_te, subj_te, flag_te, d['y'][te])
pat_used = d['subject'][masks['test']]

names = ['ResNet', 'CatBoost', 'PatchTST', 'Conformer']
P = {n: (np.load(f'{C.OUT_DIR}/{n.lower()}/val_probs.npy'), np.load(f'{C.OUT_DIR}/{n.lower()}/test_probs.npy')) for n in names}
thr = {n: E.pick_thresholds(pv, yv) for n, (pv, _) in P.items()}

# Table 4: hour- and episode-level metrics at validation-selected thresholds
rows = []
for n, (_, pt) in P.items():
    for tier, t in zip(['Warning', 'Urgent'], thr[n]):
        h, e = E.hour_metrics(yt, pt, t), E.event_metrics(pt >= t, flag_te, seg, onset)
        rows.append({'Model': n, 'Tier': tier, 'Threshold': t, **h, **{k: e[k] for k in
                     ['event_recall', 'alarm_precision', 'event_f1', 'lead_median', 'false_alarms_per_day']}})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# Table 3: AUC with patient-level bootstrap CI, paired difference vs Conformer, average precision
auc_ci, diff_ci = E.bootstrap_auc(yt, {n: p[1] for n, p in P.items()}, pat_used)
for n in names:
    print(f'{n:9s} AUC {auc_ci[n][0]:.3f} ({auc_ci[n][1]:.3f}-{auc_ci[n][2]:.3f}) | AP {average_precision_score(yt, P[n][1]):.3f}')
for n, (m, lo, hi) in diff_ci.items():
    print(f'Conformer - {n}: {m:+.3f} ({lo:+.3f} to {hi:+.3f})')

# paired episode-level differences at the urgent tier
cnt = {n: E.per_patient_counts(P[n][1] >= thr[n][1], yt, flag_te, seg, onset, subj_te)[:, 4:] for n in names}
for n, r in E.paired_event_difference(cnt).items():
    print(f'urgent tier, Conformer - {n}: episode recall {r["event_recall"][0]:+.3f} ({r["event_recall"][1]:+.3f} to '
          f'{r["event_recall"][2]:+.3f}) | episode F1 {r["event_f1"][0]:+.3f} ({r["event_f1"][1]:+.3f} to {r["event_f1"][2]:+.3f})')

# Table 5: Conformer with patient-level bootstrap CIs and lead time
p = P['Conformer'][1]
for tier, t in C.THRESHOLDS.items():
    M = E.per_patient_counts(p >= t, yt, flag_te, seg, onset, subj_te)
    est, ci = E.metrics_from_counts(M.sum(0)), E.bootstrap_counts(M)
    e = E.event_metrics(p >= t, flag_te, seg, onset)
    print(f'--- Conformer {tier} {t} | lead time median {e["lead_median"]:.0f} h (IQR {e["lead_q1"]:.0f}-{e["lead_q3"]:.0f}) '
          f'| detected {e["detected"]}/{e["episodes"]}')
    for k, v in est.items():
        print(f'  {k:22s} {v:.3f} ({ci[k][0]:.3f}-{ci[k][1]:.3f})')

# Table 6 (A): fusion ablation - GINA warning tier at the current hour, Conformer, and the more severe of the two
rule = gina_warning_tier(X_te[flag_te == 0][:, -1])
for n, a in [('GINA rule (warning tier) alone', rule), ('Conformer alone (0.47)', p >= 0.47),
             ('Fusion (0.47)', rule | (p >= 0.47)), ('Fusion (0.30)', rule | (p >= 0.30))]:
    e = E.event_metrics(a, flag_te, seg, onset)
    print(f'{n:32s} | episode recall {e["event_recall"]:.3f} | alarm precision {e["alarm_precision"]:.3f} | '
          f'episode F1 {e["event_f1"]:.3f} | false alarms/day {e["false_alarms_per_day"]:.2f}')

# Table 6 (B): training-stage ablation (thresholds re-selected on validation by the same rules)
out = f'{C.OUT_DIR}/conformer'; m = ConformerAge().to(T.DEV)
Dva, Dte = T.to_tensors(sets['val']), T.to_tensors(sets['test'])
m.load_state_dict(torch.load(f'{out}/base_best.pt', map_location=T.DEV))
stages = {'Base model, WBCE': (T.predict(m, Dva), T.predict(m, Dte))}
m.load_state_dict(torch.load(f'{out}/ft_ep03.pt', map_location=T.DEV, weights_only=False)['model'])
stages['+ focal loss, single checkpoint'] = (T.predict(m, Dva), T.predict(m, Dte))
stages['+ focal loss, three-checkpoint ensemble'] = P['Conformer']
for n, (pv, pt) in stages.items():
    u = E.pick_thresholds(pv, yv)[1]; e = E.event_metrics(pt >= u, flag_te, seg, onset)
    print(f'{n:40s} | test AUC {roc_auc_score(yt, pt):.3f} | urgent {u} | episode recall {e["event_recall"]:.3f} | '
          f'alarm precision {e["alarm_precision"]:.3f} | episode F1 {e["event_f1"]:.3f} | false alarms/day {e["false_alarms_per_day"]:.2f}')

# Table 7: threshold sensitivity
rows = []
for t in [0.40, 0.45, 0.47, 0.50, 0.60, 0.70, 0.85]:
    h, e = E.hour_metrics(yt, p, t), E.event_metrics(p >= t, flag_te, seg, onset)
    rows.append({'Threshold': t, 'Accuracy': h['Accuracy'], 'Specificity': h['Specificity'], 'Recall (hour)': h['Recall'],
                 'F1 (hour)': h['F1'], 'Episode recall': e['event_recall'], 'Alarm precision': e['alarm_precision'],
                 'Episode F1': e['event_f1'], 'False alarms/day': e['false_alarms_per_day'], 'Detected': e['detected']})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# detection at the urgent threshold by triggering criterion and episode duration
crit = gina_severe(X_te[:, -1][:, None, :])[:, 0]; prob = np.zeros(len(te)); prob[flag_te == 0] = p
ev = []
for s in np.unique(seg):
    idx = np.where(seg == s)[0]
    for k in np.where(onset[idx])[0]:
        oh = k + 1; dur = 0
        while oh + dur < len(idx) and crit[idx[oh + dur]]:
            dur += 1
        h2 = idx[oh:oh + 2]; S2, R2 = X_te[h2, -1, 2] < 92, X_te[h2, -1, 1] > 30
        typ = 'unknown' if len(h2) < 2 else 'SpO2' if S2.all() and not R2.any() else 'RR' if R2.all() and not S2.any() else 'mixed'
        ev.append(dict(type=typ, duration='2-3 h' if dur < 4 else '>=4 h', urgent=(prob[idx[max(0, oh - 6):oh]] >= 0.47).any()))
ev = pd.DataFrame(ev)
for c in ['type', 'duration']:
    print(ev.groupby(c).urgent.agg(['size', 'mean']).round(3))

# calibration before and after Platt scaling (fitted on validation only)
platt = E.fit_platt(P['Conformer'][0], yv)
print('thresholds 0.30 / 0.47 map to probabilities', np.round(platt([0.30, 0.47]), 3))
for tag, q in [('raw', p), ('Platt', platt(p))]:
    c = E.calibration(yt, q)
    print(f'{tag:5s} | AUC {roc_auc_score(yt, q):.3f} | Brier {c["brier"]:.3f} | mean {c["mean_pred"]:.3f} vs observed '
          f'{c["observed"]:.3f} | slope {c["slope"]:.2f}')
