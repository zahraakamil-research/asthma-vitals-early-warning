"""MIMIC-IV v3.1 cohort extraction (BigQuery), hourly vitals, 12 h windows, GINA 2026 labels, patient split.

Sources: Johnson et al. 2023 (MIMIC-IV); Harutyunyan et al. 2019 (hourly windows, patient split);
Wang et al. 2020 (MIMIC-Extract plausible ranges); GINA 2026 (severe criterion).
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
from sklearn.model_selection import GroupShuffleSplit

import config as C
from src.gina_rule import gina_severe

ASTHMA_SQL = f"""
SELECT DISTINCT subject_id, hadm_id FROM `{C.MIMIC_HOSP}.diagnoses_icd`
WHERE (icd_version = 9 AND icd_code LIKE '493%') OR (icd_version = 10 AND icd_code LIKE 'J45%')
"""

HOURLY_SQL = f"""
WITH asthma_stays AS (
  SELECT i.subject_id, i.stay_id, i.intime, p.anchor_age
  FROM `{C.MIMIC_ICU}.icustays` i
  JOIN ({ASTHMA_SQL}) a USING (subject_id, hadm_id)
  JOIN `{C.MIMIC_HOSP}.patients` p USING (subject_id)
)
SELECT s.subject_id, s.stay_id, s.anchor_age,
       TIMESTAMP_DIFF(c.charttime, s.intime, HOUR) AS hr_idx,
       AVG(IF(c.itemid = 220045 AND c.valuenum BETWEEN 20 AND 250, c.valuenum, NULL)) AS heart_rate,
       AVG(IF(c.itemid = 220210 AND c.valuenum BETWEEN 1 AND 80, c.valuenum, NULL))   AS resp_rate,
       AVG(IF(c.itemid = 220277 AND c.valuenum BETWEEN 50 AND 100, c.valuenum, NULL)) AS spo2
FROM `{C.MIMIC_ICU}.chartevents` c
JOIN asthma_stays s USING (stay_id)
WHERE c.itemid IN (220045, 220210, 220277) AND c.valuenum IS NOT NULL AND c.charttime >= s.intime
GROUP BY s.subject_id, s.stay_id, s.anchor_age, hr_idx
ORDER BY s.stay_id, hr_idx
"""


def query_hourly(client):
    """Hourly mean HR, RR, SpO2 per asthma ICU stay (implausible values removed before averaging)."""
    return client.query(HOURLY_SQL).to_dataframe()


def stay_windows(g, cols, w=C.INPUT_H, h=C.HORIZON_H):
    """Sliding windows for one stay: hourly grid, forward-fill <= 2 h, complete input and horizon only.
    Returns inputs (n, w, 3), horizon (n, h, 3) and the hour index of the last input hour."""
    g = g.set_index('hr_idx')[cols]
    h0 = g.index.min()
    v = g.reindex(range(h0, g.index.max() + 1)).ffill(limit=C.FFILL_LIMIT).values
    if len(v) < w + h:
        return None
    win = swv(v, (w + h, 3))[:, 0]
    inp, hor = win[:, :w], win[:, w:]
    ok = ~np.isnan(inp).any((1, 2)) & ~np.isnan(hor).any((1, 2))
    if not ok.any():
        return None
    end = (h0 + np.arange(len(win)) + w - 1)[ok]
    return inp[ok], hor[ok], end                     # float64: labels are computed before casting


def build_windows(vh, cols=C.VITALS, age_col='anchor_age', stay_col='stay_id', subj_col='subject_id'):
    """Windows with GINA 2026 labels.
    y         = 1 if the criterion holds in both horizon hours (sustained episode)
    transient = 1 if it holds in exactly one horizon hour (removed from training only)
    flag      = 1 if it already holds in the last input hour (excluded: handled by the instantaneous rule)"""
    out = {k: [] for k in ['X', 'y', 'transient', 'flag', 'subject', 'stay', 'end', 'age']}
    for (sid, stay), g in vh.groupby([subj_col, stay_col], sort=True):
        r = stay_windows(g, cols)
        if r is None:
            continue
        inp, hor, end = r
        a, ai = gina_severe(hor), gina_severe(inp)
        n = len(inp)
        out['X'].append(inp.astype(np.float32)); out['y'].append(a.all(1)); out['transient'].append(a.any(1) & ~a.all(1))
        out['flag'].append(ai[:, -1]); out['subject'].append(np.full(n, sid)); out['stay'].append(np.full(n, stay))
        out['end'].append(end); out['age'].append(np.full(n, g[age_col].iloc[0], np.float32))
    d = {k: np.concatenate(v) for k, v in out.items()}
    for k in ['y', 'transient', 'flag']:
        d[k] = d[k].astype(np.int64)
    return d


def split_patients(vh, cols=C.VITALS, seed=C.SEED):
    """70 / 15 / 15 split by patient (subject_id), so all windows of a patient stay in one set.
    As in the study, the split was drawn once, at an earlier development stage, over the patients with at
    least one 6 h input window (complete input, horizon not entirely missing; 5,537 patients) and then kept
    fixed; the 12 h dataset inherits it. Returns {subject_id: 'train' | 'val' | 'test'}."""
    keep = []
    for (sid, _), g in vh.groupby(['subject_id', 'stay_id']):
        g = g.set_index('hr_idx')[cols]
        v = g.reindex(range(g.index.min(), g.index.max() + 1)).ffill(limit=C.FFILL_LIMIT).to_numpy(np.float64)
        if len(v) < 6 + C.HORIZON_H:
            continue
        win = swv(v, (6 + C.HORIZON_H, 3))[:, 0]
        if (~np.isnan(win[:, :6]).any((1, 2)) & ~np.isnan(win[:, 6:]).all((1, 2))).any():
            keep.append(sid)
    groups = np.array(sorted(set(keep))); idx = np.arange(len(groups))
    tr, tmp = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed).split(idx, groups=groups))
    va, te = next(GroupShuffleSplit(1, test_size=0.50, random_state=seed).split(tmp, groups=groups[tmp]))
    split = {g: 'train' for g in groups[tr]}
    split.update({g: 'val' for g in groups[tmp[va]]}); split.update({g: 'test' for g in groups[tmp[te]]})
    return split


def assign_split(subject, split_map):
    return np.array([split_map[s] for s in subject], dtype='<U5')


def norm_stats(d, mask):
    """Mean and standard deviation per channel and for age, from training windows only."""
    x = d['X'][mask]
    return dict(mu=x.mean((0, 1)), sd=x.std((0, 1)), a_mu=float(d['age'][mask].mean()), a_sd=float(d['age'][mask].std()))


def check_split(d):
    """Show (not just assert) that no patient appears in two sets."""
    S = {s: set(np.unique(d['subject'][d['split'] == s])) for s in ['train', 'val', 'test']}
    print('patients:', {k: len(v) for k, v in S.items()}, '| total:', len(set.union(*S.values())))
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        print(f'overlap {a}-{b}:', len(S[a] & S[b]))


def check_labels(d):
    """Independent label check: the horizon of window i equals the last two input hours of window i+2."""
    X, subj, y = d['X'], d['subject'], d['y']
    crit = gina_severe(X)
    cont = (subj[1:] == subj[:-1]) & np.all(np.isclose(X[1:, :-1], X[:-1, 1:]), axis=(1, 2))
    ok = np.zeros(len(X), bool); ok[:-2] = cont[:-1] & cont[1:]
    i = np.where(ok)[0]
    y_hat = (crit[i + 2, -2] & crit[i + 2, -1]).astype(int)
    pos, neg = y[i] == 1, y[i] == 0
    print(f'checked {len(i):,} windows | positives {pos.sum():,} disagree {int((y_hat[pos] != 1).sum())} | '
          f'negatives {neg.sum():,} disagree {int((y_hat[neg] != 0).sum())}')


def load_sets(path, stats):
    """Standardised tensors-ready arrays for train (no flag, no transient), val and test (no flag)."""
    d = dict(np.load(path, allow_pickle=True))
    m = {'train': (d['split'] == 'train') & (d['flag'] == 0) & (d['transient'] == 0),
         'val': (d['split'] == 'val') & (d['flag'] == 0),
         'test': (d['split'] == 'test') & (d['flag'] == 0)}
    sets = {}
    for k, mk in m.items():
        x = ((d['X'][mk] - stats['mu']) / stats['sd']).astype(np.float32)
        a = ((d['age'][mk] - stats['a_mu']) / stats['a_sd']).astype(np.float32)
        sets[k] = (x, a, d['y'][mk].astype(np.float32))
    return d, m, sets
