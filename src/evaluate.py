"""Threshold selection, hour- and episode-level metrics, bootstrap intervals and calibration.

Episode definitions (Section 3.10 of the paper):
- onset   = first of two consecutive hours meeting the GINA 2026 criterion (first horizon hour of a positive window)
- alarm   = run of consecutive hours with output >= threshold, counted once
- detected episode = at least one alarm hour in the 6 h before onset; lead time = onset - first such alarm hour
- false alarm = alarm run with no onset within 6 h after any of its hours
Sources: Lipton et al. 2014; Tomasev et al. 2019; Efron & Tibshirani 1993; Wilson 1927; Platt 1999;
Van Calster et al. 2019.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score

import config as C

GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def pick_thresholds(p_val, y_val):
    """Warning: max F1 subject to recall >= 0.80. Urgent: lowest threshold with accuracy >= 0.95."""
    acc = np.array([((p_val >= t) == y_val).mean() for t in GRID])
    rec = np.array([((p_val >= t) & (y_val == 1)).sum() / y_val.sum() for t in GRID])
    f1 = np.array([2 * ((p_val >= t) & (y_val == 1)).sum() / ((p_val >= t).sum() + y_val.sum()) for t in GRID])
    return GRID[np.argmax(np.where(rec >= 0.80, f1, -1))], GRID[np.argmax(acc >= 0.95)]


def hour_metrics(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, p >= t).ravel()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(Accuracy=(tp + tn) / len(y), Precision=prec, Recall=rec,
                F1=2 * prec * rec / max(prec + rec, 1e-9), Specificity=tn / (tn + fp))


def timeline(X, group, flag, y):
    """Rebuild hourly segments from overlapping windows (consecutive windows of one stay share 11 hours)."""
    cont = (group[1:] == group[:-1]) & np.all(np.isclose(X[1:, :-1], X[:-1, 1:]), axis=(1, 2))
    seg = np.concatenate([[0], np.cumsum(~cont)])
    onset = (y == 1) & (flag == 0)
    return seg, onset


def _segment_events(a, o, lb=C.LOOKBACK_H):
    det, lead = 0, []
    for oh in o:
        js = [j for j in range(max(0, oh - lb), oh) if a[j]]
        if js:
            det += 1; lead.append(oh - js[0])
    runs = true_runs = 0; k = 0
    while k < len(a):
        if a[k]:
            e = k
            while e + 1 < len(a) and a[e + 1]:
                e += 1
            runs += 1
            true_runs += any(1 <= oh - t <= lb for t in range(k, e + 1) for oh in o)
            k = e + 1
        else:
            k += 1
    return det, lead, runs, true_runs


def event_metrics(alarm_used, flag, seg, onset):
    """alarm_used: boolean alarm for evaluated windows (flag == 0), in window order."""
    al = np.zeros(len(flag), bool); al[flag == 0] = alarm_used
    det = runs = tr = 0; lead = []
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]
        d_, l_, r_, t_ = _segment_events(al[idx], np.where(onset[idx])[0] + 1)
        det += d_; lead += l_; runs += r_; tr += t_
    n = int(onset.sum()); rec, prec = det / n, tr / max(runs, 1)
    q1, md, q3 = np.percentile(lead, [25, 50, 75]) if lead else (np.nan,) * 3
    return dict(detected=det, episodes=n, event_recall=rec, alarm_precision=prec,
                event_f1=2 * rec * prec / max(rec + prec, 1e-9), lead_median=md, lead_q1=q1, lead_q3=q3,
                false_alarms_per_day=(runs - tr) / (len(alarm_used) / 24))


def per_patient_counts(alarm_used, y_used, flag, seg, onset, patient):
    """Per-patient hour counts (TP, FP, FN, TN) and event counts (episodes, detected, alarms, true alarms)."""
    al = np.zeros(len(flag), bool); al[flag == 0] = alarm_used
    yy = np.zeros(len(flag)); yy[flag == 0] = y_used
    rows = {}
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]; o = np.where(onset[idx])[0] + 1
        u = idx[flag[idx] == 0]; a_u, y_u = al[u], yy[u] == 1
        det, _, runs, tr = _segment_events(al[idx], o)
        v = np.array([(a_u & y_u).sum(), (a_u & ~y_u).sum(), (~a_u & y_u).sum(), (~a_u & ~y_u).sum(),
                      len(o), det, runs, tr], float)
        rows[patient[idx[0]]] = rows.get(patient[idx[0]], 0) + v
    return np.array(list(rows.values()))


def metrics_from_counts(c):
    tp, fp, fn, tn, ne, det, runs, tr = c
    pr, rc = tp / (tp + fp), tp / (tp + fn); er, ep = det / ne, tr / runs
    return dict(Accuracy=(tp + tn) / c[:4].sum(), Precision=pr, Recall=rc, F1=2 * pr * rc / (pr + rc),
                Specificity=tn / (tn + fp), event_recall=er, alarm_precision=ep, event_f1=2 * er * ep / (er + ep),
                false_alarms_per_day=(runs - tr) / (c[:4].sum() / 24))


def bootstrap_counts(M, n=C.N_BOOT, seed=C.SEED):
    """Patient-level bootstrap of threshold-dependent metrics from per-patient counts."""
    rng = np.random.default_rng(seed)
    bs = [metrics_from_counts(M[rng.integers(0, len(M), len(M))].sum(0)) for _ in range(n)]
    return {k: np.percentile([b[k] for b in bs], [2.5, 97.5]) for k in bs[0]}


def bootstrap_auc(y, probs, groups, ref='Conformer', n=C.N_BOOT, seed=C.SEED):
    """Patient-level bootstrap: AUC of each model and paired AUC difference (ref minus model)."""
    rng = np.random.default_rng(seed); g = np.unique(groups); gi = {k: np.where(groups == k)[0] for k in g}
    A = {m: [] for m in probs}; D = {m: [] for m in probs if m != ref}
    for _ in range(n):
        ii = np.concatenate([gi[k] for k in rng.choice(g, len(g), replace=True)])
        if y[ii].sum() == 0 or y[ii].sum() == len(ii):
            continue
        a = {m: roc_auc_score(y[ii], p[ii]) for m, p in probs.items()}
        for m in A: A[m].append(a[m])
        for m in D: D[m].append(a[ref] - a[m])
    auc_ci = {m: (roc_auc_score(y, p), *np.percentile(A[m], [2.5, 97.5])) for m, p in probs.items()}
    diff_ci = {m: (np.mean(v), *np.percentile(v, [2.5, 97.5])) for m, v in D.items()}
    return auc_ci, diff_ci


def paired_event_difference(counts, ref='Conformer', n=C.N_BOOT, seed=C.SEED):
    """Paired patient-level bootstrap of episode recall and episode F1 differences (ref minus model).
    counts[model] = per-patient event counts (episodes, detected, alarms, true alarms), same patient order."""
    rng = np.random.default_rng(seed); n_p = len(counts[ref])

    def rf(c, ii):
        s = c[ii].sum(0); r, q = s[1] / s[0], s[3] / s[2]
        return r, 2 * r * q / (r + q)
    D = {m: [] for m in counts if m != ref}
    for _ in range(n):
        ii = rng.integers(0, n_p, n_p); r0, f0 = rf(counts[ref], ii)
        for m in D:
            r1, f1 = rf(counts[m], ii); D[m].append((r0 - r1, f0 - f1))
    return {m: {'event_recall': (np.mean(np.array(v)[:, 0]), *np.percentile(np.array(v)[:, 0], [2.5, 97.5])),
                'event_f1': (np.mean(np.array(v)[:, 1]), *np.percentile(np.array(v)[:, 1], [2.5, 97.5]))}
            for m, v in D.items()}


def wilson_ci(k, n, z=1.96):
    p = k / n; den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def _logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q)).reshape(-1, 1)


def calibration(y, p):
    """Brier score, mean predicted vs observed rate and calibration slope."""
    slope = LogisticRegression(C=1e6).fit(_logit(p), y).coef_[0, 0]
    return dict(brier=brier_score_loss(y, p), mean_pred=p.mean(), observed=y.mean(), slope=slope)


def fit_platt(p_val, y_val):
    """Platt scaling fitted on validation data only; monotone, so ranking and alarm decisions are unchanged."""
    lr = LogisticRegression(C=1e6).fit(_logit(p_val), y_val)
    return lambda q: lr.predict_proba(_logit(np.asarray(q)))[:, 1]
