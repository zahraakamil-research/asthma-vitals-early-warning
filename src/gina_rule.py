"""Instantaneous GINA 2026 rule applied to every reading (all ages).

Source: GINA 2026 report, Box 9-4 and Box 9-6 (adults, adolescents, children 6-11 y)
and Box 12-1 (children <= 5 y).
"""
import numpy as np
import pandas as pd

PLAUSIBLE = {'spo2': (50, 100), 'rr': (1, 80), 'hr': (20, 250)}


def gina_decision(spo2, rr, hr, age):
    """Return Invalid / Urgent / Warning / Normal for one reading. HR is reported only."""
    for k, v in {'spo2': spo2, 'rr': rr, 'hr': hr}.items():
        lo, hi = PLAUSIBLE[k]
        if v is None or not (lo <= v <= hi):
            return 'Invalid'
    rr_limit = 40 if age <= 5 else 30
    if spo2 < 92 or rr > rr_limit:
        return 'Urgent'
    if spo2 < 94:                       # SpO2 92-93 %
        return 'Warning'
    return 'Normal'


def gina_severe(v):
    """Severe criterion used for episode labels: SpO2 < 92 % or RR > 30 /min. v[..., (HR, RR, SpO2)]."""
    return (v[..., 2] < 92) | (v[..., 1] > 30)


def gina_warning_tier(x_last):
    """Warning tier at the current hour (SpO2 92-93 %), used in the fusion and ablation."""
    return (x_last[:, 2] >= 92) & (x_last[:, 2] <= 93)


def alarm_burden(hourly, stay_col, spo2_col, rr_col, age_col, hr_col=None):
    """Share of hours per level and alarm onsets per patient-day for an hourly table."""
    t = hourly.copy()
    inv = ~t[rr_col].between(1, 80) | ~t[spo2_col].between(50, 100)
    if hr_col is not None:
        inv |= t[hr_col].notna() & ~t[hr_col].between(20, 250)
    rr_lim = np.where(t[age_col] <= 5, 40, 30)
    t['level'] = np.select([inv, (t[spo2_col] < 92) | (t[rr_col] > rr_lim), t[spo2_col] < 94],
                           ['Invalid', 'Urgent', 'Warning'], 'Normal')
    sev = t.level.map({'Invalid': -1, 'Normal': 0, 'Warning': 1, 'Urgent': 2})
    prev = sev.groupby(t[stay_col]).shift(1).fillna(0)
    days = len(t) / 24
    share = (t.level.value_counts(normalize=True) * 100).reindex(['Normal', 'Warning', 'Urgent', 'Invalid']).fillna(0)
    return pd.Series({'patient_days': days, **{f'{k} % hours': v for k, v in share.items()},
                      'Warning-or-higher onsets / patient-day': ((sev >= 1) & (prev < 1)).sum() / days,
                      'Urgent onsets / patient-day': ((sev == 2) & (prev < 2)).sum() / days})
