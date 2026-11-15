"""eICU-CRD Demo v2.0.1 external cohort: asthma by diagnosis or past history, adults, hourly vitals, 12 h windows.
The MIMIC-IV build setting is applied unchanged (Pollard et al. 2018; Johnson et al. 2021, PhysioNet).
vitalPeriodic stores five-minute medians; sao2 is the pulse-oximetry SpO2."""
import os
import subprocess
import numpy as np
import pandas as pd

import config as C
from src.data_mimic import stay_windows
from src.gina_rule import gina_severe

BASE_URL = 'https://physionet.org/files/eicu-crd-demo/2.0.1'
COLS = ['heartrate', 'respiration', 'sao2']


def download(dest=C.EICU_RAW):
    os.makedirs(dest, exist_ok=True)
    for f in ['patient', 'diagnosis', 'pastHistory', 'vitalPeriodic']:
        subprocess.run(['wget', '-c', '-q', '-O', f'{dest}/{f}.csv.gz', f'{BASE_URL}/{f}.csv.gz'], check=True)


def cohort(src=C.EICU_RAW):
    pt = pd.read_csv(f'{src}/patient.csv.gz')
    dx = pd.read_csv(f'{src}/diagnosis.csv.gz')
    ph = pd.read_csv(f'{src}/pastHistory.csv.gz')
    m_dx = dx.diagnosisstring.str.contains('asthma', case=False, na=False) | \
        dx.icd9code.astype(str).str.contains(r'\b493|J45', na=False)
    m_ph = ph.pasthistorypath.str.contains('asthma', case=False, na=False)
    asthma = set(dx.loc[m_dx, 'patientunitstayid']) | set(ph.loc[m_ph, 'patientunitstayid'])
    pt['age_n'] = pd.to_numeric(pt.age.replace('> 89', '90'), errors='coerce')
    ast = pt[pt.patientunitstayid.isin(asthma)]
    adult = ast[ast.age_n >= C.MIN_AGE]
    return pt, ast, adult


def hourly(adult, src=C.EICU_RAW, stays=None):
    vp = pd.read_csv(f'{src}/vitalPeriodic.csv.gz',
                     usecols=['patientunitstayid', 'observationoffset', 'heartrate', 'respiration', 'sao2'])
    keep = adult.patientunitstayid if stays is None else stays
    vp = vp[vp.patientunitstayid.isin(keep)].copy()
    vp['hour'] = vp.observationoffset // 60
    for c, (lo, hi) in zip(COLS, [C.PLAUSIBLE['heart_rate'], C.PLAUSIBLE['resp_rate'], C.PLAUSIBLE['spo2']]):
        vp.loc[~vp[c].between(lo, hi), c] = np.nan
    return vp.groupby(['patientunitstayid', 'hour'])[COLS].mean().reset_index()


def build_windows(vh, adult):
    age = adult.set_index('patientunitstayid').age_n
    pid = adult.set_index('patientunitstayid').uniquepid
    out = {k: [] for k in ['X', 'y', 'flag', 'stay', 'end', 'age', 'patient']}
    for sid, g in vh.rename(columns={'hour': 'hr_idx'}).groupby('patientunitstayid', sort=False):
        r = stay_windows(g, COLS)
        if r is None:
            continue
        inp, hor, end = r; n = len(inp)
        out['X'].append(inp); out['y'].append(gina_severe(hor).all(1)); out['flag'].append(gina_severe(inp)[:, -1])
        out['stay'].append(np.full(n, sid)); out['end'].append(end)
        out['age'].append(np.full(n, age[sid], np.float32)); out['patient'].append(np.full(n, pid[sid]))
    d = {k: np.concatenate(v) for k, v in out.items()}
    d['y'], d['flag'] = d['y'].astype(np.int64), d['flag'].astype(np.int64)
    return d
