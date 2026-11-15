"""Step 6 - Integrated Gradients (Section 4.7) and Figs. 3, 4 and 5 at their printed size."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

import config as C
from src.data_mimic import load_sets
from src.models import ConformerAge
from src import evaluate as E
from src import explain as X_
from src import train as T

stats = {k: np.asarray(v, dtype=np.float32) for k, v in C.NORM_STATS.items()}
d, masks, sets = load_sets(f'{C.MIMIC_DIR}/windows_12h_2h_gina2026.npz', stats)
xt, at, yt = sets['test']; yv = sets['val'][2]
p = np.load(f'{C.OUT_DIR}/conformer/test_probs.npy')
members = T.load_ensemble(ConformerAge, f'{C.OUT_DIR}/conformer')
names = ['HR', 'RR', 'SpO2', 'Age']

# overall attribution over all test episodes
pos = np.where(yt == 1)[0]
print('all episodes:', dict(zip(names, X_.overall_share(X_.attribution(members, xt, at, pos)).round(1))))

# prespecified groups at the urgent threshold: detected, missed, random false alarms of the same size (seed 42)
neg = np.where(yt == 0)[0]
groups = {'Detected': pos[p[pos] >= 0.47], 'Missed': pos[p[pos] < 0.47],
          'False alarms': np.random.default_rng(C.SEED).choice(neg[p[neg] >= 0.47], len(pos), replace=False)}
shares, att = {}, {}
for g, idx in groups.items():
    att[g] = X_.attribution(members, xt, at, idx)
    shares[g] = X_.per_window_share(att[g])
    print(f'{g:13s} n={len(idx):5d} | ' + ' | '.join(f'{n} {v:.1f}%' for n, v in zip(names, shares[g])))

# agreement between the most attributed vital sign and the criterion that defined the episode
te = np.where(d['split'] == 'test')[0]; fl = d['flag'][te]
used = np.cumsum(fl == 0) - 1                                  # test-window index -> evaluated-window index
seg, onset = E.timeline(d['X'][te], d['subject'][te], fl, d['y'][te])
typ = {}
for s in np.unique(seg):
    idx = np.where(seg == s)[0]
    for k in np.where(onset[idx])[0]:
        h2 = idx[k + 1:k + 3]
        if len(h2) < 2:
            continue
        S2, R2 = d['X'][te][h2, -1, 2] < 92, d['X'][te][h2, -1, 1] > 30
        typ[used[idx[k]]] = 'SpO2' if S2.all() and not R2.any() else 'RR' if R2.all() and not S2.any() else 'mixed'
for g in ['Detected', 'Missed']:
    top = np.array(names)[att[g][:, :3].argmax(1)]; t_ = np.array([typ.get(i, 'unknown') for i in groups[g]])
    for c in ['SpO2', 'RR']:
        k = t_ == c
        print(f'{g:9s} {c}-defined episodes n={k.sum()} | top channel = {c}: {(top[k] == c).mean() * 100:.1f}%')

plt.rcParams.update({'font.family': 'serif', 'font.size': 8})

# Fig. 3: ROC and precision-recall curves of the four models with validation-selected operating points
cols = {'ResNet': '#1f77b4', 'CatBoost': '#2ca02c', 'PatchTST': '#9467bd', 'Conformer': '#d62728'}
fig, ax = plt.subplots(1, 2, figsize=(5.9, 2.5))
for n, c in cols.items():
    pv, pt = np.load(f'{C.OUT_DIR}/{n.lower()}/val_probs.npy'), np.load(f'{C.OUT_DIR}/{n.lower()}/test_probs.npy')
    f, tpr, _ = roc_curve(yt, pt); pr, rc, _ = precision_recall_curve(yt, pt); lw = 1.4 if n == 'Conformer' else 0.9
    ax[0].plot(f, tpr, color=c, lw=lw, label=f'{n} (AUC {roc_auc_score(yt, pt):.3f})')
    ax[1].plot(rc, pr, color=c, lw=lw, label=f'{n} (AP {average_precision_score(yt, pt):.3f})')
    for t, mk in zip(E.pick_thresholds(pv, yv), ['o', 's']):
        a = pt >= t; tp = (a & (yt == 1)).sum()
        ax[0].plot((a & (yt == 0)).sum() / (yt == 0).sum(), tp / yt.sum(), mk, color=c, ms=3.5, mec='k', mew=0.4)
        ax[1].plot(tp / yt.sum(), tp / max(a.sum(), 1), mk, color=c, ms=3.5, mec='k', mew=0.4)
ax[0].plot([0, 1], [0, 1], ':', color='grey', lw=0.8); ax[1].axhline(yt.mean(), ls=':', color='grey', lw=0.8)
ax[0].set(xlabel='False positive rate', ylabel='True positive rate', title='(a) ROC')
ax[1].set(xlabel='Recall', ylabel='Precision', title='(b) Precision-recall', ylim=(0, 1))
for a_ in ax:
    a_.legend(fontsize=6.5, frameon=False, loc='lower right' if a_ is ax[0] else 'upper right')
    a_.spines[['top', 'right']].set_visible(False)
fig.tight_layout(pad=0.4); fig.savefig(f'{C.OUT_DIR}/fig3_roc_pr.png', dpi=600, bbox_inches='tight', pad_inches=0.03)

# Fig. 4: representative cases (clean early detection, missed episode, false alarm)
Xt, subj, flt, yy = d['X'][te], d['subject'][te], fl, d['y'][te]
prob = np.zeros(len(te)); prob[fl == 0] = p
crit = (Xt[:, -1, 2] < 92) | (Xt[:, -1, 1] > 30)
hit = miss = fa = None; best_dur, min_p = -1, 9
for s in np.unique(seg):
    idx = np.where(seg == s)[0]; a = prob[idx]
    for k in np.where(onset[idx])[0]:
        oh = k + 1; dur = 0
        while oh + dur < len(idx) and crit[idx[oh + dur]]:
            dur += 1
        pw = a[max(0, oh - 6):oh]
        if oh >= 12 and not crit[idx[oh - 12:oh]].any() and (a[oh - 6:oh - 2] >= 0.47).any() and (pw >= 0.47).any() \
                and dur >= 3 and dur > best_dur:
            best_dur, hit = dur, (s, oh)
        if len(pw) and pw.max() < 0.30 and pw.max() < min_p:
            min_p, miss = pw.max(), (s, oh)
    if fa is None and onset[idx].sum() == 0 and (a >= 0.47).sum() >= 3:
        fa = s


def series(s):
    idx = np.where(seg == s)[0]; v = np.vstack([Xt[idx[0]], Xt[idx[1:], -1]])
    pr = np.full(len(v), np.nan); pr[11:11 + len(idx)] = np.where(flt[idx] == 0, prob[idx], np.nan)
    return v, pr, [11 + k + 1 for k in np.where(onset[idx])[0]]


cases = [('(A) Early detection', *hit), ('(B) Missed episode', *miss), ('(C) False alarm', fa, None)]
fig, ax = plt.subplots(2, 3, figsize=(6.1, 3.0), sharex='col',
                       gridspec_kw={'height_ratios': [1.3, 1], 'wspace': 0.95, 'hspace': 0.14})
for c, (title, s, oh) in enumerate(cases):
    v, pr, on = series(s); c0 = (11 + oh) if oh is not None else int(np.nanargmax(pr))
    lo, hi = max(0, c0 - 14), min(len(v), c0 + 6); t = np.arange(lo, hi) - c0
    a1 = ax[0, c]; a2 = a1.twinx()
    a1.plot(t, v[lo:hi, 2], color='tab:blue', lw=1.1); a1.axhline(92, color='tab:blue', ls=':', lw=0.8)
    a2.plot(t, v[lo:hi, 1], color='tab:red', lw=1.1); a2.axhline(30, color='tab:red', ls=':', lw=0.8)
    a1.set_ylabel('SpO$_2$ (%)', color='tab:blue', labelpad=2); a2.set_ylabel('RR (breaths/min)', color='tab:red', labelpad=3)
    a1.set_title(title, fontsize=8.5, pad=3)
    ax[1, c].plot(t, pr[lo:hi], 'o-', color='black', ms=1.8, lw=0.8)
    ax[1, c].axhline(0.30, color='orange', ls='--', lw=0.8, label='Warning (0.30)')
    ax[1, c].axhline(0.47, color='crimson', ls='--', lw=0.8, label='Urgent (0.47)')
    ax[1, c].set_ylim(0, 1); ax[1, c].set_ylabel('Model output', labelpad=2)
    ax[1, c].set_xlabel('Hours from onset' if oh is not None else 'Hours from peak output')
    for o in on:
        if lo <= o < hi:
            for a_ in (a1, ax[1, c]):
                a_.axvspan(o - c0, o - c0 + 2, color='grey', alpha=0.25)
ax[1, 0].legend(fontsize=6.5, loc='upper left', frameon=False, handlelength=1.5)
fig.savefig(f'{C.OUT_DIR}/fig4_cases.png', dpi=600, bbox_inches='tight', pad_inches=0.03)

# Fig. 5: mean per-window attribution share by group
fig, ax = plt.subplots(figsize=(3.2, 2.25)); yb = np.arange(4); h = 0.26
labels = {'Detected': f'Detected episodes (n = {len(groups["Detected"])})', 'Missed': f'Missed episodes (n = {len(groups["Missed"])})',
          'False alarms': f'False alarms (n = {len(groups["False alarms"]):,})'}
order = [1, 2, 0, 3]                                           # RR, SpO2, HR, Age
for i, (g, col) in enumerate(zip(groups, ['#2166ac', '#d6604d', '#f4a582'])):
    vals = shares[g][order]; ax.barh(yb + (i - 1) * h, vals, h, color=col, label=labels[g])
    for yy_, v in zip(yb + (i - 1) * h, vals):
        ax.text(v + 0.6, yy_, f'{v:.1f}', va='center', fontsize=7)
ax.set_yticks(yb); ax.set_yticklabels(['Respiratory rate', 'SpO$_2$', 'Heart rate', 'Age']); ax.invert_yaxis()
ax.set_xlim(0, 60); ax.set_xlabel('Mean share of absolute Integrated\nGradients attribution per window (%)')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(fontsize=6.5, frameon=False, loc='lower right', bbox_to_anchor=(1.02, -0.01), handlelength=1.0)
fig.tight_layout(pad=0.3); fig.savefig(f'{C.OUT_DIR}/fig5_ig_groups.png', dpi=600, bbox_inches='tight', pad_inches=0.03)
print('figures saved to', C.OUT_DIR)
