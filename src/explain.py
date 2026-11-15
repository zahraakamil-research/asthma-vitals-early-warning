"""Integrated Gradients for the Conformer ensemble (Sundararajan et al. 2017).
Baseline = training mean (zero after standardisation), 50 steps, attribution on the output logit."""
import numpy as np
import torch

from src.train import DEV


def ig_members(members, x, a, steps=50):
    """Absolute attribution per input channel (summed over the 12 hours) and for age, averaged over members.
    x: (N, 12, 3) standardised, a: (N,) standardised. Returns (N, 4) = HR, RR, SpO2, age."""
    tot_x, tot_a = torch.zeros_like(x), torch.zeros_like(a)
    for m in members:
        m.eval()
        for k in range(1, steps + 1):
            xs = (x * k / steps).requires_grad_(True); as_ = (a * k / steps).requires_grad_(True)
            gx, ga = torch.autograd.grad(m(xs, as_).sum(), [xs, as_])
            tot_x += gx / steps; tot_a += ga / steps
    n = len(members)
    att_x = (tot_x * x / n).abs().sum(1)            # (N, 3)
    att_a = (tot_a * a / n).abs().unsqueeze(1)      # (N, 1)
    return torch.cat([att_x, att_a], 1).detach().cpu().numpy()


def attribution(members, X, A, idx, batch=1024):
    torch.backends.cudnn.enabled = False            # as in the original run
    out = []
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        xb = torch.from_numpy(X[b]).to(DEV); ab = torch.from_numpy(A[b]).to(DEV)
        out.append(ig_members(members, xb, ab))
    torch.backends.cudnn.enabled = True
    return np.vstack(out)


def overall_share(att):
    """Share of total attribution per input over a set of windows (paper: 1,378 test episodes)."""
    s = att.mean(0)
    return s / s.sum() * 100


def per_window_share(att):
    """Mean per-window share of each input (paper: detected, missed, false-alarm groups)."""
    return (att / att.sum(1, keepdims=True)).mean(0) * 100
