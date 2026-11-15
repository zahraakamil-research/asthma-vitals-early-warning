"""Two-stage training used for all three networks, and the CatBoost baseline.

Stage 1: weighted binary cross-entropy (pos_weight 47.8), Adam 1e-3, batch 512, at most 100 epochs,
         early stopping on validation AUC with patience 10 (King & Zeng 2001; Kingma & Ba 2015; Prechelt 1998).
Stage 2: focal loss (alpha 0.8, gamma 2), Adam 1e-4, 10 epochs, checkpoint saved every epoch (Lin et al. 2017).
Output:  mean probability of fine-tuning epochs 3, 4 and 5 (checkpoint ensemble; Huang et al. 2017).
Checkpoints are written every epoch so a disconnected Colab session resumes where it stopped.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import config as C

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_tensors(s):
    x, a, y = s
    return torch.from_numpy(x), torch.from_numpy(a), torch.from_numpy(y)


@torch.no_grad()
def predict(model, D, bs=4096):
    model.eval()
    out = [torch.sigmoid(model(D[0][i:i + bs].to(DEV), D[1][i:i + bs].to(DEV))).cpu()
           for i in range(0, len(D[2]), bs)]
    return torch.cat(out).numpy()


def focal_loss(logit, y, alpha=C.FOCAL_ALPHA, gamma=C.FOCAL_GAMMA):
    p = torch.sigmoid(logit)
    pt = p * y + (1 - p) * (1 - y)
    at = alpha * y + (1 - alpha) * (1 - y)
    bce = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction='none')
    return (at * (1 - pt) ** gamma * bce).mean()


def _epoch(model, opt, D, loss_fn):
    model.train(); perm = torch.randperm(len(D[2])); total = 0.0
    for i in range(0, len(perm), C.BATCH):
        b = perm[i:i + C.BATCH]
        opt.zero_grad()
        loss = loss_fn(model(D[0][b].to(DEV), D[1][b].to(DEV)), D[2][b].to(DEV))
        loss.backward(); opt.step(); total += loss.item() * len(b)
    return total / len(perm)


def train_two_stage(make_model, Dtr, Dva, outdir, seed):
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    yv = Dva[2].numpy()

    # stage 1: weighted BCE with early stopping on validation AUC
    model = make_model().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)
    wbce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(C.POS_WEIGHT, device=DEV))
    st = dict(ep=0, best=0.0, wait=0, hist=[])
    if os.path.exists(f'{outdir}/base_last.pt'):
        s = torch.load(f'{outdir}/base_last.pt', map_location=DEV, weights_only=False)
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); st = s['st']
    while st['ep'] < C.MAX_EPOCHS and st['wait'] < C.PATIENCE:
        loss = _epoch(model, opt, Dtr, wbce); st['ep'] += 1
        auc = roc_auc_score(yv, predict(model, Dva)); st['hist'].append((st['ep'], loss, auc))
        if auc > st['best']:
            st['best'], st['wait'] = auc, 0
            torch.save(model.state_dict(), f'{outdir}/base_best.pt')
        else:
            st['wait'] += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), st=st), f'{outdir}/base_last.pt')
        print(f"epoch {st['ep']:3d} | loss {loss:.4f} | val AUC {auc:.4f} | best {st['best']:.4f}")

    # stage 2: focal-loss fine-tuning from the best stage-1 model, one checkpoint per epoch
    model = make_model().to(DEV)
    model.load_state_dict(torch.load(f'{outdir}/base_best.pt', map_location=DEV))
    opt = torch.optim.Adam(model.parameters(), lr=C.FT_LR)
    done = [e for e in range(1, C.FT_EPOCHS + 1) if os.path.exists(f'{outdir}/ft_ep{e:02d}.pt')]
    if done:
        s = torch.load(f'{outdir}/ft_ep{max(done):02d}.pt', map_location=DEV, weights_only=False)
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt'])
    for ep in range(max(done, default=0) + 1, C.FT_EPOCHS + 1):
        loss = _epoch(model, opt, Dtr, focal_loss)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict()), f'{outdir}/ft_ep{ep:02d}.pt')
        print(f'fine-tune epoch {ep:2d} | focal loss {loss:.5f} | val AUC {roc_auc_score(yv, predict(model, Dva)):.4f}')
    return st


def load_ensemble(make_model, outdir):
    members = []
    for e in C.ENSEMBLE_EPOCHS:
        m = make_model().to(DEV)
        m.load_state_dict(torch.load(f'{outdir}/ft_ep{e:02d}.pt', map_location=DEV, weights_only=False)['model'])
        members.append(m.eval())
    return members


def ensemble_predict(members, D):
    return np.mean([predict(m, D) for m in members], 0)


def catboost_features(x, age):
    """Raw 12 h values plus mean, min, max, last value and slope per channel, plus age."""
    t = np.arange(x.shape[1]) - (x.shape[1] - 1) / 2
    slope = (x * t[None, :, None]).sum(1) / (t ** 2).sum()
    return np.hstack([x.reshape(len(x), -1), x.mean(1), x.min(1), x.max(1), x[:, -1], slope, age[:, None]])


def train_catboost(Ftr, ytr, Fva, yva, path):
    from catboost import CatBoostClassifier
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.05, depth=6, loss_function='Logloss',
                            scale_pos_weight=C.POS_WEIGHT, eval_metric='AUC', od_type='Iter', od_wait=100,
                            random_seed=C.SEED, verbose=250, thread_count=-1)
    cb.fit(Ftr, ytr, eval_set=(Fva, yva), use_best_model=True)
    cb.save_model(path)
    return cb
