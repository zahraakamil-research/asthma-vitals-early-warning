"""Models compared under one protocol. All take x (B, 12, 3) = hourly HR, RR, SpO2 and age (B,).

Conformer: Gulati et al. 2020 (torchaudio implementation), trained from scratch.
ResNet:    Wang et al. 2017 (1D ResNet for time-series classification).
PatchTST:  Nie et al. 2023 (channel-independent patch Transformer).
"""
import torch
import torch.nn as nn
from torchaudio.models import Conformer


class ConformerAge(nn.Module):
    """Linear projection to 64 -> 2-layer Conformer (4 heads, FFN 128, kernel 3, dropout 0.2)
    -> mean over time -> concat standardised age -> linear. 126,914 parameters."""
    def __init__(self, d=64):
        super().__init__()
        self.proj = nn.Linear(3, d)
        self.conf = Conformer(input_dim=d, num_heads=4, ffn_dim=128, num_layers=2,
                              depthwise_conv_kernel_size=3, dropout=0.2)
        self.head = nn.Linear(d + 1, 1)

    def forward(self, x, age):
        h = self.proj(x)
        lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        h, _ = self.conf(h, lengths)
        return self.head(torch.cat([h.mean(1), age.unsqueeze(1)], 1)).squeeze(1)


class _Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        layers = []
        for i, k in enumerate([8, 5, 3]):
            layers += [nn.Conv1d(cin if i == 0 else cout, cout, k, padding='same'), nn.BatchNorm1d(cout)]
            if i < 2:
                layers.append(nn.ReLU())
        self.body = nn.Sequential(*layers)
        self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1), nn.BatchNorm1d(cout))

    def forward(self, x):
        return torch.relu(self.body(x) + self.skip(x))


class ResNet1D(nn.Module):
    """Three residual blocks (64, 128, 128 filters; kernels 8, 5, 3). 521,794 parameters."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(_Block(3, 64), _Block(64, 128), _Block(128, 128))
        self.head = nn.Linear(129, 1)

    def forward(self, x, age):
        h = self.net(x.transpose(1, 2)).mean(2)
        return self.head(torch.cat([h, age.unsqueeze(1)], 1)).squeeze(1)


class PatchTST(nn.Module):
    """Patches of 4 h with stride 2, d = 64, 4 heads, 2 layers, dropout 0.2. 67,778 parameters."""
    def __init__(self, L=12, P=4, S=2, d=64, heads=4, layers=2, drop=0.2):
        super().__init__()
        self.P, self.S, self.N = P, S, (L - P) // S + 1
        self.embed = nn.Linear(P, d)
        self.pos = nn.Parameter(torch.zeros(1, self.N, d))
        enc = nn.TransformerEncoderLayer(d, heads, 128, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(3 * d + 1, 1)

    def forward(self, x, age):
        B, L, Cn = x.shape
        p = x.transpose(1, 2).unfold(2, self.P, self.S).reshape(B * Cn, self.N, self.P)
        h = self.enc(self.embed(p) + self.pos).mean(1).reshape(B, Cn * self.pos.size(2))
        return self.head(torch.cat([h, age.unsqueeze(1)], 1)).squeeze(1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
