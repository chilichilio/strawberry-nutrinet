#!/usr/bin/env python3
"""
test_seq_fix.py: CPU-only check of the TCN sequence-branch problems and the fix.

The TCN classes below are copied verbatim from base pipeline dl_core.py, so this
test needs only torch, not the data or the cluster.

Checks
  A. Batch invariance. In eval mode, a sample's embedding should not depend on which
     other samples share its batch.
       base pipeline: right padding + reading the LAST time step -> it depends.
       v4    : left padding with NaN + padding masked at every layer -> identical.
  B. Receptive field. Does changing the first days of a full-season sequence change
     the embedding?
       base pipeline: 4 levels, kernel 3 -> 61-day receptive field -> no.
       v4    : 6 levels           -> 253-day receptive field -> yes.

Run:  python test_seq_fix.py      (exit code 0 = all checks pass)
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class _TCNBlock(nn.Module):          # verbatim from base pipeline dl_core.py
    def __init__(self, in_ch, out_ch, k, dilation, dropout=0.1):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dilation))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=dilation))
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.chomp = pad
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        y = self.relu(self.conv1(x)[..., :-self.chomp] if self.chomp else self.conv1(x))
        y = self.drop(y)
        y = self.relu(self.conv2(y)[..., :-self.chomp] if self.chomp else self.conv2(y))
        y = self.drop(y)
        res = self.down(x) if self.down is not None else x
        return self.relu(y + res)


class TCNEncoder(nn.Module):         # verbatim from base pipeline dl_core.py
    def __init__(self, in_ch, hidden_ch=64, levels=4, kernel_size=3, out_dim=32, dropout=0.1):
        super().__init__()
        blocks, ch = [], in_ch
        for i in range(levels):
            blocks.append(_TCNBlock(ch, hidden_ch, kernel_size, 2 ** i, dropout))
            ch = hidden_ch
        self.net = nn.Sequential(*blocks)
        self.project = nn.Linear(hidden_ch, out_dim)

    def forward(self, x):
        h = self.net(x)[..., -1]
        return self.project(h)


def masked_forward(self, x):
    """v4 TCNEncoder.forward. Padded steps are NaN; they are zeroed at EVERY layer,
    which is exactly what the causal conv's own zero padding does for an unpadded
    sequence, so a padded sample gives the same output as when it is encoded alone."""
    valid = ~torch.isnan(x[:, :1, :])
    m = valid.to(torch.float32)
    h = torch.nan_to_num(x) * m
    for blk in self.net:
        y = blk.conv1(h)
        y = blk.relu(y[..., :-blk.chomp] if blk.chomp else y) * m
        y = blk.drop(y)
        y = blk.conv2(y)
        y = blk.relu(y[..., :-blk.chomp] if blk.chomp else y) * m
        y = blk.drop(y)
        res = blk.down(h) if blk.down is not None else h
        h = blk.relu(y + res) * m
    return self.project(h[..., -1])


def receptive_field(levels, k=3):
    return 1 + 2 * (k - 1) * (2 ** levels - 1)


def pad_batch(seqs, left):
    """base pipeline: right pad with 0. v4: left pad with NaN (marks padding for the mask)."""
    m = max(s.shape[-1] for s in seqs)
    if left:
        return torch.stack([F.pad(s, (m - s.shape[-1], 0), value=float("nan")) for s in seqs])
    return torch.stack([F.pad(s, (0, m - s.shape[-1])) for s in seqs])


def main():
    torch.manual_seed(0)
    C, short_len, long_len = 15, 50, 134        # weather channels; N flight DAPs 50 and 134
    short = torch.randn(C, short_len)
    long_ = torch.randn(C, long_len)
    ok = True
    print(f"receptive field: 4 levels = {receptive_field(4)} d, 6 levels = {receptive_field(6)} d")

    for name, levels, left in (("base pipeline (L=4, right pad)", 4, False),
                               ("v4     (L=6, left pad) ", 6, True)):
        enc = TCNEncoder(C, 64, levels, 3, 32).eval()
        if left:
            enc.forward = masked_forward.__get__(enc)
        with torch.no_grad():
            alone = enc(short[None])[0]
            in_batch = enc(pad_batch([short, long_], left))[0]
            diff_a = (alone - in_batch).abs().max().item()

            x = long_.clone()[None]
            base = enc(x)[0]
            x[..., :30] += 5.0                     # change the first 30 days only
            diff_b = (enc(x)[0] - base).abs().max().item()
        a_pass = diff_a < 1e-5
        b_pass = diff_b > 1e-4
        print(f"{name}  A batch-invariant: {'yes' if a_pass else 'NO '} (max|d| = {diff_a:.2e})   "
              f"B early days reach output: {'yes' if b_pass else 'NO '} (max|d| = {diff_b:.2e})")
        if name.startswith("v4"):
            ok &= a_pass and b_pass
        else:
            ok &= (not a_pass) and (not b_pass)    # the base pipeline problems must reproduce
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
