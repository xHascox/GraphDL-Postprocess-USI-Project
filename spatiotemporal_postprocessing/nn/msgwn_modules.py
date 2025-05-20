from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LogNormal



################################################################################
# Graph modules (unchanged up to STBlock)
################################################################################
class LearnableAdjacency(nn.Module):
    def __init__(self, n, emb_dim=10, init_A: Optional[np.ndarray] = None):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(n, emb_dim))
        self.E2 = nn.Parameter(torch.randn(n, emb_dim))
        if init_A is not None:
            A = np.asarray(init_A, dtype=np.float64) + 1e-9
            U, S, Vt = np.linalg.svd(np.log(A), full_matrices=False)
            B = U[:, :emb_dim] @ np.diag(np.sqrt(S[:emb_dim]))
            self.E1.data.copy_(torch.tensor(B, dtype=torch.float32))
            self.E2.data.copy_(torch.tensor(B, dtype=torch.float32))
    def forward(self):
        return F.softmax(self.E1 @ self.E2.T, dim=1)


class HistoryDropout(nn.Module):
    def __init__(self, p: float = 0.05, block_size: int = 12):
        """
        p          – probability *per sample* of zeroing out a block
        block_size – length of the contiguous time-block to mask
        """
        super().__init__()
        self.p = p
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, T)
        if not self.training or self.p <= 0:
            return x

        B, C, N, T = x.shape
        # create a mask of ones
        mask = x.new_ones((B, 1, 1, T))
        for b in range(B):
            if torch.rand((), device=x.device) < self.p:
                # pick a random start index so the block fits
                t0 = torch.randint(0, T - self.block_size + 1, (), device=x.device)
                mask[b, :, :, t0 : t0 + self.block_size] = 0.0
        return x * mask


class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, K=2):
        super().__init__(); self.K = K
        self.theta = nn.Parameter(torch.empty(K, c_in, c_out))
        nn.init.xavier_uniform_(self.theta)
    #def forward(self, x, supports):
    #    out = 0.0
    #    for k in range(self.K):
    #        out += torch.einsum("bcn,nm,cmd->bdm", x, supports[k], self.theta[k])
    #    return out

    def forward(self, x, supports: List[torch.Tensor]):
        # x: (B, C_in, N)
        # supports[k]: (N, N)
        # theta[k]   : (C_in, C_out)
        out = 0.0
        for k in range(self.K):
            # 1) diffuse along the graph:
            #    (B,C_in,N) x (N,N) → (B,C_in,N)
            xk = torch.einsum('bcn,nm->bcm', x, supports[k])
            # 2) mix channels:
            #    (B,C_in,N) x (C_in,C_out) → (B,C_out,N)
            out = out + torch.einsum('bcm,cd->bdm', xk, self.theta[k])
        return out


class KernelDilatedTCN(nn.Module):
    """One kernel size k, four dilations 1,2,4,8, causal padding."""
    def __init__(self, c_in: int, c_out: int, k: int, dil: tuple, drop: float):
        super().__init__()
        self.kernel_size = k
        # 1×1 pre-mix
        self.pre = nn.Conv2d(c_in, c_out, kernel_size=1)
        # four dilated convs *without* internal padding
        #self.branches = nn.ModuleList([
        #    
        #    nn.Conv2d(
        #        in_channels=c_out, out_channels=2*c_out,
        #        kernel_size=(1, k),
        #        dilation=(1, d),
        #        padding=0
        #    )
        #    for d in (1, 2, 4, 8)
        #])
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(c_out),
                nn.Dropout2d(drop),
                nn.Conv2d(c_out, 2*c_out, (1,k), dilation=(1,d), padding=0)
            ) for d in dil
        ])

        self.dil = dil
        # fuse back to c_out
        #self.merge = nn.Conv2d(4*c_out, c_out, kernel_size=1)
        self.merge = nn.Conv2d(len(dil)*c_out, c_out, 1)   # ← variable
        # residual projection
        self.res   = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # x: (B, C_in, N, T)
        z = self.pre(x)  # (B, C_out, N, T)
        outs = []
        for conv, d in zip(self.branches, self.dil):
            #pad = (self.kernel_size - 1) * d
            pad_val = (self.kernel_size - 1) * d
            # only pad on the *left* of the time axis:
            # F.pad pads last two dims by (left,right) pairs = (pad_left,pad_right)
            #z_pad = F.pad(z, (pad, 0))  # → (B, C_out, N, T + pad)
            # non causal padding
            z_pad = F.pad(z, (pad_val // 2, pad_val - pad_val // 2))
            # conv without extra padding then chops back to length T
            p, q = conv(z_pad).chunk(2, dim=1)  # each (B, C_out, N, T)
            outs.append(torch.tanh(p) * torch.sigmoid(q))
        y = self.merge(torch.cat(outs, dim=1))  # (B, C_out, N, T)
        y = F.relu(y)
        return y + self.res(x)

class DynamicKernelDilatedTCN(nn.Module):
    def __init__(self, c_in, c_out, k, dilations: tuple, drop: float = 0.1, r: int = 8):
        super().__init__()
        self.kernel_size = k
        self.dilations = dilations

        # 1×1 pre-mix
        self.pre = nn.Conv2d(c_in, c_out, 1)

        # dilated branches (exactly as before)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(c_out),
                nn.Dropout2d(drop),
                nn.Conv2d(c_out, 2*c_out, (1, k), dilation=(1, d), padding=0)
            )
            for d in dilations
        ])

        # gating network: global-pool → FC → one logit per branch
        # produces shape (B, num_branches, 1, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                # (B, C, 1, 1)
            nn.Conv2d(c_out, c_out//r, 1),          # squeeze
            nn.ReLU(),
            nn.Conv2d(c_out//r, len(dilations), 1)  # one logit per branch
        )

        # no more static fuse conv
        # final 1×1 to mix back to c_out
        self.project = nn.Conv2d(c_out, c_out, 1)
        self.res     = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        # x: (B, c_in, N, T)
        z = self.pre(x)  # (B, c_out, N, T)

        # 1) compute branch outputs
        outs = []
        for conv, d in zip(self.branches, self.dilations):
            pad = (self.kernel_size - 1) * d
            z_pad = F.pad(z, (pad//2, pad - pad//2))  # or causal
            p, q = conv(z_pad).chunk(2, dim=1)        # each (B,c_out,N,T)
            outs.append(torch.tanh(p) * torch.sigmoid(q))

        # 2) compute gates
        # gate_logits: (B, num_branches, 1, 1)
        gate_logits = self.gate(z)
        # softmax over branch dim → (B, num_branches, 1, 1)
        gate_weights = F.softmax(gate_logits, dim=1)

        # 3) weighted sum of branches
        y = 0
        for i, out_i in enumerate(outs):
            # out_i: (B,c_out,N,T); weight_i broadcast across C,N,T
            w_i = gate_weights[:, i:i+1, :, :]      # (B,1,1,1)
            y = y + out_i * w_i

        # 4) final projection + residual
        y = self.project(y)                        # (B,c_out,N,T)
        return F.relu(y + self.res(x))


class MultiScaleTCN(nn.Module):
    def __init__(self, c_in, c_out, kernels: tuple, dil, drop, dynamic=False):
        super().__init__()
        if dynamic:
            self.cols = nn.ModuleList([DynamicKernelDilatedTCN(c_in, c_out, k, dil, drop) for k in kernels])
        else:
            self.cols = nn.ModuleList([KernelDilatedTCN(c_in, c_out, k, dil, drop) for k in kernels])
        self.fuse = nn.Conv2d(len(kernels)*c_out, c_out, 1); self.res = nn.Conv2d(c_in, c_out, 1)
    def forward(self, x):
        y = torch.cat([col(x) for col in self.cols], dim=1)
        y = F.relu(self.fuse(y))
        return y + self.res(x)


class STBlock(nn.Module):
    def __init__(self, c_in, channels, kernels, dil, drop, dynamic):
        super().__init__(); self.tcn = MultiScaleTCN(c_in, channels, kernels, dil, drop, dynamic)
        self.gcn = GraphConv(channels, channels)
        self.skip = nn.Conv2d(channels, channels, 1)
        self.res  = nn.Conv2d(c_in, channels, 1)
    def forward(self, x, supports):
        y = self.tcn(x)
        B,C,N,T = y.shape
        y = y.permute(0,3,1,2).reshape(B*T, C, N)
        y = self.gcn(y, supports).view(B,T,C,N).permute(0,2,3,1)
        return F.relu(y + self.res(x)), self.skip(y)