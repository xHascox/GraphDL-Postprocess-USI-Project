from einops import rearrange
from numpy._core.numeric import False_
import torch.nn as nn 
import torch
from typing import Literal
from tsl.nn.layers import GatedGraphNetwork, NodeEmbedding, BatchNorm
from tsl.nn.models import GraphWaveNetModel
from spatiotemporal_postprocessing.nn.probabilistic_layers import dist_to_layer, LogNormalLayer
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import inits
from torch.nn.init import xavier_uniform_

from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LogNormal
from spatiotemporal_postprocessing.nn.msgwn_modules import *


class LayeredGraphRNN(nn.Module):
    def __init__(self, input_size, hidden_channels, num_layers=1, dropout_p = 0.1, mode: Literal['forwards', 'backwards'] = 'forwards', **kwargs) -> None:
        super().__init__(**kwargs)
        layers_ = []
        
        self.input_encoder = nn.Linear(input_size, hidden_channels)
        
        for _ in range(num_layers):
            layers_.append(GatedGraphNetwork(input_size=hidden_channels*2,
                                             output_size=hidden_channels))

        
        self.mp_layers = torch.nn.ModuleList(layers_)
        self.state_size = hidden_channels * num_layers
        self.num_layers = num_layers
        self.mode = mode
        self.dropout = nn.Dropout(p=dropout_p)
            
    def iterate_layers(self, state, x, edge_index):
        output = []
        state_ = rearrange(state, "b n ... (h l) -> l b n ... h", l=self.num_layers)
        for l, layer in enumerate(self.mp_layers):
            state_in_ = state_[l]
            
            input_ = torch.concatenate([state_in_, x], dim=-1) # recurrency 
            input_ = self.dropout(input_)
            
            x = layer(input_, edge_index) # potential to do x += here
            if isinstance(x, tuple):
                x = x[0] # if cell is a GAT, it returns the alphas
            output.append(x)
        
        return torch.cat(output, dim=-1)   
        
        
    def forward(self, x, edge_index):
        batch_size, win_size, num_nodes, num_feats = x.size()
        state = torch.zeros(batch_size, num_nodes, self.state_size, device=x.device)

        states = []
        # iterate forwards or backwards in time
        t0 = 0 if self.mode == 'forwards' else win_size - 1
        tn = win_size if self.mode == 'forwards' else -1 
        step = 1 if self.mode == 'forwards' else -1
        
        for t in range(t0, tn, step):
            x_ = self.input_encoder(x[:,t])

            # iterate over the depth
            state_out = self.iterate_layers(state=state, x=x_, edge_index=edge_index)
            
            state = state_out + state # skip connection in time
            
            if self.mode == 'forwards':
                states.append(state)
            else: 
                states.insert(0, state)

        
        return torch.stack(states, dim=1)
            


class BiDirectionalSTGNN(nn.Module):
    def __init__(self, input_size, hidden_channels, n_stations, output_dist: str, num_layers=1, dropout_p = 0.1, kernel_size=None, causal_conv=None, **kwargs) -> None:
        super().__init__(**kwargs)
        
        
        
        self.encoder = nn.Linear(input_size, hidden_channels)
        self.station_embeddings = NodeEmbedding(n_stations, hidden_channels)
        self.forward_model = LayeredGraphRNN(input_size=hidden_channels, hidden_size=hidden_channels, n_layers=num_layers, mode='forwards', dropout_p=dropout_p)
        self.backward_model = LayeredGraphRNN(input_size=hidden_channels, hidden_size=hidden_channels, n_layers=num_layers, mode='backwards', dropout_p=dropout_p)
        
        self.output_distr = dist_to_layer[output_dist](input_size=hidden_channels)
            
        self.skip_conn = nn.Linear(input_size, 2*hidden_channels*num_layers)
        #self.skip_conn = nn.Linear(input_size, hidden_channels*num_layers) # TODO RM

        
        self.readout = nn.Sequential(
            nn.Linear(2*hidden_channels*num_layers, hidden_channels),
            #nn.Linear(hidden_channels*num_layers, hidden_channels),
            BatchNorm(in_channels=hidden_channels, track_running_stats=False),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        
    def forward(self, x, edge_index):
        x0 = x
        x = self.encoder(x)
        x = x + self.station_embeddings()
        states_forwards = self.forward_model(x, edge_index)  
        states_backwards = self.backward_model(x, edge_index)
        
        # TODO
        states = torch.concatenate([states_forwards, states_backwards], dim=-1)
        states = states + self.skip_conn(x0) # skip conn 
        
        output = self.readout(states)
        
        return self.output_distr(output)
    
    
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_dist: str, dropout_p, activation: str = "relu", **kwargs):
        super().__init__()
        
        activation_map = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU
        }
        layers = []
        self.input_size = input_size
        for hs in hidden_sizes:
            layers.append(nn.Linear(input_size, hs))
            layers.append(activation_map[activation]())
            layers.append(nn.Dropout(p=dropout_p))
            
            input_size = hs 
        self.layers = nn.Sequential(*layers)
        
        self.output_distr = dist_to_layer[output_dist](input_size=hidden_sizes[-1])
        
        self.skip_conn = nn.Linear(self.input_size, hidden_sizes[-1])
        
    def forward(self, x, **kwargs):
        # ignore edge index
        x_skip = self.skip_conn(x)
        
        x = self.layers(x)
        x = x + x_skip
        
        return self.output_distr(x)
    
    
class WaveNet(nn.Module):
    
    def __init__(self, input_size, time_steps, hidden_size, n_stations, output_dist, ff_size=256, n_layers=6, temporal_kernel_size=3, spatial_kernel_size=2, **kwargs):
        super().__init__()
    
        self.wavenet = GraphWaveNetModel(input_size=input_size, 
                          output_size=hidden_size,
                          horizon=time_steps, 
                          hidden_size=hidden_size, 
                          ff_size=ff_size,
                          n_layers=n_layers,
                          temporal_kernel_size=temporal_kernel_size, spatial_kernel_size=spatial_kernel_size, 
                          dilation=2, dilation_mod=3, n_nodes=n_stations)
        self.output_distr = dist_to_layer[output_dist](input_size=hidden_size)
        
    def forward(self, x, edge_index):
        output = self.wavenet(x, edge_index)
        
        return self.output_distr(output)
   

# Adapted from https://torch-spatiotemporal.readthedocs.io/en/latest/_modules/tsl/nn/layers/norm/layer_norm.html#LayerNorm
class LayerNorm(torch.nn.Module):
    r"""Applies layer normalization.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
    """

    def __init__(self, in_channels, eps=1e-5, affine=True):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        if affine:
            self.weight = Parameter(torch.Tensor(in_channels))
            self.bias = Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        inits.ones(self.weight)
        inits.zeros(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """"""
        mean = torch.mean(x, dim=-1, keepdim=True)

        std = torch.std(x, dim=-1, unbiased=False, keepdim=True) 
         # NOTE here we get some inf values, so this fixes it
         # NOTE max has to be 1e18 at least, performance is worse if it is too small
        std = torch.clamp(std, min=1e-8, max=1e19)

        out = (x - mean) / (std + self.eps)

        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels})'


class EnhancedBiDirectionalSTGNN(nn.Module):
    def __init__(self, input_size, hidden_channels, n_stations, output_dist: str,
                 num_layers=1, dropout_p=0.1, kernel_size=None, causal_conv=None,  **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_stations = n_stations

        # Linear layer to encode input features to hidden channels
        self.encoder = nn.Linear(input_size, hidden_channels)

        # Xavier Initialization of Station Embeddings
        embedding_init = torch.empty(n_stations, hidden_channels)
        xavier_uniform_(embedding_init)
        self.station_embeddings = NodeEmbedding(n_stations, hidden_channels, initializer=embedding_init)
        
        # Forward and backward Layered Graph RNN models (the "Bi-Directional" in the name)
        self.forward_model = LayeredGraphRNN(input_size=hidden_channels, hidden_channels=hidden_channels, num_layers=num_layers, mode='forwards', dropout_p=dropout_p)
        self.backward_model = LayeredGraphRNN(input_size=hidden_channels, hidden_channels=hidden_channels, num_layers=num_layers, mode='backwards', dropout_p=dropout_p)

        # Layer normalization and attention for temporal features, starting from here we need 
        # shape 2*hidden_channels*num_layers because of the bidirectional layers.
        #self.temporal_ln = nn.LayerNorm(2*hidden_channels*num_layers, eps=1e-4)
        self.temporal_ln = LayerNorm(2*hidden_channels*num_layers, eps=1e-4) # NOTE: Default tsl implementation leads to errors
        self.temporal_attn = nn.MultiheadAttention(2*hidden_channels*num_layers, num_heads=4, batch_first=True)

        self.gate = nn.Linear(2*hidden_channels*num_layers, 2*hidden_channels*num_layers)
        
        self.skip_conn = nn.Linear(input_size, 2*hidden_channels*num_layers)
        
        # Layer normalization and attention for spatial features (stations)
        self.station_ln = nn.LayerNorm(2*hidden_channels*num_layers)
        self.station_attn = nn.MultiheadAttention(2*hidden_channels*num_layers, num_heads=2, batch_first=True)
        
        # Readout layer: Here we change back to shape hidden_channels from 2*hidden_channels*num_layers
        self.readout = nn.Sequential(
            nn.Linear(2*hidden_channels*num_layers, hidden_channels),
            BatchNorm(in_channels=hidden_channels, track_running_stats=False),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        self.output_distr = dist_to_layer[output_dist](input_size=hidden_channels)
        
    def forward(self, x, edge_index):
        B, T, N, C = x.shape
        x0 = x

        x = self.encoder(x)
        x = x + self.station_embeddings()

        states_forwards = self.forward_model(x, edge_index)  
        states_backwards = self.backward_model(x, edge_index)
        states = torch.concatenate([states_forwards, states_backwards], dim=-1)
        states = states + self.skip_conn(x0)

        # Rearrange states for temporal attention
        states = rearrange(states, 'b t n c -> (b n) t c')
        states = self.temporal_ln(states)
        attn_out, _ = self.temporal_attn(states, states, states)
        states = states + attn_out
        
        # Apply gating mechanism and another skip connection
        gate = torch.sigmoid(self.gate(states))
        skip = self.skip_conn(rearrange(x0, 'b t n f -> (b n) t f'))
        states = gate * states + (1 - gate) * skip
        
        # Spatial attention
        states2 = rearrange(states, '(b n) t c -> (b t) n c', b=B, n=N)
        states2 = self.station_ln(states2)
        stat_out, _ = self.station_attn(states2, states2, states2)
        stat_out = rearrange(stat_out, '(b t) n c -> b t n c', b=B, t=T)

        states = rearrange(states, '(b n) t c -> b t n c', n=N)
        states = states + stat_out
        
        output = self.readout(states)
        
        return self.output_distr(output)



import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tsl.nn.layers import GatedGraphNetwork, BatchNorm

class CausalConv1d(nn.Conv1d):
    """Causal convolution with padding=(kernel_size−1)*dilation on the left."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__(in_channels, out_channels,
                         kernel_size=kernel_size,
                         padding=0,
                         dilation=dilation,
                         **kwargs)
        self._padding = (kernel_size - 1) * dilation

    def forward(self, x):
        # pad only on the left
        x = F.pad(x, (self._padding, 0))
        return super().forward(x)


class EnhancedTCNGNN(nn.Module):
    """
    Enhanced TCN and GNN spatiotemporal backbone.

    This model begins by learning a distinct embedding for each weather station and for each
    forecast horizon, which are added to the initial input projections. By giving the network
    explicit station and time step embeddings it can learn location specific biases (e.g. alpine
    versus lowland) and dynamically adjust its treatment of near term versus long term lead
    times across the full 96 hour window.

    In each dilated TCN layer we follow the causal convolution with BatchNorm and a ReLU,
    then fuse its output via a residual connection. Between every TCN block and the GNN
    message passing we reshape via einops so the graph sees a clean [B, T, N, C] tensor.
    After stacking these TCN and GNN blocks we apply a temporal LayerNorm and full
    multi head self attention along the time axis. This combination stabilizes training,
    allows arbitrarily deep receptive fields, and via attention lets the model reweight
    past and future time steps beyond fixed dilation patterns.

    To prevent over reliance on deep representations we compute a learned sigmoid gate
    that interpolates between the self attended features and a linear skip projection of
    the raw inputs. Finally we reshape back to [B, T, N, C] and apply a second
    multi head attention across the station dimension, discovering non local spatial
    dependencies that a fixed distance based graph might miss.
    """
    def __init__(
        self,
        input_size: int,
        hidden_channels: int,
        output_dist: str,
        n_stations: int,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout_p: float = 0.2,
        causal_conv: bool = True,
        attn_heads: int = 4,
        station_heads: int = 2,
        max_lead: int = 96,
    ):
        super().__init__()
        # embeddings
        self.station_emb = nn.Embedding(n_stations, hidden_channels)
        self.horizon_emb = nn.Embedding(max_lead + 1, hidden_channels)

        # initial encoder
        self.input_proj = nn.Linear(input_size, hidden_channels)

        # einops helpers
        self.to_tcn    = Rearrange('b t n c -> (b n) c t', n=n_stations)
        self.from_tcn  = Rearrange('(b n) c t -> (b n) t c', n=n_stations)
        self.unflatten = Rearrange('(b n) t c -> b t n c', n=n_stations)

        self.tcn_layers  = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.gnn_layers  = nn.ModuleList()
        for l in range(num_layers):
            in_ch  = hidden_channels
            out_ch = hidden_channels
            conv = (CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**l)
                    if causal_conv
                    else nn.Conv1d(in_ch, out_ch, kernel_size,
                                   padding=(kernel_size-1)//2,
                                   dilation=2**l))
            self.tcn_layers.append(conv)
            self.norm_layers.append(nn.BatchNorm1d(out_ch))
            self.gnn_layers.append(GatedGraphNetwork(input_size=out_ch,
                                                     output_size=out_ch))

        self.temporal_ln   = nn.LayerNorm(hidden_channels)
        self.temporal_attn = nn.MultiheadAttention(hidden_channels,
                                                   num_heads=attn_heads,
                                                   batch_first=True)

        self.gate      = nn.Linear(hidden_channels, hidden_channels)
        self.skip_proj = nn.Linear(input_size, hidden_channels)

        self.station_ln   = nn.LayerNorm(hidden_channels)
        self.station_attn = nn.MultiheadAttention(hidden_channels,
                                                  num_heads=station_heads,
                                                  batch_first=True)

        # output head
        self.output_distr = dist_to_layer[output_dist](input_size=hidden_channels)

    def forward(self, x, edge_index):
        B, T, N, C = x.shape

        h = self.to_tcn(x)

        ids      = repeat(torch.arange(N, device=x.device),
                          'n -> b n', b=B)
        ids      = rearrange(ids, 'b n -> (b n)')
        stat_emb = self.station_emb(ids)
        stat_emb = stat_emb.unsqueeze(-1).expand(-1, -1, T)


        h_proj = self.input_proj(rearrange(h, 'bn f t -> bn t f'))
        h = rearrange(h_proj, 'bn t h -> bn h t')
        h = h + stat_emb

        skips = []
        for tcn, bn, gnn in zip(self.tcn_layers,
                                self.norm_layers,
                                self.gnn_layers):
            # TCN
            res = h
            h   = tcn(h)
            h   = bn(h)
            h   = F.relu(h)
            h   = h + res
            skips.append(h)

            h_seq = self.unflatten(self.from_tcn(h))
            # GNN
            h_seq = gnn(h_seq, edge_index)
            h = self.to_tcn(h_seq)

        h = sum(skips)
        h = self.unflatten(self.from_tcn(h))

        h = rearrange(h, 'b t n h -> (b n) t h')
        h = self.temporal_ln(h)
        attn_out, _ = self.temporal_attn(h, h, h)
        h = h + attn_out

        raw  = rearrange(x, 'b t n f -> (b n) t f')
        skip = self.skip_proj(raw)
        gate = torch.sigmoid(self.gate(h))
        h    = gate * h + (1 - gate) * skip

        steps = repeat(torch.arange(T, device=x.device),
                       't -> (b n) t', b=B, n=N)
        h = h + self.horizon_emb(steps)

        h2 = rearrange(h, '(b n) t h -> (b t) n h', b=B, n=N)
        h2 = self.station_ln(h2)
        stat_out, _ = self.station_attn(h2, h2, h2)
        stat_out = rearrange(stat_out, '(b t) n h -> b t n h', b=B, t=T)

        h = rearrange(h, '(b n) t h -> b t n h', b=B, n=N)
        h = h + stat_out

        return self.output_distr(h)




################################################################################
# ─────────────────── MSGWN main class (distribution‑aware) ────────────────────
################################################################################
class MultiScaleGraphWaveNet(nn.Module):
    def __init__(self,
                 n_stations: int,
                 in_channels: int,
                 history_len: int,
                 horizon: int = 1, # only for point forecasts
                 layers: int = 3,
                 channels: int = 16,
                 emb_dim: int = 16,
                 node_emb_dim: int = 20,
                 adj_matrix: Optional[np.ndarray] = None,
                 learnable_adj: bool = True,
                 kernels = (1, 3, 5, 7),
                 dil = (1, 2, 4, 8),
                 drop = 0.2,
                 edge_drop_p = 0.2,
                 history_dropout_p: float = 0.07,
                 history_block:   int   = 12,
                 dynamic=True,
                 output_dist: Optional[str] = "LogNormal"):
        super().__init__()
        self.horizon = horizon
        self.edge_drop_p = edge_drop_p
        self.num_nodes = n_stations
        # adjacency
        self.node_embeddings = nn.Embedding(self.num_nodes, node_emb_dim) # node_emb_dim is a new hyperparameter
        nn.init.xavier_uniform_(self.node_embeddings.weight)
        if adj_matrix is not None and not learnable_adj:
            self.register_buffer('A_fixed', torch.tensor(adj_matrix, dtype=torch.float32))
            self.adj = None
        else:
            self.A_fixed = None
            self.adj = LearnableAdjacency(self.num_nodes, emb_dim, init_A=adj_matrix)
        # layers
        #self.in_proj = nn.Conv2d(in_channels, channels, 1)

        self.history_dropout = HistoryDropout(p=history_dropout_p,
                                              block_size=history_block)
        self.in_proj = nn.Conv2d(in_channels + node_emb_dim, channels, 1)
        self.blocks = nn.ModuleList([STBlock(channels, channels, kernels, dil, drop, dynamic) for _ in range(layers)])
        self.skip_proj = nn.Conv2d(layers*channels, channels, 1)
        self.end1 = nn.Conv2d(channels, channels, 1)

        # param projection (μ, σ) per horizon
        out_dim = 2 * horizon if output_dist=='LogNormal' else horizon
        self.param_conv = nn.Conv2d(channels, out_dim, 1)
        self.output_dist = output_dist
        if output_dist == 'LogNormal':
            # project to model channels (distribution input size)
            self.param_conv = nn.Conv2d(channels, channels, 1)
            self.dist_layer = LogNormalLayer(input_size=channels)
        else:
            # point forecast: one output per horizon
            self.param_conv = nn.Conv2d(channels, horizon, 1)

    # supports helper
    def _supports(self):
        A = self.A_fixed if self.A_fixed is not None else self.adj()
        return [A, A @ A]

        # ---- insert edge-dropout here ------------------------------------
        if self.training and self.edge_drop_p > 0:          # dropout only while training
            mask = torch.bernoulli(A.new_full(A.shape, 1.0 - self.edge_drop_p))
            A = A * mask                                    # zero out ~p fraction of edges
        # ------------------------------------------------------------------

        return [A, A @ A]      # first- and second-order supports

    def forward(self, x):  # x: (B, C_in, N, T)
        #x = x.permute(0, 3, 2, 1)

        # --- START OF NODE EMBEDDING INTEGRATION ---
        B, C_original, N, T = x.shape # Get dimensions from input
        
        # 1. Get node embeddings
        node_idx = torch.arange(self.num_nodes, device=x.device)
        n_emb = self.node_embeddings(node_idx)  # Shape: (N, node_emb_dim)
        
        # 2. Expand embeddings to match input tensor's batch and time dimensions
        #    Target shape for n_emb: (B, node_emb_dim, N, T)
        n_emb_expanded = n_emb.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, T)
        n_emb_expanded = n_emb_expanded.permute(0, 2, 1, 3) # Permute to (B, node_emb_dim, N, T)

        x = self.history_dropout(x)
        # 3. Concatenate with original input features along the channel dimension
        x_augmented = torch.cat([x, n_emb_expanded], dim=1) # Shape: (B, C_original + node_emb_dim, N, T)
        # --- END OF NODE EMBEDDING INTEGRATION ---
        
        # Now, use x_augmented for the rest of the forward pass, starting with in_proj
        supports = self._supports()
        x = self.in_proj(x_augmented) # Pass the augmented features to in_pro
        supports = self._supports()
        #x = self.in_proj(x)
        skips = []
        for blk in self.blocks:
            x, s = blk(x, supports)
            skips.append(s)
        x = F.relu(self.skip_proj(torch.cat(skips, dim=1)))
        x = F.relu(self.end1(x))
        params = self.param_conv(x)    # (B,out_dim,N,T)
        if self.output_dist=='LogNormal':
            # shape to (B,N,T,2)
            params = params.permute(0,2,3,1)
            return self.dist_layer(params)
        else:
            out = params.squeeze(1)     # (B,N,T)
            return out.permute(0,2,1)   # (B,T,N)


##### MY MODEL BASELINE #####
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class EnhancedGRUBaseline(nn.Module):
    r"""
    # EnhancedGRUBaseline

    EnhancedGRUBaseline takes as input a tensor
    $$
      X \in \mathbb{R}^{B\times T\times N\times F}
    $$
    (batch size $B$, forecast length $T$, $N$ stations, $F$ features) and outputs a
    LogNormal distribution over wind speeds at each station and lead time. Its key components:

    We embed station identity via a learned matrix
    $E\in\mathbb{R}^{N\times H}$, so that each station $s$ has embedding $e_s\in\mathbb{R}^H$.
    Adding $e_s$ to the projected inputs breaks permutation invariance and lets the
    network learn station‐specific biases (e.g.\ elevation effects).

    Similarly we learn horizon embeddings
    $$
      h_\ell\in\mathbb{R}^H,\quad \ell=0,\dots,T-1,
    $$
    which are added after the gated‐skip block to encode the systematic decay of
    forecast skill with lead time.

    A linear layer
    $$
      W_{\mathrm{proj}}\!: \mathbb{R}^F\to\mathbb{R}^H
    $$
    projects raw predictors into a hidden space of dimension $H$.

    We chose a bidirectional GRU because gated recurrent units efficiently capture
    local temporal dynamics and handle vanishing gradients via update/reset gates.
    Splitting the hidden size into $H/2$ forward and $H/2$ backward dimensions yields
    a full state $h_t\in\mathbb{R}^H$ that integrates information from both past
    and future within the window.

    Immediately after the GRU we apply LayerNorm:
    $$
      \mathrm{LN}(z_t)=\gamma\frac{z_t-\mu_z}{\sigma_z}+\beta,
    $$
    which stabilizes training by normalizing across the feature dimension, reducing
    internal covariate shift and allowing larger learning rates.

    To capture long‐range dependencies, we add a temporal multi‐head attention block:
    $$
      \mathrm{Attn}(Q,K,V)
      = \mathrm{softmax}\!\bigl(\tfrac{QK^\top}{\sqrt{H/A}}\bigr)\,V,
    $$
    with $A$ heads operating over the time axis.  This augments the GRU’s local
    memory with global context, letting the model re‐weigh past events adaptively.

    A gated residual skip then fuses this attention output $a_t$ with the original
    projected input $p_t$ via
    $$
      g_t=\sigma(W_g a_t + b_g),\quad
      h_t = g_t\odot a_t + (1-g_t)\odot p_t,
    $$
    which both preserves raw ensemble statistics and improves gradient flow.

    We reshape back to $(B,T,N,H)$ and perform station‐axis self‐attention:
    $$
      \mathrm{Attn}\!\bigl(H_t\bigr),\quad H_t\in\mathbb{R}^{(B\,T)\times N\times H},
    $$
    so the model learns spatial correlations among stations in complex terrain.

    Finally, a small linear “distribution head” outputs parameters
    $(\mu,\sigma)$ for a LogNormal:
    $$
      \mu=W_\mu^\top h + b_\mu,\quad
      \sigma=\mathrm{softplus}(W_\sigma^\top h + b_\sigma)+\varepsilon,
    $$
    ensuring $\sigma>0$.  Training minimizes the closed‐form CRPS for LogNormal
    forecasts, which balances calibration and sharpness.
    """
    def __init__(
        self,
        input_size: int,
        hidden_channels: int,
        output_dist: str,
        n_stations: int,
        num_layers: int = 2,
        dropout_p: float = 0.1,
        attn_heads: int = 4,
        station_heads: int = 2,
        max_lead: int = 96,
        kernel_size=None, 
        causal_conv=None,
    ):
        super().__init__()
        # station embeddings
        self.station_emb = nn.Embedding(n_stations, hidden_channels)

        # time embeddings
        self.horizon_emb = nn.Embedding(max_lead+1, hidden_channels)


        # project raw features to hidden dim
        self.input_proj = nn.Linear(input_size, hidden_channels)

        self.gru = nn.GRU(
            hidden_channels,
            hidden_channels // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # layer‐norm on hidden
        self.ln = nn.LayerNorm(hidden_channels)

        # temporal self‐attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=attn_heads,
            batch_first=True
        )

        # gated residual skip
        self.gate = nn.Linear(hidden_channels, hidden_channels)

        # skip from raw inputs
        self.skip_proj = nn.Linear(input_size, hidden_channels)

        # station‐axis self‐attention
        self.station_attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=station_heads,
            batch_first=True
        )

        self.output_distr = dist_to_layer[output_dist](input_size=hidden_channels)

        self.to_seq = Rearrange('b t n f -> (b n) t f')
        self.from_seq = Rearrange('(b n) t h -> b t n h', n=n_stations)

    def forward(self, x, edge_index=None):
        """
        x: [B, T, N, F]
        returns a torch.distributions.Distribution over [B, T, N]
        """
        B, T, N, F = x.shape

        h_in = self.to_seq(x)

        ids = repeat(torch.arange(N, device=x.device), 'n -> b n', b=B)
        ids = rearrange(ids, 'b n -> (b n)')

        emb = self.station_emb(ids)
        emb = emb.unsqueeze(1).expand(-1, T, -1)

        # project inputs + add station embedding
        h_proj = self.input_proj(h_in) + emb

        h_out, _ = self.gru(h_proj)
        h_out = self.ln(h_out)

        # temporal self-attention
        attn_out, _ = self.attn(h_out, h_out, h_out)
        h_temp = h_out + attn_out

        # gated residual skip
        gate = torch.sigmoid(self.gate(h_temp))
        skip = self.skip_proj(h_in)
        h_res = gate * h_temp + (1 - gate) * skip
        le = repeat(torch.arange(T, device=h_res.device), 't -> (b n) t', b=B, n=N)
        h_res = h_res + self.horizon_emb(le)


        h_final = self.from_seq(h_res)


        h2 = rearrange(h_final, 'b t n h -> (b t) n h')
        attn_stat, _ = self.station_attn(h2, h2, h2)
        attn_stat = rearrange(attn_stat, '(b t) n h -> b t n h', b=B, t=T)

        h_final = h_final + attn_stat

        h_final = self.from_seq(h_res)

        return self.output_distr(h_final)


