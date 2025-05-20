from einops import rearrange
from numpy._core.numeric import False_
import torch.nn as nn 
import torch
from typing import Literal
from tsl.nn.layers import GatedGraphNetwork, NodeEmbedding, BatchNorm
from tsl.nn.models import GraphWaveNetModel
from spatiotemporal_postprocessing.nn.probabilistic_layers import dist_to_layer
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import inits
from torch.nn.init import xavier_uniform_


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