import torch 
import torch.nn as nn
from einops.layers.torch import Rearrange
from tsl.nn.layers import GatedGraphNetwork, NodeEmbedding, BatchNorm
from spatiotemporal_postprocessing.nn.probabilistic_layers import dist_to_layer

'''
These models are in the prototype phase and might not perform well yet. 
'''

class CausalConv1d(nn.Conv1d):
    """Causal Convolution ensuring no information leakabe from future to past timesteps"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, padding=0, dilation=dilation, **kwargs)
        self._padding = (kernel_size - 1) * dilation
        
    def forward(self, x):
        x = nn.functional.pad(x, (self._padding, 0))
        return super().forward(x)
        
class TCNLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_p=0.1, causal_conv=True):
        
        super().__init__()
        
        if causal_conv:
            self.cconv = CausalConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation)
        else:
            padding = int(dilation * (kernel_size -1) / 2) # Padding to keep the input shape the same as the output shape
            self.cconv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout1d(p=dropout_p)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = x
        x = self.cconv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        residual = self.downsample(residual) # may be an identity

        return residual + x, x
    
    
class TCN_GNN(nn.Module):
    def __init__(self, num_layers, input_size, output_dist, hidden_channels, n_stations, kernel_size=3, dropout_p=0.2, causal_conv=True,**kwargs):
        super().__init__()
        
        tcn_layers = []
        gnn_layers = []
        norm_layers = []
        
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]*num_layers
        assert len(hidden_channels) == num_layers
        
        self.rearrange_for_gnn = Rearrange('(b n) c t -> b t n c', n=n_stations)
        self.rearrange_for_tcn = Rearrange('b t n c -> (b n) c t')

        self.station_embeddings = NodeEmbedding(n_stations, hidden_channels[0])
        self.encoder = nn.Linear(input_size, hidden_channels[0])

        for l in range(num_layers):
            dilation = 2**l
            input_size = hidden_channels[0] if l == 0 else hidden_channels[l-1]
            tcn_layers.append(TCNLayer(in_channels=input_size, 
                                       out_channels=hidden_channels[l], 
                                       kernel_size=kernel_size, 
                                       dilation=dilation, dropout_p=dropout_p, causal_conv=causal_conv))
            
            gnn_layers.append(GatedGraphNetwork(input_size=hidden_channels[l], output_size=hidden_channels[l]))
            norm_layers.append(BatchNorm(in_channels=hidden_channels[l]))
            
        self.tcn_layers = nn.ModuleList(tcn_layers)
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.norm_layers = nn.ModuleList(norm_layers)
        
        self.output_distr = dist_to_layer[output_dist](input_size=hidden_channels[-1])
        
    def forward(self, x, edge_index):
        
        x = self.encoder(x)
        x = x + self.station_embeddings()
        
        x = self.rearrange_for_tcn(x)
        
        skips = []
        for tcn_l, gnn_l, norm_l in zip(self.tcn_layers, self.gnn_layers, self.norm_layers):
            x, skip = tcn_l(x)          # Temporal convolution
            x = self.rearrange_for_gnn(x)
            x = gnn_l(x, edge_index)    # Spatial convolution \forall time steps t
            x = norm_l(x)
            x = self.rearrange_for_tcn(x)
            
            skips.append(skip)
            
        skips_stack = torch.stack(skips, dim=-2)
        result = skips_stack.sum(dim=-2) + x

        output = self.rearrange_for_gnn(result)
        
        return self.output_distr(output)