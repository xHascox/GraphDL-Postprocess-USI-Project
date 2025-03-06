import torch.nn as nn
import torch
from abc import ABC, abstractmethod
from typing import Literal
import sys 
from inspect import getmembers, isclass

class SoftplusWithEps(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.softplus = nn.Softplus()
        self.eps = eps

    def forward(self, x):
        return self.softplus(x) + self.eps
    

class DistributionLayer(nn.Module, ABC):
    def __init__(self, input_size):
        super().__init__()
        
        self.distribution = getattr(torch.distributions, self.name)
        
        self.encoder = nn.Linear(input_size, self.num_parameters)
            
    @property
    @abstractmethod
    def num_parameters(self):
        pass
    
    @property
    @abstractmethod
    def name(self):
        pass
    
    @abstractmethod
    def process_params(self, x):
        pass

    
    def forward(self, x, return_type: Literal['samples', 'distribution']='distribution', reparametrized=True, num_samples=1):
        params = self.encoder(x)
        distribution = self.process_params(params)
        if return_type == 'distribution':
            return distribution
        return distribution.rsample((num_samples,)) if reparametrized else distribution.sample((num_samples,))
        
        
class LogNormalLayer(DistributionLayer):
    _name = 'LogNormal'
    def __init__(self, input_size):
        super(LogNormalLayer, self).__init__(input_size=input_size)
        self.get_positive_std = SoftplusWithEps()
        
    @property
    def name(self):
        return self._name
    
    @property
    def num_parameters(self):
        return 2
    
    def process_params(self, x):
        new_moments = x.clone()  
        new_moments[...,1] = self.get_positive_std(x[...,1])
        
        log_normal_dist = self.distribution(new_moments[...,0:1], new_moments[...,1:2])
        return log_normal_dist
    
    
class NormalLayer(DistributionLayer):
    _name = 'Normal'
    def __init__(self, input_size):
        super(NormalLayer, self).__init__(input_size=input_size)
        self.get_positive_std = SoftplusWithEps()
        
    @property
    def name(self):
        return self._name
    
    @property
    def num_parameters(self):
        return 2
    
    def process_params(self, x):
        new_moments = x.clone()  
        new_moments[...,1] = self.get_positive_std(x[...,1])
        
        normal_dist = self.distribution(new_moments[...,0:1], new_moments[...,1:2])
        return normal_dist
        

prob_layers = [obj[1] for obj in getmembers(sys.modules[__name__], isclass) if issubclass(obj[1], DistributionLayer) and obj[0] != 'DistributionLayer']
dist_to_layer = {
    l._name: l for l in prob_layers
}
