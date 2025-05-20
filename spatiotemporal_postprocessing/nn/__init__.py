import sys

from .models import BiDirectionalSTGNN, MLP, WaveNet, EnhancedBiDirectionalSTGNN
from .prototypes import TCN_GNN

def get_model(model_type, **kwargs):
     
    model_class = getattr(sys.modules[__name__], model_type)
    
    try:
        return model_class(**kwargs)
    except:
        raise NotImplementedError(f'Could not resolve model {model_type}')