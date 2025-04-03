import sys

from .models import BiDirectionalSTGNN, MLP, WaveNet
from .prototypes import TCN_GNN
from .prototypes import RNN_Model_1

def get_model(model_type, **kwargs):
     
    model_class = getattr(sys.modules[__name__], model_type)
    
    try:
        return model_class(**kwargs)
    except:
        raise NotImplementedError(f'Could not resolve model {model_type}')