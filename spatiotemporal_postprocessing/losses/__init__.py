import sys

from .deterministic import MaskedL1Loss
from .probabilistic import MaskedCRPSNormal, MaskedCRPSEnsemble, MaskedCRPSLogNormal

def get_loss(loss_type):
    return getattr(sys.modules[__name__], loss_type)()