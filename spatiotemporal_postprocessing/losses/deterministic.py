import torch 
import torch.nn as nn

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, predictions, targets):

        mask = ~torch.isnan(targets)
        
        predictions = predictions[mask]
        targets = targets[mask]
        
        # Compute the L1 loss (MAE) on the masked values
        loss = torch.abs(predictions - targets).mean()
        return loss
    
class MaskedMAEGraphWavenet(MaskedL1Loss):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        # build mask: True where target is finite

        mask = torch.isfinite(target)
        valid = mask.sum()
        if valid == 0:
            # no valid points
            return torch.tensor(float('nan'), device=pred.device)
        # compute abs error only on valid entries
        abs_err = (pred[mask] - target[mask]).abs()
        return abs_err.mean(), valid