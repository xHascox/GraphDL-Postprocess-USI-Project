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