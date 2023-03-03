from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY

_reduction_modes = ['none', 'mean', 'sum']
@LOSS_REGISTRY.register()
class DistillLoss(nn.Module):
    """Distill Loss.

    Args:
        loss_weight (float): Loss weight for Example loss. Default: 1.0.
    """

    def __init__(self, loss_weight= [0.5, 0.5], reduction='mean'):
        super(DistillLoss, self).__init__()
        self.loss_weight = loss_weight
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

    def forward(self, pred, target, teacher, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight[0] * F.l1_loss(pred, target, reduction='mean') + self.loss_weight[1] * F.l1_loss(pred, teacher, reduction='mean')
