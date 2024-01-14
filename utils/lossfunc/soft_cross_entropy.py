import torch
import torch.nn.functional as F


class SoftCrossEntropyLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(SoftCrossEntropyLoss, self).__init__()
        
    def forward(self, inputs, target, reduction='average') -> torch.Tensor:
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
        return loss