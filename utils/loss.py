from torch.nn import functional as F
from torch import nn
import torch
import pdb

def create_criterion(loss_name):
    
    if loss_name == 'CTCLoss':
        criterion = nn.CTCLoss(blank=2, zero_infinity=True)
    elif loss_name == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_name == 'BCEWithLogits':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_name == 'regression':
        criterion = mean_error
    elif loss_name == "BCE":
        criterion = nn.BCELoss()
    return criterion


def mean_error(output, target, loss_type='MSE'):

    # Align the time_steps of output and target
    pdb.set_trace()
    N = min(output.shape[1], target.shape[1])

    output = output[:, 0: N, :]
    target = target[:, 0: N, :]
    
    out = torch.sqrt(torch.sum((output - target)**2))

    return out
