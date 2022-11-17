import numpy as np
import torch

def CELoss(logits, labels):
    acc_loss = 0
    for i in range(len(logits)):
        tmp = logits[i]
        y = labels[i]
        loss = np.log(sum(np.exp(tmp))) - tmp[y]
        acc_loss += loss
    
    return torch.Tensor(acc_loss / len(logits))