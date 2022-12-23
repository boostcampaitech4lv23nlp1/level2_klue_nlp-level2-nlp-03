import torch
import numpy as np

def CEloss(outputs, labels):
    criterion = torch.nn.CrossEntropyLoss()
    
    return criterion(outputs, labels)

def Focalloss(outputs, labels):
    focal_loss = torch.hub.load(
    'adeelh/pytorch-multi-class-focal-loss',
    model='FocalLoss',
    alpha=torch.tensor([.75, .25]),
    gamma=2,
    reduction='mean',
    force_reload=False
    )
    
    return focal_loss(outputs, labels)