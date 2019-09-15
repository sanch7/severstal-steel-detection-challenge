import numpy as np
import torch
from fastai.torch_core import *

def dice(input:Tensor, targs:Tensor, iou:bool=False, eps:float=1e-8)->Rank0Tensor:
    n,c = targs.shape[0], input.shape[1]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    intersect,union = [],[]
    for i in range(1,c):
        intersect.append(((input==i) & (targs==i)).sum(-1).float())
        union.append(((input==i).sum(-1) + (targs==i).sum(-1)).float())
    intersect = torch.stack(intersect)
    union = torch.stack(union)
    if not iou: return ((2.0*intersect + eps) / (union+eps)).mean()
    else: return ((intersect + eps) / (union - intersect + eps)).mean()