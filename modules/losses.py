import torch
import torch.nn as nn
import torch.nn.functional as F


class SteelLoss(nn.Module):
    def __init__(self, loss_dict):
        super(SteelLoss, self).__init__()

        assert len(loss_dict) > 0
        self.lweights = []
        for k,v in loss_dict.items():
            assert 'weight' in v
            self.lweights.append(v['weight'])
            del v['weight']
        self.lweights = torch.tensor(self.lweights).cuda()

        self.loss_dict = loss_dict
        self.losses = [globals()[k](**v) for k,v in self.loss_dict.items()]

    def forward(self, inputs, targets):
        loss = self.losses[0](inputs, targets)
        for i in range(1, len(self.losses)):
            loss += self.losses[i](inputs, targets)
        loss = (loss*self.lweights).sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.permute(1,0,2,3).reshape(4, -1)
        targets = targets.permute(1,0,2,3).reshape(4, -1)
        
        intersection = (inputs * targets).sum(1)                            
        dice = (2.*intersection + smooth)/(inputs.sum(1) + targets.sum(1) + smooth)  
        dice = torch.mean(dice)

        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.permute(1,0,2,3).reshape(4, -1)
        targets = targets.permute(1,0,2,3).reshape(4, -1)

        #first compute binary cross-entropy 
        BCE = torch.mean(F.binary_cross_entropy(inputs, targets, reduction='none'), dim=1)
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
        focal_loss = torch.mean(focal_loss)

        return focal_loss


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.permute(1,0,2,3).reshape(4, -1)
        targets = targets.permute(1,0,2,3).reshape(4, -1)
        
        intersection = (inputs * targets).sum(1)                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum(1) + targets.sum(1) + smooth)  
        BCE = torch.mean(F.binary_cross_entropy(inputs, targets, reduction='none'), dim=1)
        Dice_BCE = BCE + dice_loss
        Dice_BCE = torch.mean(Dice_BCE)

        return Dice_BCE

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.permute(1,0,2,3).reshape(4, -1)
        targets = targets.permute(1,0,2,3).reshape(4, -1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum(1)
        total = (inputs + targets).sum(1)
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
        IoU = torch.mean(IoU)

        return 1 - IoU


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       

        #flatten label and prediction tensors
        inputs = inputs.permute(1,0,2,3).reshape(4, -1)
        targets = targets.permute(1,0,2,3).reshape(4, -1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum(1)    
        FP = ((1-targets) * inputs).sum(1)
        FN = (targets * (1-inputs)).sum(1)

        Tversky = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)  
        Tversky = torch.mean(Tversky)

        return 1 - Tversky

