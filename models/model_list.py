import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.models import DynamicUnet
from torchvision.models.resnet import *
from fastai.vision.models.xresnet import *
from models.presnet import *
#from fastai.vision.models.xresnet2 import *
#from fastai.vision.models.presnet import *
#from x2resnet import *
from models.mxresnet import mxresnet18, mxresnet34, mxresnet50, mxresnet101, mxresnet152
from models.mres2net import mres2net34, mres2net50
from models.dynamic_munet import DynamicMUnet

def UnetMxResnet(encoder='mxresnet18', n_classes=5, img_size=(256, 256),
        blur=False, blur_final=True, self_attention=False, y_range=None,
        last_cross=True, bottle=False):
    Net = globals()[encoder]()
    if encoder in ['mxresnet18', 'mxresnet34', 'mxresnet50', 'mxresnet101', 'mxresnet152']:
        NetBase = nn.Sequential(*[i for i in Net.children()][:-3])
    elif encoder in ['mres2net34', 'mres2net50', 'presnet18', 'presnet34', 'presnet50', 'resnet18', 'resnet34']:
        NetBase = nn.Sequential(*[i for i in Net.children()][:-2])
    else:
        raise NotImplementedError
    Unet = DynamicUnet(encoder=NetBase, n_classes=n_classes, img_size=img_size,
        blur=blur, blur_final=blur_final, self_attention=self_attention, y_range=y_range,
        last_cross=last_cross, bottle=bottle)
    return Unet

def MUnetMxResnet(encoder='mxresnet18', n_classes=5, img_size=(256, 256),
        blur=False, blur_final=True, self_attention=False, y_range=None,
        last_cross=True, bottle=False):
    assert encoder in ['mxresnet18', 'mxresnet34', 'mxresnet50', 'mxresnet101', 'mxresnet152']
    MXResnet = globals()[encoder]()
    MXResnetBase = nn.Sequential(*[i for i in MXResnet.children()][:-3])
    MUnet = DynamicMUnet(encoder=MXResnetBase, n_classes=n_classes)
    return MUnet
