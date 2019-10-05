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
from models.deeplab.deeplab import DeepLab

def UnetMxResnet(config):
    Net = globals()[config.unet_encoder]()
    if config.unet_encoder in ['mxresnet18', 'mxresnet34', 'mxresnet50', 'mxresnet101', 'mxresnet152']:
        NetBase = nn.Sequential(*[i for i in Net.children()][:-3])
    elif config.unet_encoder in ['mres2net34', 'mres2net50', 'presnet18', 'presnet34', 'presnet50', 'resnet18', 'resnet34']:
        NetBase = nn.Sequential(*[i for i in Net.children()][:-2])
    else:
        raise NotImplementedError
    Unet = DynamicUnet(encoder=NetBase, n_classes=config.num_classes, img_size=(config.imsize, config.imsize),
        blur=config.unet_blur, blur_final=config.unet_blur_final, self_attention=config.unet_self_attention,
        y_range=config.unet_y_range, last_cross=config.unet_last_cross, bottle=config.unet_bottle)
    return Unet

def MUnetMxResnet(config):
    assert config.unet_encoder in ['mxresnet18', 'mxresnet34', 'mxresnet50', 'mxresnet101', 'mxresnet152']
    MXResnet = globals()[config.unet_encoder]()
    MXResnetBase = nn.Sequential(*[i for i in MXResnet.children()][:-3])
    MUnet = DynamicMUnet(encoder=MXResnetBase, n_classes=config.num_classes)
    return MUnet

def deeplab(config):
    return DeepLab(backbone=config.deeplab_backbone, output_stride=16, num_classes=config.num_classes,
                 sync_bn=config.deeplab_sync_bn, freeze_bn=config.deeplab_freeze_bn)
