import os
import sys
import glob
from tqdm import tqdm
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from easydict import EasyDict

from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from fastai.vision.models import DynamicUnet
from fastai.vision.data import imagenet_stats
from fastai.torch_core import spectral_norm, tensor, Module

START_POS = list(range(0, 1600 - 256 + 32, 256 - 32))
TEST_IMAGES_PATH = './data/test_images'
TRAIN_IMAGES_PATH = './data/train_images'

config = EasyDict()

# experiment details
config.exp_name = "run14"
config.metric_name = "loss"
config.gpu = None
config.fp16 = True

# model framework
config.batch_size = 32
config.imsize = 256
# config.num_workers = os.cpu_count()
config.num_workers = 0

# architecture details
config.model_save_path = lambda : './model_weights/{}/best_{}.pth'.format(config.exp_name, config.metric_name)
config.model_name = "deeplab"
config.unet_encoder = "mxresnet34"
config.num_classes = 4
config.unet_blur = True
config.unet_blur_final = True
config.unet_self_attention = False
config.unet_y_range = None
config.unet_last_cross = True
config.unet_bottle = False

config.deeplab_backbone='xception'
config.deeplab_sync_bn = True
config.deeplab_freeze_bn = False

#inference details
config.best_threshold = 0.5
config.min_size = 3500

cudnn.benchmark = True
cudnn.enabled = True

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        # print("Mish activation loaded...")

    def forward(self, x):
        # save 1 second per epoch with no x= x*() and then return x...just inline it.
        return x * (torch.tanh(F.softplus(x)))

    # Unmodified from https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py


def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)


# Adapted from SelfAttention layer at
# https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
# Inspired by https://arxiv.org/pdf/1805.08318.pdf
class SimpleSelfAttention(nn.Module):

    def __init__(self, n_in: int, ks=1, sym=False):  # , n_out:int):
        super().__init__()

        self.conv = conv1d(n_in, n_in, ks, padding=ks // 2, bias=False)

        self.gamma = nn.Parameter(tensor([0.]))

        self.sym = sym
        self.n_in = n_in

    def forward(self, x):
        if self.sym:
            # symmetry hack by https://github.com/mgrankin
            c = self.conv.weight.view(self.n_in, self.n_in)
            c = (c + c.t()) / 2
            self.conv.weight = c.view(self.n_in, self.n_in, 1)

        size = x.size()
        x = x.view(*size[:2], -1)  # (C,N)

        # changed the order of multiplication to avoid O(N^2) complexity
        # (x*xT)*(W*x) instead of (x*(xT*(W*x)))

        convx = self.conv(x)  # (C,C) * (C,N) = (C,N)   => O(NC^2)
        xxT = torch.bmm(x, x.permute(0, 2, 1).contiguous())  # (C,N) * (N,C) = (C,C)   => O(NC^2)

        o = torch.bmm(xxT, convx)  # (C,C) * (C,N) = (C,N)   => O(NC^2)

        o = self.gamma * o + x

        return o.view(*size).contiguous()


__all__ = ['MXResNet', 'mxresnet18', 'mxresnet34', 'mxresnet50', 'mxresnet101', 'mxresnet152']

# or: ELU+init (a=0.54; gain=1.55)
act_fn = Mish()  # nn.ReLU(inplace=True)


class Flatten(Module):
    def forward(self, x): return x.view(x.size(0), -1)


def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d, nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)


def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks // 2, bias=bias)


def noop(x): return x


def conv_layer(ni, nf, ks=3, stride=1, zero_bn=False, act=True):
    bn = nn.BatchNorm2d(nf)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv(ni, nf, ks, stride=stride), bn]
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)


class ResBlock(Module):
    def __init__(self, expansion, ni, nh, stride=1, sa=False, sym=False):
        nf, ni = nh * expansion, ni * expansion
        layers = [conv_layer(ni, nh, 3, stride=stride),
                  conv_layer(nh, nf, 3, zero_bn=True, act=False)
                  ] if expansion == 1 else [
            conv_layer(ni, nh, 1),
            conv_layer(nh, nh, 3, stride=stride),
            conv_layer(nh, nf, 1, zero_bn=True, act=False)
        ]
        self.sa = SimpleSelfAttention(nf, ks=1, sym=sym) if sa else noop
        self.convs = nn.Sequential(*layers)
        # TODO: check whether act=True works better
        self.idconv = noop if ni == nf else conv_layer(ni, nf, 1, act=False)
        self.pool = noop if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x): return act_fn(self.sa(self.convs(x)) + self.idconv(self.pool(x)))


def filt_sz(recep): return min(64, 2 ** math.floor(math.log2(recep * 0.75)))


class MXResNet(nn.Sequential):
    def __init__(self, expansion, layers, c_in=3, c_out=1000, sa=False, sym=False):
        stem = []
        sizes = [c_in, 32, 64, 64]  # modified per Grankin
        for i in range(3):
            stem.append(conv_layer(sizes[i], sizes[i + 1], stride=2 if i == 0 else 1))
            # nf = filt_sz(c_in*9)
            # stem.append(conv_layer(c_in, nf, stride=2 if i==1 else 1))
            # c_in = nf

        block_szs = [64 // expansion, 64, 128, 256, 512]
        blocks = [self._make_layer(expansion, block_szs[i], block_szs[i + 1], l, 1 if i == 0 else 2,
                                   sa=sa if i in [len(layers) - 4] else False, sym=sym)
                  for i, l in enumerate(layers)]
        super().__init__(
            *stem,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *blocks,
            nn.AdaptiveAvgPool2d(1), Flatten(),
            nn.Linear(block_szs[-1] * expansion, c_out),
        )
        init_cnn(self)

    def _make_layer(self, expansion, ni, nf, blocks, stride, sa=False, sym=False):
        return nn.Sequential(
            *[ResBlock(expansion, ni if i == 0 else nf, nf, stride if i == 0 else 1, sa if i in [blocks - 1] else False,
                       sym)
              for i in range(blocks)])


def mxresnet(expansion, n_layers, name, pretrained=False, **kwargs):
    model = MXResNet(expansion, n_layers, **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls[name]))
        print("No pretrained yet for MXResNet")
    return model


me = sys.modules[__name__]
for n, e, l in [
    [18, 1, [2, 2, 2, 2]],
    [34, 1, [3, 4, 6, 3]],
    [50, 4, [3, 4, 6, 3]],
    [101, 4, [3, 4, 23, 3]],
    [152, 4, [3, 8, 36, 3]],
]:
    name = f'mxresnet{n}'
    setattr(me, name, partial(mxresnet, expansion=e, n_layers=l, name=name))


class SteelEvalDataSet(Dataset):
    def __init__(self, data_df, test=True, image_position=0, transform=None):
        assert image_position in range(7)
        self.crop_idx = START_POS[image_position]
        self.imlist = []
        if test:
            for i in range(0, len(data_df), 4):
                self.imlist.append(os.path.join(TEST_IMAGES_PATH, data_df.loc[i, 'ImageId_ClassId'].split('_')[0]))
        else:
            for i in range(0, len(data_df), 4):
                self.imlist.append(os.path.join(TRAIN_IMAGES_PATH, data_df.loc[i, 'ImageId_ClassId'].split('_')[0]))
        self.transform = transform

    def __len__(self):
        return len(self.imlist)

    def __getitem__(self, index):
        img = Image.open(self.imlist[index])
        img = img.crop((self.crop_idx, 0, self.crop_idx + 256, 256))

        if self.transform:
            img = self.transform(img)

        return img.half(), self.imlist[index]


def get_dataloader(data_df, test=True, image_position=0, flip_p=0):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=flip_p),
        transforms.ToTensor(),
        transforms.Normalize(*imagenet_stats)
    ])

    test_dataset = SteelEvalDataSet(data_df=data_df, test=test, image_position=image_position, transform=transform)

    test_loader = DataLoader(test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=config.num_workers,
                             pin_memory=True,
                             drop_last=False)

    return test_loader


def UnetMxResnet(encoder='mxresnet18', n_classes=5, img_size=(256, 256),
                 blur=False, blur_final=True, self_attention=False, y_range=None,
                 last_cross=True, bottle=False):
    assert encoder in ['mxresnet18', 'mxresnet34', 'mxresnet50', 'mxresnet101', 'mxresnet152']
    MXResnet = globals()[encoder]()
    MXResnetBase = nn.Sequential(*[i for i in MXResnet.children()][:-3])
    Unet = DynamicUnet(encoder=MXResnetBase, n_classes=n_classes, img_size=img_size,
                       blur=blur, blur_final=blur_final, self_attention=self_attention, y_range=y_range,
                       last_cross=last_cross, bottle=bottle)
    return Unet


def load_weights(net, weight_path):
    net.load_state_dict(torch.load(weight_path)['model'])
    net.eval()
    return net


def stitch_preds(preds):
    fullpred = torch.zeros(preds[0].shape[0], preds[0].shape[1], 256, 1600, dtype=torch.half, device='cuda:0')
    for pos, pred in enumerate(preds):
        fullpred[:,:,:,START_POS[pos]:START_POS[pos]+256] += pred
    for pos in START_POS[1:]:
        fullpred[:,:,:,pos:pos+32] /= 2.
    return fullpred


def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


if __name__ == '__main__':
    jit_path = config.model_save_path().replace('.pth', '_jit.pth')
    if os.path.exists(jit_path):
        print("Loading jit model")
        net = torch.jit.load(jit_path)
    else:
        print("Loading defined model")
        net = UnetMxResnet(encoder=config.unet_encoder, n_classes=config.num_classes,
                       img_size=(config.imsize, config.imsize),
                       blur=config.unet_blur, blur_final=config.unet_blur_final,
                       self_attention=config.unet_self_attention,
                       y_range=config.unet_y_range, last_cross=config.unet_last_cross, bottle=config.unet_bottle)

    net = net.cuda()

    net = load_weights(net, config.model_save_path())

    net = net.half()
    net.eval()
    
    subm_df = pd.read_csv('./data/sample_submission.csv')
    loaders = [get_dataloader(data_df=subm_df, image_position=pos) for pos in range(7)]

    subm_idx = 0
    with torch.no_grad():
        for batches in tqdm(zip(*loaders), total=len(loaders[1])):
            preds = [net(b[0].cuda()) for b in batches]
            fpreds = stitch_preds(preds)

            for img in range(fpreds.shape[0]):
                for def_idx in range(4):
                    pred, num = post_process(fpreds[img,def_idx,:,:].detach().cpu().float().numpy(), config.best_threshold, config.min_size)
                    rle = mask2rle(pred)
                    subm_df.loc[subm_idx, 'EncodedPixels'] = rle
                    subm_idx+=1
    print(subm_idx)
    subm_df.to_csv('./subm/submission.csv', index=False)
