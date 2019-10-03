# the code mostly from https://github.com/sdoria/SimpleSelfAttention
#based on @grankin FastAI forum script
#updated by lessw2020 to use Mish XResNet

# adapted from https://github.com/fastai/fastai/blob/master/examples/train_imagenette.py
# changed per gpu bs for bs_rat

import os, sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastprogress import fastprogress
from torchvision.models import *
import wandb
# from wandb.fastai import WandbCallback
from utils.wandbcallback import WandbCallback

from functools import partial

from models import model_list
from modules.losses import SteelLoss
from modules.metrics import accuracy, dice
from utils.databunch import get_data_bunch
from utils.callbacks import SaveBestModel

#from radam import *
#from novograd import *
#from rangervar import *
from modules.ranger import *
#from ralamb import *
#from over9000 import *
#from lookahead import *
#from adams import *
#from rangernovo import *
#from rangerlars import *

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

from config.config import config


def fit_with_annealing(learn:Learner, num_epoch:int, lr:float=defaults.lr, annealing_start:float=0.7,
                    callbacks:list=None)->None:
    n = len(learn.data.train_dl)
    anneal_start = int(n*num_epoch*annealing_start)
    phase0 = TrainingPhase(anneal_start).schedule_hp('lr', lr)
    phase1 = TrainingPhase(n*num_epoch - anneal_start).schedule_hp('lr', lr, anneal=annealing_cos)
    phases = [phase0, phase1]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    learn.fit(num_epoch, callbacks=callbacks)

def train(config, get_learn=False):

    bs_one_gpu = config.batch_size
    gpu = setup_distrib(config.gpu)
    if gpu is None: config.batch_size *= torch.cuda.device_count()
    
    opt = config.optimizer
    mom = config.mom
    alpha = config.alpha
    eps = config.eps
    if   opt=='adam' : opt_func = partial(optim.Adam, betas=(mom,alpha), eps=eps)
    elif opt=='radam' : opt_func = partial(RAdam, betas=(mom,alpha), eps=eps)
    elif opt=='novograd' : opt_func = partial(Novograd, betas=(mom,alpha), eps=eps)
    elif opt=='rms'  : opt_func = partial(optim.RMSprop, alpha=alpha, eps=eps)
    elif opt=='sgd'  : opt_func = partial(optim.SGD, momentum=mom)
    elif opt=='rangervar'  : opt_func = partial(RangerVar,  betas=(mom,alpha), eps=eps)
    elif opt=='ranger'  : opt_func = partial(Ranger,  betas=(mom,alpha), eps=eps)
    elif opt=='ralamb'  : opt_func = partial(Ralamb,  betas=(mom,alpha), eps=eps)
    elif opt=='over9000'  : opt_func = partial(Over9000,  k=12, betas=(mom,alpha), eps=eps)
    elif opt=='lookahead'  : opt_func = partial(LookaheadAdam, betas=(mom,alpha), eps=eps)
    elif opt=='Adams': opt_func=partial(Adams)
    elif opt=='rangernovo': opt_func=partial(RangerNovo)
    elif opt=='rangerlars':opt_func=partial(RangerLars)

    split_df = pd.read_csv(config.split_csv)
    if config.debug_run: split_df = split_df.loc[:100]
    data = get_data_bunch(split_df, config=config)

    bs_rat = config.batch_size/bs_one_gpu   #originally bs/256
    if gpu is not None: bs_rat *= max(num_distrib(), 1)
    if not gpu: print(f'lr: {config.lr}; eff_lr: {config.lr*bs_rat}; size: {config.imsize}; alpha: {alpha}; mom: {mom}; eps: {eps}')
    config.lr *= bs_rat

    Net = getattr(model_list, config.model_name)
    net = Net(encoder=config.unet_encoder, n_classes=config.num_classes, img_size=(config.imsize, config.imsize),
        blur=config.unet_blur, blur_final=config.unet_blur_final, self_attention=config.unet_self_attention,
        y_range=config.unet_y_range, last_cross=config.unet_last_cross, bottle=config.unet_bottle)
    
    log_cb = partial(CSVLogger,filename=config.log_file)

    loss_func = SteelLoss(loss_dict=config.loss_dict)

    ckpt_dir = "./model_weights/{}/".format(config.exp_name)
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    callback_fns=[WandbCallback] if (config.wandb and not get_learn) else []

    learn = (Learner(data, net, wd=config.weight_decay, opt_func=opt_func,
             metrics=[accuracy,dice],
             bn_wd=False, true_wd=True,
             loss_func = loss_func,
             # loss_func = LabelSmoothingCrossEntropy(),
             callback_fns=callback_fns,
             model_dir=ckpt_dir)
            )

    if config.wandb and not get_learn:
        wandb.init(project="Severstal Steel Defect", name=config.exp_name, config=config,
                    notes=config.desc)

    print("Learn path: ", learn.path)
    n = len(learn.data.train_dl)
    ann_start2= int(n*config.epochs*config.ann_start)
    print(ann_start2," annealing start")
    
    if config.dump: print(learn.model); exit()
    if config.mixup: learn = learn.mixup(alpha=config.mixup)
    if config.fp16: learn = learn.to_fp16(dynamic=True)
    if gpu is None:       learn.to_parallel()
    elif num_distrib()>1: learn.to_distributed(gpu) # Requires `-m fastai.launch`

    if get_learn:
        return learn
    
    if config.lrfinder:
        # run learning rate finder
        learn.lr_find(wd=config.weight_decay)
        learn.recorder.plot()
        config.lr = float(input())

    best_save_cb = SaveBestModel(learn, ckpt_name='best_dice')
    if config.sched_type == 'one_cycle': 
        learn.fit_one_cycle(config.epochs, config.lr, div_factor=10, pct_start=0.3,
                callbacks=[best_save_cb])
    elif config.sched_type == 'flat_and_anneal': 
        fit_with_annealing(learn, config.epochs, config.lr, config.ann_start,
                callbacks=[best_save_cb])
    
    learn.save('basic_model')

    return learn.recorder.metrics[-1][0]

def main():
    run = 1
    acc = np.array([train(config) for i in range(run)])

    print(acc)
    print(np.mean(acc))
    print(np.std(acc))

if __name__ == '__main__':
    main()
