import os
from easydict import EasyDict

config = EasyDict()

# experiment details
config.exp_name = "run1"
config.tboard = True
config.preload_data = True
config.desc = "Initial run"

# model framework
config.model_name = "mxresnet18"
config.batch_size = 8
config.epochs = 100
config.imsize = 256
config.fp16 = False
config.num_workers = os.cpu_count()

# training details
config.loss = "focal"
config.focal_gamma = 0.5
config.optimizer = "ranger"
config.lr = 1e-3
config.reduce_lr_plateau = True
config.lr_scale = 0.1
config.lr_patience = 16
config.final_lr = 1e-1
config.weight_decay = 0.001
