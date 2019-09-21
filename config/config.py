import os
from easydict import EasyDict
import getpass

config = EasyDict()

# experiment details
config.exp_name = "run2"
config.tboard = True
config.preload_data = True
config.desc = "Initial run"
config.split_csv = "./data/split.csv"
config.gpu = None
config.fp16 = True
config.debug_run = False
config.model_save_path = lambda : './model_weights/{}/best_dice.pth'.format(config.exp_name)

# model framework
config.batch_size = 36
username = getpass.getuser()
if username == 'litemax2':
	config.batch_size = 32
config.epochs = 64
config.imsize = 256
config.load_valid_crops = True
config.load_train_crops = False
config.num_workers = os.cpu_count()

# archetecture details
config.model_name = "UnetMxResnet"
config.unet_encoder = "mxresnet18"
config.num_classes = 4
config.unet_blur = False
config.unet_blur_final = True
config.unet_self_attention = False
config.unet_y_range = None
config.unet_last_cross = True
config.unet_bottle = False

# training details
config.loss = "focal"
config.focal_alpha = 0.8
config.focal_gamma = 2
config.optimizer = "ranger"
config.lr = 1e-3
config.weight_decay = 1e-2
config.alpha = 0.99
config.mom = 0.9 # Momentum
config.eps = 1e-6
config.mixup = 0.
config.sched_type = "flat_and_anneal" # LR schedule type
config.ann_start = -1.0 # Annealing start

# misc
config.lrfinder = 0 # Run learning rate finder
config.dump = 0 # Print model; don't train"
config.log_file = "./logs/{}".format(config.exp_name) # Log file name
