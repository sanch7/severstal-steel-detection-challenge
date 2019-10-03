import os
from easydict import EasyDict
import getpass

config = EasyDict()

# experiment details
config.exp_name = "run8"
config.tboard = True
config.preload_data = True
config.desc = "3 combination loss, mxresnet14, unet sa, lr=0.5 with unet blur and bottle"
config.split_csv = "./data/split.csv"
config.gpu = None
config.fp16 = True
config.debug_run = False
config.random_seed = 42
config.wandb = True

# model framework
config.batch_size = 36
username = getpass.getuser()
if username == 'litemax2':
	config.batch_size = 28
config.epochs = 32
config.imsize = 256
config.load_valid_crops = True
config.load_train_crops = False
config.one_hot_labels = True
config.num_workers = os.cpu_count()

# archetecture details
config.model_name = "UnetMxResnet"
config.unet_encoder = "mxresnet34"
config.num_classes = 4
config.unet_blur = True
config.unet_blur_final = True
config.unet_self_attention = True
config.unet_y_range = None
config.unet_last_cross = True
config.unet_bottle = True

# training details
config.loss_dict = {"FocalLoss": {'weight': 0.4, 'alpha': 0.8, 'gamma': 2},
				"TverskyLoss": {'weight': 0.2}, "DiceBCELoss": {'weight': 0.4}}
config.optimizer = "ranger"
config.lr = 5e-1
config.weight_decay = 1e-2
config.alpha = 0.99
config.mom = 0.9 # Momentum
config.eps = 1e-6
config.mixup = 0.
config.sched_type = "one_cycle" # LR schedule type
config.ann_start = -1.0 # Annealing start
config.oversample = False
config.train_duplicate = 3 # Duplicate train items so less validation

# misc
config.lrfinder = 0 # Run learning rate finder
config.dump = 0 # Print model; don't train"
config.log_file = "./logs/{}".format(config.exp_name) # Log file name

if config.debug_run:
	config.wandb = False
