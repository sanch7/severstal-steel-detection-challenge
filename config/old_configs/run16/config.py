import os
from easydict import EasyDict
import getpass

config = EasyDict()

# experiment details
config.exp_name = "run16"
config.exp_type = "segmenter"
config.tboard = True
config.preload_data = True
config.desc = "deeplab drn"
config.split_csv = "./data/split2.csv"
config.gpu = None
config.fp16 = True
config.debug_run = False
config.random_seed = 42
config.wandb = True

# model framework
config.batch_size = 32
username = getpass.getuser()
if username == 'litemax2':
	config.batch_size = 32
config.epochs = 64
config.imsize = 256
config.load_valid_crops = True
config.load_train_crops = False
config.one_hot_labels = True
config.num_workers = os.cpu_count()

# archetecture details
config.model_name = "deeplab"
config.unet_encoder = "mxresnet34"
config.num_classes = 4
config.unet_blur = False
config.unet_blur_final = True
config.unet_self_attention = False
config.unet_y_range = None
config.unet_last_cross = True
config.unet_bottle = False

config.deeplab_backbone='drn'
config.deeplab_sync_bn = True
config.deeplab_freeze_bn = False

# training details
config.loss_dict = {"FocalLoss": {'weight': 0.15, 'mag_scale': 1.0, 'alpha': 0.8, 'gamma': 2},
				"TverskyLoss": {'weight': 0.1, 'mag_scale': 1.0}, "DiceBCELoss": {'weight': 0.15, 'mag_scale': 1.0},
				"BiTemperedLoss": {'weight': 0.6, 'mag_scale': 20.0, 't1': 0.8, 't2': 1.3, 'label_smoothing': 0.2}}
config.optimizer = "ranger"
config.lr = 1e-3
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
