import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# 基础和路径设置 (BASE & PATHS)
# -----------------------------------------------------------------------------
_C.BASE = [''] # 用于继承其他yaml文件
_C.OUTPUT = './training_results' # 总输出目录
_C.TAG = 'default' # 实验标签，用于区分不同的实验运行
_C.SEED = 42 # 随机种子，确保实验可复现

# -----------------------------------------------------------------------------
# 系统设置 (SYSTEM)
# -----------------------------------------------------------------------------
_C.SYSTEM = CN()
_C.SYSTEM.DEVICE = 'cuda:0'
_C.SYSTEM.NUM_WORKERS = 8
_C.SYSTEM.PIN_MEMORY = True

# -----------------------------------------------------------------------------
# 数据设置 (DATA)
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.DATA_PATH = './datasets/CSV/shale_256.csv'
_C.DATA.CLASS_NAMES = ['OM', 'OP', 'IOP']
_C.DATA.IMG_SIZE = 256
_C.DATA.BATCH_SIZE = 8
_C.DATA.VALID_BATCH_SIZE = 8 # 为验证集单独设置batch_size
_C.DATA.TRAIN_RATIO = 0.7
_C.DATA.VAL_RATIO = 0.2
_C.DATA.SPLIT_FLAG = False # 是否在每次运行时重新划分数据集
_C.DATA.NUM_SMALL_DATA = None # 若非None，则使用小数据集进行快速调试

# 数据预处理的均值和标准差
_C.DATA.TRANSFORMS = CN()
_C.DATA.TRANSFORMS.MEAN = (0.485, 0.456, 0.406)
_C.DATA.TRANSFORMS.STD = (0.229, 0.224, 0.225)

# 数据增强 (AUGMENTATION) 设置 (这是新增的关键部分)
_C.DATA.AUG = CN()
_C.DATA.AUG.ENABLE = False # 是否启用数据增强(在data_split中)
_C.DATA.AUG.ROOT_PATH = "./datasets" # 数据增强文件保存的根目录
_C.DATA.AUG.AUG_TIMES = 60 # 增强倍数
_C.DATA.AUG.VERSION = "v2" # 增强版本号，用于区分不同增强结果

# -----------------------------------------------------------------------------
# 模型设置 (MODEL)
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'swin'
_C.MODEL.NAME = 'unet'
_C.MODEL.RESUME = None # 断点续传的权重路径
_C.MODEL.BASE_CHANNELS = 32 # U-Net系列模型的基础通道数
_C.MODEL.DROPOUT_P = 0.2

# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.PRETRAIN_CKPT = './pretrained_ckpt/swin_tiny_patch4_window7_224.pth'
# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.DECODER_DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 8
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
_C.MODEL.SWIN.FINAL_UPSAMPLE= "expand_first"

# -----------------------------------------------------------------------------
# 训练设置 (TRAIN)
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 200
_C.TRAIN.EVAL_INTERVAL = 1
_C.TRAIN.AMP = True # 是否使用混合精度训练
_C.TRAIN.EARLY_STOP_PATIENCE = 30 # 早停的patience

# 优化器设置
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'AdamW'
_C.TRAIN.OPTIMIZER.BASE_LR = 5e-4
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 0.0
_C.TRAIN.OPTIMIZER.ADAMW_BETAS = (0.95, 0.999)
_C.TRAIN.OPTIMIZER.SGD_MOMENTUM = 0.9

# 学习率调度器设置
_C.TRAIN.SCHEDULER = CN()
_C.TRAIN.SCHEDULER.NAME = 'CosineAnnealingLR'
_C.TRAIN.SCHEDULER.WARMUP_EPOCHS = 5
_C.TRAIN.SCHEDULER.COSINE_T_MAX = 60 # CosineAnnealingLR的T_max
_C.TRAIN.SCHEDULER.COSINE_ETA_MIN = 1e-8 # CosineAnnealingLR的eta_min
_C.TRAIN.SCHEDULER.PLATEAU_FACTOR = 0.1 # ReduceLROnPlateau的factor
_C.TRAIN.SCHEDULER.PLATEAU_PATIENCE = 5 # ReduceLROnPlateau的patience

# 损失函数设置
_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.NAME = 'DiceLoss'
_C.TRAIN.LOSS.ELN_LOSS = True # 是否使用Elastic Net正则化
_C.TRAIN.LOSS.L1_LAMBDA = 0.0001
_C.TRAIN.LOSS.L2_LAMBDA = 0.0001

# -----------------------------------------------------------------------------
# 日志和保存设置 (LOG & SAVE)
# -----------------------------------------------------------------------------
_C.LOG = CN()
_C.LOG.USE_TENSORBOARD = True
_C.LOG.USE_SWANLAB = True
_C.LOG.SAVE_FLAG = True
# 新增: 控制是否在运行时交互式修改参数的标志
_C.LOG.MDF_PARAMS = False 

# =============================================================================
# Helper Functions 
# =============================================================================
def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.safe_load(f)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print(f'=> merge config from {cfg_file}')
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    if args.cfg:
        _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)
    return config