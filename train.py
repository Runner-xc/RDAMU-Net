import torch
import datetime
from torch.utils.data import DataLoader
from utils.my_data import SEM_DATA
from utils import data_split
from utils.writing_logs import writing_logs
import argparse
import os
from torch.optim import Adam, SGD, RMSprop, AdamW
import time
from model.u2net import u2net_full_config, u2net_lite_config
from model.rdam_unet import *
from model.unet_3plus import UNet_3Plus
from model.swin_unet import SwinUnet
from model.res50_unet import RES50_UNet
from monai.networks.nets import SegResNet, SwinUNETR, UNet
from model.vim_unet import VMUNet
from model.att_unet import Attention_UNet
from model.unet_plusplus import UnetPlusPlus
from tabulate import tabulate
from utils.train_and_eval import *
from utils.model_initial import *
from utils import param_modification
from utils import write_experiment_log
from utils.loss_fn import *
from utils.metrics import Evaluate_Metric
from torch.utils.tensorboard import SummaryWriter
import utils.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import LambdaLR
import math
from typing import Union, List
from utils.run_tensorboard import run_tensorboard
from torchinfo import summary
from config import get_config

# ---------------------------- Constants and Presets ----------------------------
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ConfigPreset:
    """数据预处理配置预设"""
    @staticmethod
    def train_preset(base_size, crop_size=256, hflip_prob=0.5):
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    @staticmethod
    def eval_preset(base_size):
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

# ---------------------------- Core Components ----------------------------
class TrainingComponents:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        
    def get_model(self):
        model_map = {
            "unet": UNet(spatial_dims=2,in_channels=3, out_channels=4, channels=(16*2, 32*2, 64*2, 128*2, 256*2), strides=(2, 2, 2, 2)),
            "a_unet": A_UNet(in_channels=3, num_classes=4, base_channels=32, p=args.dropout_p),
            "m_unet": M_UNet(in_channels=3, num_classes=4, base_channels=32, p=args.dropout_p, w=args.w),
            "rdam_unet": RDAM_UNet(in_channels=3, num_classes=4, base_channels=32, p=args.dropout_p, w=args.w),

            "unet_plusplus": UnetPlusPlus(in_channels=3, num_classes=4, deep_supervision=False),
            "unet_3plus": UNet_3Plus(in_channels=3, num_classes=4),
            # "swin_unet": SwinUnet(config=args.cfg,  num_classes=4),
            "res50_unet": RES50_UNet(in_channels=3, num_classes=4),
            "att_unet": Attention_UNet(in_channels=3, num_classes=4, p=args.dropout_p, base_channels=32),
            "vim_unet": VMUNet(input_channels=3, num_classes=4, dropout_r=args.dropout_p),
        }
        model = model_map.get(self.args.model)
        if not model:
            raise ValueError(f"Invalid model name: {self.args.model}")
        kaiming_initial(model)
        return model.to(self.device)

    def get_optimizer(self, model):
        """优化器"""
        optim_map = {
            'AdamW': lambda: AdamW(model.parameters(), self.args.lr, weight_decay=self.args.wd),
            'SGD': lambda: SGD(model.parameters(), self.args.lr, momentum=0.9, weight_decay=self.args.wd),
            'RMSprop': lambda: RMSprop(model.parameters(), self.args.lr, alpha=0.9, eps=1e-8, weight_decay=self.args.wd)
        }
        return optim_map.get(self.args.optimizer, optim_map['AdamW'])()

    def get_scheduler(self, optimizer, train_loader):
        """学习率调度器"""
        if self.args.scheduler == 'CosineAnnealingLR':
            num_batches = len(train_loader)
            warmup_steps = self.args.warmup_epochs * num_batches
            Tmax_steps = self.args.Tmax * num_batches
            lr_initial = optimizer.param_groups[0]['lr']
            
            return LambdaLR(optimizer, lambda step: (
                (step / warmup_steps) if step < warmup_steps
                else (self.args.eta_min + (lr_initial - self.args.eta_min) * 
                    (1 + math.cos(math.pi * (step - warmup_steps) / Tmax_steps)) / 2) / lr_initial
            ), last_epoch=-1)
            
        elif self.args.scheduler == 'ReduceLROnPlateau':
            return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        else:
            raise ValueError(f"Unsupported scheduler: {self.args.scheduler}")

    def get_loss_fn(self):
        """损失函数"""
        loss_map = {
            'CrossEntropyLoss': CrossEntropyLoss(),
            'DiceLoss': diceloss(),
            'FocalLoss': Focal_Loss(),
            'WDiceLoss': WDiceLoss(),
            'DWDLoss': DWDLoss(),
            'IoULoss': IOULoss(),
            'dice_hd': AdaptiveSegLoss(4)
        }
        return loss_map.get(self.args.loss_fn)

    def get_writer(self, detailed_time_str):
        # 日志保存路径
        save_logs_path = f"{self.args.log_path}/{self.args.model}/L_{self.args.loss_fn}--S_{self.args.scheduler}"
        
        if not os.path.exists(save_logs_path):
            os.makedirs(save_logs_path)
        
        if self.args.elnloss:
            log_path = f'{save_logs_path}/optim_{self.args.optimizer}-lr_{self.args.lr}-l1_{self.args.l1_lambda}-l2_{self.args.l2_lambda}/{detailed_time_str}'
            return SummaryWriter(log_path), log_path
        else:
            log_path = f'{save_logs_path}/optim_{self.args.optimizer}-lr_{self.args.lr}-wd_{self.args.wd}/{detailed_time_str}'
            return SummaryWriter(log_path), log_path
        

# ---------------------------- Data Management ----------------------------
class DataManager:
    def __init__(self, args):
        self.args = args
        self.base_transform = ConfigPreset()

    def load_datasets(self):
        """加载数据集"""
        if self.args.num_small_data:
            train, val, test = data_split.small_data_split_to_train_val_test(
                self.args.data_path, self.args.num_small_data, self.args.split_flag, self.args.train_ratio, self.args.val_ratio,self.args.data_root_path)
        else:
            train, val, test = data_split.data_split_to_train_val_test(
                self.args.data_path, self.args.split_flag, self.args.train_ratio, self.args.val_ratio, 
                self.args.data_root_path)
        T_transforms = T.Resize((224, 224)) if 'swin_unet' in args.model else self.base_transform.train_preset(256)
        V_transforms = T.Resize((224, 224)) if 'swin_unet' in args.model else self.base_transform.eval_preset(256)
        train_set = SEM_DATA(train,  transforms=T_transforms)
        val_set = SEM_DATA(val, transforms=V_transforms)
        return train_set, val_set

    def get_dataloaders(self, train_set, val_set):
        num_workers = min(os.cpu_count(), self.args.batch_size if self.args.batch_size > 1 else 0, 8)
        train_loader = DataLoader(train_set, self.args.batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, 8, shuffle=False, 
                              num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader

# ---------------------------- Training Logic ----------------------------
class TrainingManager:
    def __init__(self, args, model, optimizer, scheduler, loss_fn, device):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.metrics = Evaluate_Metric()
        self.scaler = torch.amp.GradScaler() if args.amp else None
     
    def _train_epoch(self, epoch, dataloader):
        """
        训练单个epoch
        """
        start_time = time.time()
        components_dict = {"model"      :self.model, 
                           "optimizer"  :self.optimizer, 
                           "epoch"      :epoch, 
                           "dataloader" :dataloader, 
                           "device"     :self.device, 
                           "loss_fn"    :self.loss_fn, 
                           "scaler"     :self.scaler,
                           "metrics"    :self.metrics,
                           "scheduler"  :self.scheduler,
                           "elnloss"    :self.args.elnloss, 
                           "l1_lambda"  :self.args.l1_lambda,
                           "l2_lambda"  :self.args.l2_lambda}
        # results = train_loss, T_OM_loss, T_OP_loss, T_IOP_loss, T_Metric_list
        results = train_one_epochv2(components_dict)
        loss = [x / len(dataloader) for x in results[:-1]]
        self.train_mean_loss = loss[3]
        results_dict = {"Loss"      : loss, 
                        "Recall"    :results[-1][0], 
                        "Precision" :results[-1][1], 
                        "Dice"      :results[-1][2], 
                        "F1_scores" :results[-1][3], 
                        "mIoU"      :results[-1][4], 
                        "Accuracy"  :results[-1][5]}
        names = ["Loss", "Recall", "Precision", "Dice", "F1_scores", "mIoU", "Accuracy"]
        train_metrics = {f"{name}" : results_dict[name] for name in names}

        end_time = time.time()
        train_cost_time = end_time - start_time
        print(
              f"train_OM_loss: {loss[0]:.3f}\n"
              f"train_OP_loss: {loss[1]:.3f}\n"
              f"train_IOP_loss: {loss[2]:.3f}\n"
              f"train_mean_loss: {loss[3]:.3f}\n"
              f"train_cost_time: {train_cost_time:.2f}s\n")
        return train_metrics

    def _validate_epoch(self,val_dataloader):
        """
        验证单个epoch
        """
        start_time = time.time()
        # results = OM_loss,OP_loss,IOP_loss, mean_loss, Metric_list
        results = evaluate(self.model, self.device, val_dataloader, self.loss_fn, self.metrics)
        loss = [x / len(val_dataloader) for x in results[:-1]]
        self.val_mean_loss = loss[3]
        results_dict = {"Loss"      : loss, 
                        "Recall"    : results[-1][0], 
                        "Precision" : results[-1][1], 
                        "Dice"      : results[-1][2], 
                        "F1_scores" : results[-1][3], 
                        "mIoU"      : results[-1][4], 
                        "Accuracy"  : results[-1][5]}
        names = ["Loss", "Recall", "Precision", "Dice", "F1_scores", "mIoU", "Accuracy"]
        self.val_metrics = {f"{name}" : results_dict[name] for name in names}
        # 获取当前学习率
        current_lr = self.scheduler.get_last_lr()[0]
        # 结束时间
        end_time = time.time()
        self.val_cost_time = end_time - start_time
        print(
            f"val_OM_loss: {loss[0]:.3f}\n"
            f"val_OP_loss: {loss[1]:.3f}\n"
            f"val_IOP_loss: {loss[2]:.3f}\n"
            f"val_mean_loss: {loss[3]:.3f}\n"
            f"val_cost_time: {self.val_cost_time:.2f}s\n")
        print(f"Current learning rate: {current_lr}\n")
        return self.val_metrics, loss[3]
    
    def run_logging(self,writer, train_metrics, val_metrics, epoch):
        if self.args.tb:
            writing_logs(writer, train_metrics, val_metrics, epoch)
    
    def save_metrics(self, args, epoch, end_epoch, best_epoch):
        metrics_table_header    = ['Metrics_Name', 'Mean', 'OM', 'OP', 'IOP']
        metrics_table_left      = ['Dice', 'Recall', 'Precision', 'F1_scores', 'mIoU', 'Accuracy']
        epoch_s                 = f"✈✈✈✈✈ epoch : {epoch + 1} / {end_epoch} ✈✈✈✈✈✈\n"
        model_s                 = f"model : {args.model} \n"
        lr_s                    = f"lr : {args.lr} \n"
        wd_s                    = f"wd : {args.wd} \n"  #####
        dropout_s               = f"dropout : {args.dropout_p} \n"
        l1_lambda               = f"l1_lambda : {args.l1_lambda} \n"
        l2_lambda               = f"l2_lambda : {args.l2_lambda} \n"
        scheduler_s             = f"scheduler : {args.scheduler} \n"
        loss_fn_s               = f"loss_fn : {args.loss_fn} \n"
        best_epoch_s            = f"best_epoch : {best_epoch} \n"
        time_s                  = f"time : {datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')} \n"
        cost_s                  = f"cost_time :{self.val_cost_time / 60:.2f}mins \n"
        
        metrics_dict    = {scores : self.val_metrics[scores] for scores in metrics_table_left}
        metrics_table   = [[metric_name,
                            metrics_dict[metric_name][-1],
                            metrics_dict[metric_name][0],
                            metrics_dict[metric_name][1],
                            metrics_dict[metric_name][2]
                        ]
                            for metric_name in metrics_table_left
                        ]
        table_s         = tabulate(metrics_table, headers=metrics_table_header, tablefmt='grid')
        train_loss_s    = f"train_loss : {self.train_mean_loss:.3f}  🍎🍎🍎\n"
        loss_s          = f"val_loss : {self.val_mean_loss:.3f}   🍎🍎🍎\n"

        # 记录每个epoch对应的train_loss、lr以及验证集各指标
        write_info      = epoch_s + model_s + lr_s + wd_s + dropout_s + l1_lambda + l2_lambda + loss_fn_s + scheduler_s + train_loss_s + loss_s + table_s + '\n' + best_epoch_s + cost_s + time_s
        print(write_info)

        # 保存结果
        save_scores_path = f'{args.save_scores_path}/{args.model}/L_{args.loss_fn}--S_{args.scheduler}'
        if args.elnloss:
            results_file = f"optim_{args.optimizer}-lr_{args.lr}-l1_{args.l1_lambda}-l2_{args.l2_lambda}/{detailed_time_str}.txt"
        else:
            results_file = f"optim_{args.optimizer}-lr_{args.lr}-wd_{args.wd}/{detailed_time_str}.txt"
        file_path = os.path.join(save_scores_path, results_file)

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        if args.save_flag:
            with open(file_path, "a") as f:
                f.write(write_info)
    
    def save_weights(self, args, epoch, best_mean_loss, best_epoch, model_info):
        if args.save_flag:
            # 保存best模型
            if args.elnloss:
                save_weights_path = f"{args.save_weight_path}/{args.model}/L_{args.loss_fn}--S_{args.scheduler}/optim_{args.optimizer}-lr_{args.lr}-l1_{args.l1_lambda}-l2_{args.l2_lambda}/{detailed_time_str}"  # 保存权重路径
            else:
                save_weights_path = f"{args.save_weight_path}/{args.model}/L_{args.loss_fn}--S_{args.scheduler}/optim_{args.optimizer}-lr_{args.lr}-wd_{args.wd}/{detailed_time_str}"
                
            if not os.path.exists(save_weights_path):
                os.makedirs(save_weights_path)

            save_file = {"model"        : self.model.state_dict(),
                        "optimizer"     : self.optimizer.state_dict(),
                        "Metrics"       : self.metrics.state_dict(),
                        "scheduler"     : self.scheduler.state_dict(),
                        "best_mean_loss": best_mean_loss,
                        "best_epoch"    : best_epoch,
                        "step"          : self.scheduler.last_epoch,
                        "model_info"    : model_info}
            
            # 保存当前最佳模型的权重
            best_model_path = f"{save_weights_path}/model_best_ep_{best_epoch}_w{args.w}.pth"
            torch.save(save_file, best_model_path)
            print(f"Best model saved at epoch {best_epoch} with mean loss {best_mean_loss}")
            # 删除之前保存的所有包含"model_best"的文件
            path_list = os.listdir(save_weights_path)
            for i in path_list:
                if "model_best" in i and i != f"model_best_ep_{best_epoch}_w{args.w}.pth":
                    os.remove(os.path.join(save_weights_path, i))
                    print(f"remove last best weight:{i}")
                    
            # only save latest 3 epoch weights
            if os.path.exists(f"{save_weights_path}/model_ep_{epoch-3}.pth"):
                os.remove(f"{save_weights_path}/model_ep_{epoch-3}.pth")
                
            if not os.path.exists(save_weights_path):
                os.makedirs(save_weights_path)
            torch.save(save_file, f"{save_weights_path}/model_ep_{epoch}.pth") 

# ---------------------------- Main Function ----------------------------
def main(args, detailed_time_str):
    # 设置随机种子
    set_seed(args.seed)

    # 初始化组件
    components = TrainingComponents(args)
    data_mgr = DataManager(args)
    
    # 加载数据
    train_set, val_set = data_mgr.load_datasets()
    train_loader, val_loader = data_mgr.get_dataloaders(train_set, val_set)
    
    # 初始化模型和训练组件
    model = components.get_model()
    model_info = str(summary(model))

    optimizer = components.get_optimizer(model)
    scheduler = components.get_scheduler(optimizer, train_loader)
    loss_fn = components.get_loss_fn()
    if args.save_flag:
        writer, log_path = components.get_writer(detailed_time_str)
    
    # 初始化训练管理器
    trainer = TrainingManager(args, model, optimizer, scheduler, loss_fn, components.device)

    best_mean_loss, current_miou = float('inf'), 0.0
    best_epoch = 0 
    patience = 0 
    current_mean_loss = float('inf')
    start_epoch = args.start_epoch
    """断点续传"""    
    if args.resume:
        torch.serialization.add_safe_globals([argparse.Namespace])
        checkpoint = torch.load(args.resume, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scheduler.last_epoch = checkpoint['step']
        best_mean_loss = checkpoint['best_mean_loss']
        start_epoch = checkpoint['best_epoch']
        best_epoch = checkpoint['best_epoch']
        print(f"Resume from epoch: {start_epoch}")

    # 训练循环
    for epoch in range(start_epoch, args.end_epoch):
        print(f"\n✈✈✈ Epoch {epoch+1}/{args.end_epoch} ✈✈✈")
        
        # 训练阶段
        train_metrics = trainer._train_epoch(epoch, train_loader)
        
        # 验证阶段
        if epoch % args.eval_interval == 0 or epoch == args.end_epoch - 1:
            val_metrics, val_mean_loss = trainer._validate_epoch(val_loader)
            
            # 记录日志
            trainer.run_logging(writer, train_metrics, val_metrics, epoch)
            if epoch == 5:
                    run_tensorboard(log_path)

            if best_mean_loss >= val_mean_loss:
                best_mean_loss = val_mean_loss
                best_epoch = epoch + 1
                
            # 保存指标
            trainer.save_metrics(args, epoch, args.end_epoch, best_epoch)
            # 保存权重
            trainer.save_weights(args, epoch, best_mean_loss, best_epoch, model_info)

            # 记录验证loss是否出现上升       
            if val_mean_loss < current_mean_loss:
                current_mean_loss = val_mean_loss 
                patience = 0   
            else:
                patience += 1
            # 早停检查
            if patience >= args.patience:
                print(f"Early stopping triggered after {args.patience} epochs without improvement!")
                break

    # 清理资源
    writer.close()
    print(f"Training completed in {time.time()-detailed_time_str:.2f} seconds")

def parse_args():
    """参数解析"""
    parser = argparse.ArgumentParser(description="SEM图像分割训练脚本")
    # 保存路径
    parser.add_argument('--data_path',          type=str, 
                        default="/mnt/e/VScode/WS-Hub/Linux-RDAMU_Net/RDAMU-Net/datasets/CSV/rock_sem_chged_256_a50_c80.csv", 
                        help="path of csv dataset")
    
    parser.add_argument('--data_root_path',  type=str,
                        default="/mnt/e/VScode/WS-Hub/Linux-RDAMU_Net/RDAMU-Net/datasets/CSV")
    
    # results
    parser.add_argument('--save_scores_path',   type=str, 
                        default='/mnt/e/VScode/WS-Hub/Linux-RDAMU_Net/RDAMU-Net/results/save_scores')
    
    parser.add_argument('--save_weight_path',   type=str,
                        default="/mnt/e/VScode/WS-Hub/Linux-RDAMU_Net/RDAMU-Net/results/save_weights")
    
    parser.add_argument('--log_path',  type=str,
                        default="/mnt/e/VScode/WS-Hub/Linux-RDAMU_Net/RDAMU-Net/results/logs")
    
    parser.add_argument('--modification_path', type=str,
                        default="/mnt/e/VScode/WS-Hub/Linux-RDAMU_Net/RDAMU-Net/results/modification_log")
    
    # 模型配置
    parser.add_argument('--model',              type=str, 
                        default="swin_unet", 
                        help=" unet, rdam_unet, a_unet, m_unet, unet_3plus, unet_plusplus, swin_unet, res50_unet, vim_unet\
                               att_unet, u2net_full, u2net_lite,")
    
    parser.add_argument('--loss_fn',            type=str, 
                        default='DiceLoss', 
                        help="'CrossEntropyLoss', 'FocalLoss', 'DiceLoss', 'WDiceLoss', 'DWDLoss', 'IoULoss', 'dice_hd'")
    
    parser.add_argument('--optimizer',          type=str, 
                        default='AdamW', 
                        help="'AdamW', 'SGD' or 'RMSprop'.")
    
    parser.add_argument('--scheduler',          type=str, 
                        default='CosineAnnealingLR', 
                        help="'CosineAnnealingLR', 'ReduceLROnPlateau'.")
    
    # 正则化
    parser.add_argument('--elnloss',        type=bool,  default=False)
    parser.add_argument('--l1_lambda',      type=float, default=0.001)
    parser.add_argument('--l2_lambda',      type=float, default=0.001)
    parser.add_argument('--dropout_p',      type=float, default=0.4  )
     
    parser.add_argument('--device',         type=str,   default='cuda:0')
    parser.add_argument('--resume',         type=str,   default=None,   help="the path of weight for resuming")
    parser.add_argument('--amp',            type=bool,  default=True,   help='use mixed precision training or not')
    
    # flag参数
    parser.add_argument('--tb',             type=bool,  default=True,   help='use tensorboard or not')   
    parser.add_argument('--save_flag',      type=bool,  default=True,   help='save weights or not')    
    parser.add_argument('--split_flag',     type=bool,  default=False,  help='split data or not')
    parser.add_argument('--change_params',  type=bool,  default=False,  help='change params or not')       
    
    # 训练参数

    parser.add_argument('--seed',           type=int,   default=42) 
    parser.add_argument('--train_ratio',    type=float, default=0.7) 
    parser.add_argument('--val_ratio',      type=float, default=0.1)
    parser.add_argument('--batch_size',     type=int,   default=8  ) 
    parser.add_argument('--w',              type=float, default=0.7  ) 
    parser.add_argument('--start_epoch',    type=int,   default=0,      help='start epoch')
    parser.add_argument('--end_epoch',      type=int,   default=200,    help='ending epoch')
    parser.add_argument('--warmup_epochs',  type=int,   default=10,     help='number of warmup epochs')


    # parser.add_argument('--cfg',            type=str,   default="/mnt/e/VScode/WS-Hub/Linux-RDAMU_Net/RDAMU-Net/cfg/swin_unet.yaml", help='Path to the .yaml config file.')
    parser.add_argument('--lr',             type=float, default=8e-4,   help='learning rate')
    parser.add_argument('--wd',             type=float, default=1e-6,   help='weight decay')
    # parser.add_argument(
    #     "--opts",
    #     help="Modify config options by adding 'KEY VALUE' pairs, e.g. --opts TRAIN.EPOCHS 300 DATA.BATCH_SIZE 32",
    #     default=None,
    #     nargs='+',
    # )
    
    parser.add_argument('--eval_interval',  type=int,   default=1,      help='interval for evaluation')
    parser.add_argument('--patience',       type=int,   default=50,     help='early stopping patience')
    parser.add_argument('--num_small_data', type=int,   default=None,   help='number of small data')
    parser.add_argument('--Tmax',           type=int,   default=45,     help='the numbers of half of T for CosineAnnealingLR')
    parser.add_argument('--eta_min',        type=float, default=1e-8,   help='minimum of lr for CosineAnnealingLR')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.change_params:
        args = param_modification.param_modification(args)

    detailed_time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
    # args.cfg = get_config(args)
    main(args, detailed_time_str)
