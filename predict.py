import torch
from PIL import Image
import torchvision
from torch.utils.data import DataLoader, Dataset
from utils.my_data import SEM_DATA
import argparse
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
from tqdm import tqdm
from tabulate import tabulate
from utils.train_and_eval import *
from utils.model_initial import *
from utils.loss_fn import *
from utils.metrics import Evaluate_Metric
import utils.transforms as T
from typing import Union, List
from utils.slide_predict import SlidingWindowPredictor

class SODPresetEval:
    def __init__(self, base_size: Union[int, List[int]], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=False),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        data = self.transforms(img, target)
        return data
    
t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载数据集
    test_dataset = SEM_DATA(args.data_path, transforms=SODPresetEval([256, 256]))
    
    num_workers = 4
    test_loader = DataLoader(test_dataset, 
                             batch_size=1, 
                             shuffle=False, 
                             num_workers=num_workers)
    
    
    # 加载模型
    model_map = {
            "unet": UNet(spatial_dims=2,in_channels=3, out_channels=4, channels=(16*2, 32*2, 64*2, 128*2, 256*2), strides=(2, 2, 2, 2)),
            "a_unet": A_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p),
            "m_unet": M_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p, w=args.w),
            "rdam_unet": RDAM_UNet(in_channels=3, n_classes=4, base_channels=32, bilinear=True, p=args.dropout_p, w=args.w),

            "unet_plusplus": UnetPlusPlus(in_channels=3, num_classes=4, deep_supervision=False),
            "unet_3plus": UNet_3Plus(in_channels=3, num_classes=4),
            # "swin_unet": SwinUnet(config=args.cfg,  num_classes=4),
            "res50_unet": RES50_UNet(in_channels=3, num_classes=4),
            "att_unet": Attention_UNet(in_channels=3, num_classes=4, p=args.dropout_p, base_channels=32),
            "vim_unet": VMUNet(input_channels=3, num_classes=4, dropout_r=args.dropout_p),
        }
    model = model_map.get(args.model)
    if not model:
        raise ValueError(f"Invalid model name: {args.model}")
    
    
    # 加载模型权重
    pretrain_weights = torch.load(args.weights_path, weights_only=False)
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
        
    model = model.to(device)
    model.eval()
    Metric = Evaluate_Metric()

    # 创建优化后的预测器
    predictor = SlidingWindowPredictor(
        model=model,
        device=device,
        window_size=1024,
        stride=512  # 重叠步长设置为窗口大小的一半
    )

    if args.single:
        # 单张预测
        image_path = "/mnt/e/VScode/WS-Hub/Linux-RDAMU_Net/RDAMU-Net/Image1 - 003.jpeg"
        # 滑窗预测
        if args.slide:
            save_path = f"{args.single_path}/{args.model_name}_sliding.png"
            predictor.predict(image_path, save_path)
        
        else:
            img = Image.open(image_path).convert('RGB')
            img = np.array(img)
            img = torchvision.transforms.ToTensor()(img).to(device)
            img = torchvision.transforms.Resize((1792, 2048))(img)
            img = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(img)
            img = img.unsqueeze(0)
            logits = model(img)
            pred_mask = torch.argmax(logits, dim=1)  
            pred_mask = pred_mask.squeeze(0)          
            pred_mask = pred_mask.to(torch.uint8).cpu()       
            pred_mask_np = pred_mask.numpy()
            pred_img_pil = Image.fromarray(pred_mask_np)
            # 保存图片
            single_path = args.single_path
            if not os.path.exists(single_path):
                os.mkdir(single_path)
            pred_img_pil.save(f"{single_path}/{args.model_name}_V1.png")        
            print("预测完成!")
        
    else:
        # test 多张
        with torch.no_grad():
            Metric_list = np.zeros((6, 4))
            test_loader = tqdm(test_loader, desc=f"  Validating  😀", leave=False)
            save_path = f"{args.save_path}/{args.model_name}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            for data in test_loader:
                images, masks = data[0][0].to(device), data[0][1].to(device)
                img_name = data[1][0]
                logits = model(images)  # [1, 4, 320, 320]
                masks = masks.to(torch.int64)
                masks = masks.squeeze(1)
                
                # 使用 argmax 获取每个像素点预测最大的类别索引
                pred_mask = torch.softmax(logits, dim=1)
                metrics = Metric.update(pred_mask, masks)
                Metric_list += metrics
                pred_mask = torch.argmax(logits, dim=1)  # [1, 320, 320]

                pred_mask = pred_mask.squeeze(0)  # [320, 320]
                pred_mask = pred_mask.to(torch.uint8)
                pred_mask = pred_mask.cpu()
                pred_mask_np = pred_mask.numpy()
                pred_img_pil = Image.fromarray(pred_mask_np)
                
                # 保存图片
                if not os.path.exists(f"{save_path}/pred_img/V2/{t}"):
                    os.makedirs(f"{save_path}/pred_img/V2/{t}")
                pred_img_pil.save(f"{save_path}/pred_img/V2/{t}/{os.path.splitext(img_name)[0]}.png")
            Metric_list /= len(test_loader)
        metrics_table_header = ['Metrics_Name', 'Mean', 'OM', 'OP', 'IOP']
        metrics_table_left = ['Dice', 'Recall', 'Precision', 'F1_scores', 'mIoU', 'Accuracy']
        val_metrics = {}
        val_metrics["Recall"] = Metric_list[0]
        val_metrics["Precision"] = Metric_list[1]
        val_metrics["Dice"] = Metric_list[2]
        val_metrics["F1_scores"] = Metric_list[3]
        val_metrics["mIoU"] = Metric_list[4]
        val_metrics["Accuracy"] = Metric_list[5]
        metrics_dict = {scores : val_metrics[scores] for scores in metrics_table_left}
        metrics_table = [[metric_name,
                            metrics_dict[metric_name][-1],
                            metrics_dict[metric_name][0],
                            metrics_dict[metric_name][1],
                            metrics_dict[metric_name][2]
                        ]
                            for metric_name in metrics_table_left
                        ]
        table_s = tabulate(metrics_table, headers=metrics_table_header, tablefmt='grid')
        write_info = f"{args.model_name}" + "\n" + table_s
        file_path = f'{save_path}/scores/{t}.txt'
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path)) 
        with open(file_path, "a") as f:
                    f.write(write_info)
        print(table_s)
        print("预测完成！")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',      type=str,       default='/mnt/e/VScode/WS-Hub/WS-UNet/UNet/datasets/CSV/test_shale_256_v2.csv')
    parser.add_argument('--base_size',      type=int,       default=256)
    parser.add_argument('--w',              type=float,     default=0.65)
    parser.add_argument('--model',          type=str, 
                        default="vim_unet", 
                        help=" unet, rdam_unet, a_unet, m_unet, unet_3plus, unet_plusplus, swin_unet, res50_unet, vim_unet\
                               att_unet, u2net_full, u2net_lite,")
    
    parser.add_argument('--weights_path',   type=str,       
                                            default='/mnt/e/VScode/WS-Hub/WS-UNet/UNet/old/training_results/weights/msaf_unetv2/L: DiceLoss--S: CosineAnnealingLR/optim: AdamW-lr: 0.0008-wd: 1e-06/2025-03-10_09-44-44/model_best_ep_121.pth')
    
    parser.add_argument('--save_path',      type=str,       default='/mnt/e/VScode/WS-Hub/Linux-RDAMU_Net/RDAMU-Net/results/predict')
    parser.add_argument('--single_path',    type=str,       default='/mnt/e/VScode/WS-Hub/WS-UNet/UNet/results/single_predict')
    parser.add_argument('--single',         type=bool,      default=False,          help='test single img or not')
    parser.add_argument('--slide',          type=bool,      default=False)
    
    args = parser.parse_args()
    main(args)
