import torch
import numpy as np
from PIL import Image
import torchvision
from torch.utils.data import DataLoader
from utils.my_data import SEM_DATA
import argparse
import os
import time
from model.rdam_unet import *
from model.unet_3plus import UNet_3Plus
from model.swin_unet import SwinUnet
from model.res50_unet import RES50_UNet
from monai.networks.nets import SegResNet, SwinUNETR, UNet, AttentionUnet
from model.vim_unet import VMUNet
from model.unet_plusplus import UnetPlusPlus
from tqdm import tqdm
from tabulate import tabulate
from utils.model_initial import *
from utils.metrics import Evaluate_Metric
import utils.transforms as T
from typing import Union, List


class SODPresetEval:
    def __init__(self, base_size: Union[int, List[int]], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=False),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def count_flops(model, input_size=(1, 3, 256, 256), device='cpu'):
    """使用 torch.profiler 估算 FLOPs"""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        inp = torch.randn(input_size).to(device)
        with FlopCounterMode(display=False) as flop_counter:
            model(inp)
        total_flops = flop_counter.get_total_flops()
        return total_flops
    except ImportError:
        # fallback: 使用 torchinfo
        from torchinfo import summary
        stats = summary(model, input_size=input_size, verbose=0)
        return stats.total_mult_adds


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # -------------------- 加载数据集 --------------------
    if 'swin_unet' in args.model:
        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            T.Resize((224, 224)),
        ])
    else:
        transforms = SODPresetEval([args.base_size, args.base_size])

    test_dataset = SEM_DATA(args.test_csv, transforms=transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # -------------------- 加载模型 --------------------
    model_map = {
        "unet": UNet(spatial_dims=2, in_channels=3, out_channels=4,
                      channels=(16*2, 32*2, 64*2, 128*2, 256*2), strides=(2, 2, 2, 2)),
        "a_unet": A_UNet(in_channels=3, num_classes=4, base_channels=32, p=args.dropout_p),
        "m_unet": M_UNet(in_channels=3, num_classes=4, base_channels=32, p=args.dropout_p, w=args.w),
        "rdam_unet": RDAM_UNet(in_channels=3, num_classes=4, base_channels=32, p=args.dropout_p, w=args.w),
        "unet_plusplus": UnetPlusPlus(in_channels=3, num_classes=4, deep_supervision=False),
        "unet_3plus": UNet_3Plus(in_channels=3, num_classes=4),
        "swin_unet": SwinUnet(in_channels=3, num_classes=4, img_size=224, drop_rate=args.dropout_p),
        "res50_unet": RES50_UNet(in_channels=3, num_classes=4),
        "att_unet": AttentionUnet(spatial_dims=2, in_channels=3, out_channels=4,
                                        channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2)),
        "vim_unet": VMUNet(input_channels=3, num_classes=4, dropout_r=args.dropout_p),
    }
    model = model_map.get(args.model)
    if not model:
        raise ValueError(f"Invalid model name: {args.model}")

    # 加载权重
    pretrain_weights = torch.load(args.weights_path, weights_only=False)
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)

    model = model.to(device)
    model.eval()

    # -------------------- 计算 FLOPs --------------------
    img_size = 224 if 'swin_unet' in args.model else args.base_size
    total_flops = count_flops(model, input_size=(1, 3, img_size, img_size), device=device)
    flops_g = total_flops / 1e9

    # -------------------- 推理 + 指标计算 --------------------
    Metric = Evaluate_Metric()
    Metric_list = np.zeros((6, 4))
    total_inference_time = 0.0
    num_samples = 0

    save_path = f"{args.save_path}/{args.model}"

    with torch.no_grad():
        test_iter = tqdm(test_loader, desc="  Testing  😀", leave=False)
        for data in test_iter:
            images, masks = data[0][0].to(device), data[0][1].to(device)
            img_name = data[1][0]

            # 推理计时
            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()

            logits = model(images)

            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()

            total_inference_time += (end_time - start_time)
            num_samples += 1

            masks = masks.to(torch.int64).squeeze(1)

            # 指标计算
            pred_softmax = torch.softmax(logits, dim=1)
            metrics = Metric.update(pred_softmax, masks)
            Metric_list += metrics

            # 保存预测图
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).to(torch.uint8).cpu().numpy()
            pred_img_pil = Image.fromarray(pred_mask)
            pred_dir = f"{save_path}/pred_img/{t}"
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            pred_img_pil.save(f"{pred_dir}/{os.path.splitext(img_name)[0]}.png")

    Metric_list /= num_samples

    # -------------------- 统计结果 --------------------
    avg_inference_time = total_inference_time / num_samples * 1000  # ms
    fps = num_samples / total_inference_time

    val_metrics = {
        "Recall":    Metric_list[0],
        "Precision": Metric_list[1],
        "Dice":      Metric_list[2],
        "F1_scores": Metric_list[3],
        "mIoU":      Metric_list[4],
        "Accuracy":  Metric_list[5],
    }

    metrics_table_header = ['Metrics_Name', 'Mean', 'OM', 'OP', 'IOP']
    metrics_table_left = ['Dice', 'Recall', 'Precision', 'F1_scores', 'mIoU', 'Accuracy']

    metrics_table = [
        [name,
         val_metrics[name][-1],
         val_metrics[name][0],
         val_metrics[name][1],
         val_metrics[name][2]]
        for name in metrics_table_left
    ]

    table_s = tabulate(metrics_table, headers=metrics_table_header, tablefmt='grid')

    # 性能统计表
    perf_table = [
        ['FLOPs (G)',          f'{flops_g:.2f}'],
        ['Avg Inference (ms)', f'{avg_inference_time:.2f}'],
        ['FPS',                f'{fps:.2f}'],
        ['Total Samples',      f'{num_samples}'],
    ]
    perf_s = tabulate(perf_table, headers=['Performance', 'Value'], tablefmt='grid')

    # 打印结果
    print(f"\n{'='*60}")
    print(f"  Model: {args.model}")
    print(f"{'='*60}")
    print(table_s)
    print()
    print(perf_s)
    print(f"{'='*60}\n")

    # 保存结果到文件
    write_info = (
        f"Model: {args.model}\n"
        f"Weights: {args.weights_path}\n"
        f"Test CSV: {args.test_csv}\n\n"
        f"{table_s}\n\n"
        f"{perf_s}\n"
    )
    scores_dir = f"{save_path}/scores"
    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)
    file_path = f"{scores_dir}/{t}.txt"
    with open(file_path, "w") as f:
        f.write(write_info)
    print(f"Results saved to {file_path}")
    print("预测完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SEM图像分割预测脚本")

    # 数据
    parser.add_argument('--test_csv',       type=str,
                        default='./test/aug0/test_aug0.csv',
                        help='path of test csv dataset')
    parser.add_argument('--base_size',      type=int,   default=256)

    # 模型
    parser.add_argument('--model',          type=str,   default="res50_unet",
                        help="unet, rdam_unet, a_unet, m_unet, unet_3plus, unet_plusplus, "
                             "swin_unet, res50_unet, att_unet, vim_unet")
    parser.add_argument('--weights_path',   type=str,   default="/root/RDAMU-Net/results/save_weights/res50_unet/L_DiceLoss--S_CosineAnnealingLR/optim_AdamW-lr_0.0008-wd_1e-06/2026-02-22_19:58:13/model_best_ep_2_w0.7_seed42.pth",
                        help='path of model weights (.pth)')
    parser.add_argument('--w',              type=float, default=0.7)
    parser.add_argument('--dropout_p',      type=float, default=0.4)

    # 保存
    parser.add_argument('--save_path',      type=str,
                        default='./results/predict',
                        help='path to save prediction results')

    args = parser.parse_args()
    main(args)
