"""
注意力机制
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
"""-------------------------------------------------------------HAM--------------------------------------------"""
class ChannelAttention(nn.Module):
    def __init__(self, Channel_nums):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化
        self.alpha = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.beta = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.gamma = 2
        self.b = 1
        self.k = self.get_kernel_num(Channel_nums)
        self.conv1d = nn.Conv1d(kernel_size=self.k, in_channels=1, out_channels=1, padding=self.k // 2)  # C1D 一维卷积
        self.sigmoid = nn.Sigmoid()

    def get_kernel_num(self, C):  # 根据通道数求一维卷积大卷积核大小 odd|t|最近奇数
        t = math.log2(C) / self.gamma + self.b / self.gamma
        floor = math.floor(t)
        k = floor + (1 - floor % 2)
        return k

    def forward(self, x):
        F_avg = self.avg_pool(x)
        F_max = self.max_pool(x)
        F_add = 0.5 * (F_avg + F_max) + self.alpha * F_avg + self.beta * F_max
        F_add_ = F_add.squeeze(-1).permute(0, 2, 1)
        F_add_ = self.conv1d(F_add_).permute(0, 2, 1).unsqueeze(-1)
        out = self.sigmoid(F_add_)
        return out
    
class SpatialAttention(nn.Module):
    def __init__(self, Channel_num):
        super(SpatialAttention, self).__init__()
        self.channel = Channel_num
        self.Lambda = 0.6  # separation rate 论文中经过实验发现0.6效果最佳
        self.C_im = self.get_important_channelNum(Channel_num)
        self.C_subim = Channel_num - self.C_im
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.norm_active = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def get_important_channelNum(self, C):  # 根据通道数以及分离率确定重要通道的数量 even|t|最近偶数
        t = self.Lambda * C
        floor = math.floor(t)
        C_im = floor + floor % 2
        return C_im

    
    def get_im_subim_channels(self, C_im, M): # 根据Channel_Attention_Map得到重要通道以及不重要的通道
        _, topk = torch.topk(M, dim=1, k=C_im)
        important_channels = torch.zeros_like(M)
        subimportant_channels = torch.ones_like(M)
        important_channels = important_channels.scatter(1, topk, 1)
        subimportant_channels = subimportant_channels.scatter(1, topk, 0)
        return important_channels, subimportant_channels

    def get_features(self, im_channels, subim_channels, channel_refined_feature):
        import_features = im_channels * channel_refined_feature
        subimportant_features = subim_channels * channel_refined_feature
        return import_features, subimportant_features

    def forward(self, x, M):
        important_channels, subimportant_channels = self.get_im_subim_channels(self.C_im, M)
        important_features, subimportant_features = self.get_features(important_channels, subimportant_channels, x)

        im_AvgPool = torch.mean(important_features, dim=1, keepdim=True) * (self.channel / self.C_im)
        im_MaxPool, _ = torch.max(important_features, dim=1, keepdim=True)

        subim_AvgPool = torch.mean(subimportant_features, dim=1, keepdim=True) * (self.channel / self.C_subim)
        subim_MaxPool, _ = torch.max(subimportant_features, dim=1, keepdim=True)

        im_x = torch.cat([im_AvgPool, im_MaxPool], dim=1)
        subim_x = torch.cat([subim_AvgPool, subim_MaxPool], dim=1)

        A_S1 = self.norm_active(self.conv(im_x))
        A_S2 = self.norm_active(self.conv(subim_x))

        F1 = important_features * A_S1
        F2 = subimportant_features * A_S2

        refined_feature = F1 + F2

        return refined_feature
    
class Res_HAM(nn.Module):
    def __init__(self, Channel_nums):
        super(Res_HAM, self).__init__()
        self.channel = Channel_nums
        self.ChannelAttention = ChannelAttention(self.channel)
        self.SpatialAttention = SpatialAttention(self.channel)
        self.relu = nn.ReLU()

    def forward(self, x_in):
        residual = x_in
        channel_attention_map = self.ChannelAttention(x_in)
        channel_refined_feature = channel_attention_map * x_in
        final_refined_feature = self.SpatialAttention(channel_refined_feature, channel_attention_map)
        out = self.relu(final_refined_feature + residual)
        return out
    
class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.group = factor
        assert channels // self.group > 0
        self.softmax = nn.Softmax(dim=-1)
        self.averagePooling = nn.AdaptiveAvgPool2d((1,1))
        self.Pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.Pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.groupNorm = nn.GroupNorm(channels // self.group, channels//self.group)
        self.conv1x1 = nn.Conv2d(channels // self.group, channels // self.group, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.group, channels // self.group, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b*self.group, -1, h, w)
        x_h = self.Pool_h(group_x)  # 高度方向池化
        x_w = self.Pool_w(group_x).permute(0, 1, 3, 2)  # 宽度方向池化

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2)) # 拼接之后卷积
        x_h, x_w = torch.split(hw, [h, w], dim=2)       # 拆分

        # 1×1
        x1 = self.groupNorm(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())          # 高度的注意力
        x11 = self.softmax(self.averagePooling(x1).reshape(b*self.group, -1, 1).permute(0, 2, 1)) # 对 x1 进行平均池化，然后进行 softmax 操作
        x12 = x1.reshape(b*self.group, c//self.group, -1)

        # 3×3
        x2 = self.conv3x3(group_x) # 通过 3x3卷积层
        x21 = self.softmax(self.averagePooling(x2).reshape(b*self.group, -1, 1).permute(0, 2, 1)) # 对 x2 进行平均池化，然后进行 softmax 操作
        x22 = x2.reshape(b*self.group, c//self.group, -1)

        weights = (torch.matmul(x11, x22) + torch.matmul(x21, x12)).reshape(b * self.group, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
    
class Att_gate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Att_gate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        return x * psi
    
""""----------------------------------------------------MDAM-----------------------------------------------------"""
class MDAM(nn.Module):
    def __init__(self, channels, w=0.5,factor=32):
        super(MDAM, self).__init__()
        self.group = factor
        self.w = w
        assert channels // self.group > 0
        self.softmax = nn.Softmax(dim=1)
        self.averagePooling = nn.AdaptiveAvgPool2d((1,1))
        self.Pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.Pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.groupNorm = nn.GroupNorm(channels // self.group, channels//self.group)
        self.conv1x1 = nn.Conv2d(channels // self.group, channels // self.group, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.group, channels // self.group, kernel_size=3, stride=1, padding=1)

        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Conv2d(channels // self.group, channels // self.group, kernel_size=1, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Conv2d(channels // self.group, channels // self.group, kernel_size=1, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, inputs):
        b, c, h, w = inputs.size()
        group_x = inputs.reshape(b*self.group, -1, h, w)
        x_h = self.Pool_h(group_x)  # 高度方向池化
        x_w = self.Pool_w(group_x).permute(0, 1, 3, 2)  # 宽度方向池化

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2)) # 拼接之后卷积
        x_h, x_w = torch.split(hw, [h, w], dim=2)       # 拆分

        # H W
        x = self.groupNorm(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())          # 高度的注意力
        x = self.softmax(self.averagePooling(x))

        # SE
        y = self.conv3x3(group_x) # 通过 3x3卷积层
        y = self.gap(y)
        y = self.fc(y)

        weights = self.w*x + (1-self.w)*y

        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ema = MDAM(32).to(device)
    input_data = torch.rand(1, 32, 256, 256).to(device)
    output_data = ema(input_data)

    print(ema)

    print(output_data.shape)