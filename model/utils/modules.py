import torch
import torch.nn as nn
import torch.nn.functional as F

"""-------------------------------------------------Convolution----------------------------------------------"""
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.cbr1 = nn.Sequential(self.conv1, self.bn1, self.relu)

        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.c2 = nn.Sequential(self.conv2, self.bn2, self.relu)
        
    def forward(self, x1):
        x = self.cbr1(x1)
        x = self.c2(x)
        return x

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DWConv, self).__init__()
        self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=in_channels)
        self.pconv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.dconv(x)
        x = self.pconv(x)
        return x

class RDModule(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2 
        self.DalitConv = nn.Sequential(
        nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        ),

        nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        res = x
        res = self.conv(res)
        x = self.DalitConv(x)
        x = torch.add(x, res)
        x = self.relu(x)
        return x

class AMSFN(nn.Module):  
    def __init__(self, in_channels, mid_channels=None):
        super(AMSFN, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2
        
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        
        # 3×3
        self.cbrp1 = nn.Sequential(
            DWConv(in_channels, in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
              
        # 5×5
        self.cbrp2 = nn.Sequential(
            DWConv(in_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DWConv(mid_channels, in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
               
        # 7×7
        self.cbrp3 = nn.Sequential(
            DWConv(in_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DWConv(mid_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DWConv(mid_channels, in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sgm = nn.Sigmoid()
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=1)
        )

    def forward(self, x):
        inputs = x
        b, c, h, w = inputs.size()
        c1 = self.maxpool(inputs)
        s1 = self.avg_pool(c1)

        c2 = self.cbrp1(inputs)
        s2 = self.avg_pool(c2)

        c3 = self.cbrp2(inputs)
        s3 = self.avg_pool(c3)

        c4 = self.cbrp3(inputs)
        s4 = self.avg_pool(c4)

        out = torch.cat([s1, s2, s3, s4], dim=1)
        out = self.mlp(out)
        weights = self.sgm(out)

        # 将权重作用到各个卷积结果上并相加
        c1_weighted = c1 * weights[:, 0:c, ...]
        c2_weighted = c2 * weights[:, c:c*2 , ...]
        c3_weighted = c3 * weights[:, c*2:c*3, ...]
        c4_weighted = c4 * weights[:, c*3:, ...]

        output = c1_weighted + c2_weighted + c3_weighted + c4_weighted
        output = self.final_conv(output)
        return output

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.resd = RDModule(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels)
        self.amsfn = AMSFN(out_channels, mid_channels=mid_channels)
        
    def forward(self, x):
        x = self.resd(x)
        x = self.amsfn(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.resd = RDModule(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels)
        self.amsfn = AMSFN(out_channels, mid_channels=mid_channels)
        
    def forward(self, x):
        x = self.resd(x)
        x = self.amsfn(x)
        return x
    
### CAB
# 直通估计器（Straight-Through Estimator），用于二值激活函数
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 前向传播，直接返回大于0的输入作为1，否则为0
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播，使用hardtanh函数来处理梯度
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        return STEFunction.apply(x)


# 因果图块（Causality Map Block）
class CausalityMapBlock(nn.Module):
    def __init__(self):
        super(CausalityMapBlock, self).__init__()
        # print("CausalityMapBlock initialized")

    def forward(self, x):
        """
        计算因果图，输入x的形状为(bs, k, n, n)，其中bs是批量大小，k是特征图的数量，
        n是特征图的空间维度。
        
        :param x: 网络的潜在特征
        :return: 因果图，形状为(bs, k, k)
        """
        if torch.isnan(x).any():
            # print("...the current feature maps object contains NaN")
            raise ValueError

        # 计算每个特征图的最大值
        maximum_values = torch.max(torch.flatten(x, 2), dim=2)[0]
        MAX_F = torch.max(maximum_values, dim=1)[0]
        x_div_max = x / (MAX_F.unsqueeze(1).unsqueeze(2).unsqueeze(3) + 1e-8)

        x = torch.nan_to_num(x_div_max, nan=0.0)

        # 计算因果图
        sum_values = torch.sum(torch.flatten(x, 2), dim=2)
        sum_values = torch.nan_to_num(sum_values, nan=0.0)
        maximum_values = torch.max(torch.flatten(x, 2), dim=2)[0]
        mtrx = torch.einsum('bi,bj->bij', maximum_values, maximum_values)
        tmp = mtrx / (sum_values.unsqueeze(1) + 1e-8)
        causality_maps = torch.nan_to_num(tmp, nan=0.0)

        # 归一化因果图
        max_cmaps = torch.max(causality_maps, dim=1, keepdim=True)[0]
        min_cmaps = torch.min(causality_maps, dim=1, keepdim=True)[0]
        causality_maps = (causality_maps - min_cmaps) / (max_cmaps - min_cmaps + 1e-8)

        # print(f"causality_maps: min {torch.min(causality_maps)}, max {torch.max(causality_maps)}, mean {torch.mean(causality_maps)}")
        return causality_maps


# 因果因子提取器（Causality Factors Extractor）
class CausalityFactorsExtractor(nn.Module):
    def __init__(self):
        super(CausalityFactorsExtractor, self).__init__()
        self.STE = StraightThroughEstimator()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # print("CausalityFactorsExtractor initialized")

    def forward(self, x, causality_maps):
        """
        根据因果图计算因果因子，输入x的形状为(bs, k, h, w)，其中bs是批量大小，k是特征图的数量，
        h和w是特征图的空间维度。因果图的形状为(bs, k, k)。
        
        :param x: 网络的潜在特征
        :param causality_maps: 因果图
        :return: 因果加权的特征图
        """
        triu = torch.triu(causality_maps, 1)
        tril = torch.tril(causality_maps, -1).permute((0, 2, 1)).contiguous()

        e = tril - triu
        e = self.STE(e)
        e = e.permute((0, 2, 1))

        f = triu - tril
        f = self.STE(f)
        bool_matrix = e + f

        by_col = torch.sum(bool_matrix, 2)
        by_row = torch.sum(bool_matrix, 1)

        multiplicative_factors = by_col - by_row
        multiplicative_factors = self.relu(multiplicative_factors)
        max_factors = torch.max(multiplicative_factors, dim=1, keepdim=True)[0]
        min_factors = torch.min(multiplicative_factors, dim=1, keepdim=True)[0]
        multiplicative_factors = (multiplicative_factors - min_factors) / (max_factors - min_factors + 1e-8)

        # print(f"multiplicative_factors: min {torch.min(multiplicative_factors)}, max {torch.max(multiplicative_factors)}, mean {torch.mean(multiplicative_factors)}")
        return torch.einsum('bkmn,bk->bkmn', x, multiplicative_factors)
    
"""---------------------------------------------Pyramid Pooling Module----------------------------------------------------"""
class PSPModule(nn.Module):
    def __init__(self, in_channels: int, bin_size_list: list = [1, 2, 4, 8]):
        super(PSPModule, self).__init__()
        branch_channels = in_channels // 4        ## C/4
        self.branches = nn.ModuleList()
        # CAB 模块
        self.causality_map_block = CausalityMapBlock()
        self.causality_factors_extractor = CausalityFactorsExtractor()
        for i in range(len(bin_size_list)):
            branch = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=bin_size_list[i]),  # 使用平均池化
                nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU()
            )
            self.branches.append(branch)       

    def forward(self, inputs):
        if not self.branches:
            return inputs
        final = None
        for i, branch in enumerate(self.branches):
            out = branch(inputs)
            # # CAB 模块
            # causality_maps = self.causality_map_block(out)
            # enhanced_features = self.causality_factors_extractor(out, causality_maps)
            
            out = F.interpolate(out, size=inputs.shape[2:], mode='bilinear', align_corners=True)
            if final is None:
                final = out
            else:
                final = torch.cat([final, out], dim=1)
        final = torch.cat([inputs, final], dim=1)  # 将各特征图在通道维上拼接起来
        return final
   
class AMSFN(nn.Module):  
    def __init__(self, in_channels, mid_channels=None):
        super(AMSFN, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels // 2 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # 3×3
        self.cbrp1 = nn.Sequential(
            DWConv(in_channels, in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
              
        # 5×5
        self.cbrp2 = nn.Sequential(
            DWConv(in_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DWConv(mid_channels, in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
               
        # 7×7
        self.cbrp3 = nn.Sequential(
            DWConv(in_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DWConv(mid_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DWConv(mid_channels, in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sgm = nn.Sigmoid()
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=1)
        )

    def forward(self, x):
        inputs = x
        b, c, h, w = inputs.size()
        c1 = self.maxpool(inputs)
        s1 = self.avg_pool(c1)

        c2 = self.cbrp1(inputs)
        s2 = self.avg_pool(c2)

        c3 = self.cbrp2(inputs)
        s3 = self.avg_pool(c3)

        c4 = self.cbrp3(inputs)
        s4 = self.avg_pool(c4)

        out = torch.cat([s1, s2, s3, s4], dim=1)
        out = self.mlp(out)
        weights = self.sgm(out)

        # 将权重作用到各个卷积结果上并相加
        c1_weighted = c1 * weights[:, 0:c   , ...]
        c2_weighted = c2 * weights[:, c:c*2 , ...]
        c3_weighted = c3 * weights[:, c*2:c*3, ...]
        c4_weighted = c4 * weights[:, c*3:  , ...]

        output = c1_weighted + c2_weighted + c3_weighted + c4_weighted
        output = self.final_conv(output)
        return output

class DynamicAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, C, H, W = x.size()
        Q = self.query(x).view(batch, -1, H*W).permute(0,2,1)  # [B, N, C']
        K = self.key(x).view(batch, -1, H*W)                   # [B, C', N]
        V = self.value(x).view(batch, -1, H*W)                 # [B, C, N]
        
        attention = torch.softmax(torch.bmm(Q, K) / (C**0.5), dim=-1)
        out = torch.bmm(V, attention.permute(0,2,1)).view(batch, C, H, W)
        return self.gamma * out + x

class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features


class DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(DenseASPPBlock, self).__init__()
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1,
                                      norm_layer, norm_kwargs)
        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels2 * 5, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)

        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)

        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)

        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)

        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)

        x = self.cbr1(x)
        return x

if __name__ == '__main__':
    from attention import *
    x = torch.randn(16, 3, 256, 256)
    model = AMSFN(3, 4)
    out = model(x)
    print(out.shape, "\n",
          model)

else:
    from model.utils.attention import *

        
     

        