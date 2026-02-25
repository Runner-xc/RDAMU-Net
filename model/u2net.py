"""
u2net model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int=3, dilation: int=1):
        super(ConvBNReLU, self).__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, 
                              out_ch, 
                              kernel_size=kernel_size, 
                              padding=padding, 
                              dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    
    
    def forward(self, x :torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))
    
class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int=3, dilation: int=1, run: bool=True):
        super(DownConvBNReLU, self).__init__(in_ch, out_ch, kernel_size, dilation)
        self.run = run
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.run:
            x = F.max_pool2d(x, stride=2, kernel_size=2, ceil_mode=True)
        return self.relu(self.bn(self.conv(x)))
    
class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int=3, dilation: int=1, run: bool=True):
        super(UpConvBNReLU, self).__init__(in_ch, out_ch, kernel_size, dilation)
        self.run = run

    def forward(self, x1: torch.Tensor, x2:torch.Tensor) -> torch.Tensor:
        # x1是最后一次卷积（不是下采样）的输出，x2是上一次下采样的输出。
        if self.run:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))

class RSU(nn.Module):
    # 使用 depth 控制编码解码模块层数
    def __init__(self, depth: int, in_ch, mid_ch, out_ch):
        super(RSU, self).__init__()
        
        assert depth >= 2
        self.conv_first = ConvBNReLU(in_ch, out_ch)
        
        encode_list = [DownConvBNReLU(out_ch, mid_ch, run=False)]
        decode_list = [UpConvBNReLU(mid_ch*2, mid_ch, run=False)]
        for i in range(depth - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch*2, mid_ch if i < depth -3 else out_ch))
        
        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)
    
    def forward(self, x: torch.Tensor):
        x_first = self.conv_first(x) #  (1, 1, 288, 288) - > (1, 64， 288， 288）

        x = x_first
        # 遍历编码器
        encode_outputs = []
        for a in self.encode_modules:
            x = a(x)
            encode_outputs.append(x)
        # 遍历解码器
        x = encode_outputs.pop()
        for a in self.decode_modules:
            x_up  = encode_outputs.pop()  # 这里的x_up是上一次下采样的输出
            x = a(x, x_up)
        return x+x_first

class RSU4F(nn.Module):
    def __init__(self, in_ch,mid_ch, out_ch):
        super().__init__()
        self.con_first = ConvBNReLU(in_ch, out_ch)
        self.encode_list = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                          ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                          ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                          ConvBNReLU(mid_ch, mid_ch, dilation=8)])
        
        self.decode_list = nn.ModuleList([ConvBNReLU(mid_ch*2, mid_ch, dilation=4),
                                          ConvBNReLU(mid_ch*2, mid_ch, dilation=2),
                                          ConvBNReLU(mid_ch*2, out_ch)])

    def forward(self, x:torch.Tensor):
        x_first = self.con_first(x)
        encode_outputs = []

        x = x_first
        for i in self.encode_list:
            x = i(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for i in self.decode_list:
            x_up = encode_outputs.pop()
            x = i(torch.cat([x, x_up], dim=1))

        return x + x_first

class U2net(nn.Module):
    def __init__(self, cfg: dict, out_ch: int=3):
        super().__init__()
        assert "encode" in cfg
        assert "decode" in cfg
        self.encode_num = len(cfg["encode"])
        # 定义编码器
        encode_list = []
        side_list = []
        for a in cfg["encode"]:
            # a = [depth, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(a) == 6
            encode_list.append(RSU(*a[:4]) if a[4] is False else RSU4F(*a[1:4]))
            # 添加 side 列表元素
            if a[5] is True:
                side_list.append(nn.Conv2d(a[3], out_ch, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)

        # 定义解码器
        decode_list = []
        for a in cfg["decode"]:
            # a = [depth, in_ch, mid_ch, out_ch]
            assert len(a) == 6
            decode_list.append(RSU(*a[:4]) if a[4] is False else RSU4F(*a[1:4]))
            # 添加 side 列表元素
            if a[5] is True:
                side_list.append(nn.Conv2d(a[3], out_ch, kernel_size=3, padding=1))

        self.decode_modules = nn.ModuleList(decode_list)
        self.side_list = nn.ModuleList(side_list)
        self.out_conv = nn.Conv2d(self.encode_num*out_ch, out_ch, kernel_size=1)       
            
    def forward(self, x:torch.Tensor):
        _, _, h, w = x.shape
        # 记录编码器输出
        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            x = m(x)
            encode_outputs.append(x)
            if i != (self.encode_num - 1):
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        # 记录解码器输出
        x = encode_outputs.pop()
        decode_outputs = [x]
        for m in self.decode_modules:
            x_last = encode_outputs.pop()
            x = F.interpolate(x, size=x_last.shape[2:], mode="bilinear", align_corners=False)
            x =m(torch.cat([x, x_last], dim=1))
            decode_outputs.insert(0, x)
        # 记录 side 输出
        side_outputs = []
        for m in self.side_list:
            x = decode_outputs.pop()
            x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
            side_outputs.insert(0, x)
        x = self.out_conv(torch.concat(side_outputs, dim=1))

        if self.training:
            # do not use torch.sigmoid for amp safe
            return [x] + side_outputs
        else:
            return torch.softmax(x, dim=1) 

def u2net_full_config(out_ch: int = 4):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 32, 64, False, False],      # En1
                   [6, 64, 32, 128, False, False],    # En2
                   [5, 128, 64, 256, False, False],   # En3
                   [4, 256, 128, 512, False, False],  # En4
                   [4, 512, 256, 512, True, False],   # En5
                   [4, 512, 256, 512, True, True]],   # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 1024, 256, 512, True, True],   # De5
                   [4, 1024, 128, 256, False, True],  # De4
                   [5, 512, 64, 128, False, True],    # De3
                   [6, 256, 32, 64, False, True],     # De2
                   [7, 128, 16, 64, False, True]]     # De1
    }

    return U2net(cfg, out_ch)

def u2net_lite_config(out_ch: int = 4):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 16, 64, False, False],  # En3
                   [4, 64, 16, 64, False, False],  # En4
                   [4, 64, 16, 64, True, False],  # En5
                   [4, 64, 16, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 128, 16, 64, True, True],  # De5
                   [4, 128, 16, 64, False, True],  # De4
                   [5, 128, 16, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return U2net(cfg, out_ch)

def convert_onnx(model, save_path):
    model.eval()
    x = torch.rand(1, 1, 288, 288, requires_grad=True)

    # export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      save_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,
                      opset_version=11)
 
