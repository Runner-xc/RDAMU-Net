import torch
from torchinfo import summary
import torch.nn as nn
from model.utils.attention import Att_gate
from model.utils.modules import DoubleConv
from tensorboardX import SummaryWriter 

class Attention_UNet(nn.Module):
    def __init__(self, 
                 in_channels,
                 num_classes,
                 p, 
                 base_channels=32,
                 ):
        super().__init__()
        # 编码器
        self.encoder1 = DoubleConv(in_channels, base_channels)
        self.encoder2 = DoubleConv(base_channels, base_channels*2)
        self.encoder3 = DoubleConv(base_channels*2, base_channels*4)
        self.encoder4 = DoubleConv(base_channels*4, base_channels*8)

        self.att_gate1 = Att_gate(base_channels,   base_channels,   base_channels//2)
        self.att_gate2 = Att_gate(base_channels*2, base_channels*2, base_channels)
        self.att_gate3 = Att_gate(base_channels*4, base_channels*4, base_channels*2)
        self.att_gate4 = Att_gate(base_channels*8, base_channels*8, base_channels*4) 

        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        self.dropout = nn.Dropout2d(p=p)

        # 解码器
        self.decoder1 = DoubleConv(base_channels * 16 ,  base_channels * 4) 
        self.decoder2 = DoubleConv(base_channels * 8 ,   base_channels * 2)
        self.decoder3 = DoubleConv(base_channels * 4 ,   base_channels)
        self.decoder4 = DoubleConv(base_channels * 2,    base_channels)
    def forward(self, x):
        x1 = self.encoder1(x)               # [1, 32, 320, 320]
        x2 = self.down(x1)
        x2 = self.dropout(x2)

        x2 = self.encoder2(x2)              # [1, 64, 160, 160]
        x3 = self.down(x2)
        x3 = self.dropout(x3)

        x3 = self.encoder3(x3)              # [1, 128, 80, 80]
        x4 = self.down(x3)
        x4 = self.dropout(x4)

        x4 = self.encoder4(x4)              # [1, 256, 40, 40]
        x5 = self.down(x4)
        x5 = self.dropout(x5)
        
        x = self.center_conv(x5)            # [1, 256, 20, 20]
        x = self.dropout(x)
        
        x = self.up(x)                      # [1, 256, 40, 40]
        x4 = self.att_gate4(x, x4)
        x = torch.cat([x, x4], dim=1)       # [1, 256+256, 40, 40]            
        x = self.decoder1(x)                # [1, 128, 40, 40]
        x = self.dropout(x)

        x = self.up(x)                      # [1, 128, 80, 80]
        x3 = self.att_gate3(x, x3)
        x = torch.cat([x, x3], dim=1)       # [1, 128+128, 80, 80]
        x = self.decoder2(x)                # [1, 64, 80, 80]
        x = self.dropout(x)

        x = self.up(x)                      # [1, 64, 160, 160]
        x2 = self.att_gate2(x, x2)
        x = torch.cat([x, x2], dim=1)       # [1, 64+64, 160, 160]
        x = self.decoder3(x)                # [1, 32, 160, 160]
        x = self.dropout(x)

        x = self.up(x)                      # [1, 32, 160, 160]
        x1 = self.att_gate1(x, x1)
        x = torch.cat([x, x1], dim=1)       # [1, 32+32, 320, 320]
        x = self.decoder4(x)                # [1, 32, 320, 320]
        x = self.dropout(x)
        
        logits = self.out_conv(x)           # [1, c, 320, 320]       
        return logits