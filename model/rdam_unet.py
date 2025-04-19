import torch
from torchinfo import summary
import torch.nn as nn
from tensorboardX import SummaryWriter

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv(x)
        return x
       
class RDAM_UNet(nn.Module):
    def __init__(self, in_channels,
                 n_classes,
                 p, 
                 base_channels=32,
                 bilinear=True
                 ):
        super(RDAM_UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_channels = base_channels
        # 编码器
        self.inconv = DoubleConv(in_channels, base_channels)
        self.acpn0 = AMSFN(base_channels)
        self.msaf1 = MDAM(base_channels) 

        self.down1 = ResD_Down(base_channels, base_channels*2)
        self.acpn1 = AMSFN(base_channels*2)       
        self.msaf2 = MDAM(base_channels*2)
        self.encoder_dropout2 = nn.Dropout2d(p=p-0.2 if p!=0 else 0)  

        self.down2 = ResD_Down(base_channels*2, base_channels*4)
        self.acpn2 = AMSFN(base_channels*4) 
        self.msaf3 = MDAM(base_channels*4)
        self.encoder_dropout3 = nn.Dropout2d(p=p-0.1 if p!=0 else 0)  

        self.down3 = ResD_Down(base_channels*4, base_channels*8)
        self.acpn3 = AMSFN(base_channels*8)
        self.msaf4 = MDAM(base_channels*8)
        self.encoder_dropout4 = nn.Dropout2d(p=p)                            # 编码器更高dropout
    
        # self.dense_aspp = DenseASPPBlock(base_channels*8, base_channels*4, base_channels*8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center_conv = DoubleConv(base_channels*8, base_channels*8, mid_channels=base_channels*16)
        self.bottleneck_dropout = nn.Dropout2d(p=p+0.1 if p!=0.0 else 0.0)

        # 解码器
        self.up1 = ResD_Up(base_channels * 8 , base_channels * 4, bilinear=bilinear)
        self.acpn5 = AMSFN(base_channels*4)
        self.decoder_dropout1 = nn.Dropout2d(0.15 if p!=0 else 0)   

        self.up2 = ResD_Up(base_channels * 4, base_channels * 2 , bilinear=bilinear)
        self.acpn6 = AMSFN(base_channels*2)    
        self.decoder_dropout2 = nn.Dropout2d(0.1 if p!=0 else 0) 

        self.up3 = ResD_Up(base_channels * 2, base_channels, bilinear=bilinear)
        self.acpn7 = AMSFN(base_channels)
        self.decoder_dropout3 = nn.Dropout2d(0.05 if p!=0 else 0) 

        self.up4 = Up(base_channels, base_channels, bilinear=bilinear)
        self.acpn8 = AMSFN(base_channels)
        # 输出层
        self.out_conv = OutConv(base_channels, n_classes)

    def forward(self, x):
        e1 = self.inconv(x)
        x1 = self.acpn0(e1)
        m1 = self.msaf1(e1)                              # [b, 32, 256, 256]

        e2 = self.down1(x1)                              # [b, 64, 128, 128]
        a2 = self.acpn1(e2)
        m2 = self.msaf2(e2)
        x2 = self.encoder_dropout2(a2)
        
        e3 = self.down2(x2)                              # [b, 128, 64, 64]
        a3 = self.acpn2(e3)
        m3 = self.msaf3(e3)
        x3 = self.encoder_dropout3(a3)

        e4 = self.down3(x3)                              # [b, 256, 32, 32]
        a4 = self.acpn3(e4)
        m4 = self.msaf4(e4)
        x4 = self.encoder_dropout4(a4)
        
        x5 = self.pool(x4)                   # [b, 512, 16, 16]   
        c5 = self.center_conv(x5)                        # [b, 512, 16, 16]
        c5 = self.bottleneck_dropout(c5)

        d4 = self.up1(c5, m4)                            # [b, 256, 32, 32]
        d4 = self.acpn5(d4)
        x = self.decoder_dropout1(d4)   

        d3 = self.up2(x, m3)                             # [b, 128, 64, 64]
        d3 = self.acpn6(d3)
        x = self.decoder_dropout2(d3)

        d2 = self.up3(x, m2)                             # [b, 64, 128, 128]
        d2 = self.acpn7(d2)
        x = self.decoder_dropout3(d2)

        d1 = self.up4(x, m1)                             # [1, 64, 256, 256]
        x = self.acpn8(d1)
        logits = self.out_conv(x)                        # [1, c, 256, 256]
        
        return logits

if __name__ == "__main__":
    from utils.attention import *
    from utils.modules import * 
    model = RDAM_UNet(in_channels=3, n_classes=4, p=0)
    summary(model, (1, 3, 256, 256))

else:
    from model.utils.attention import *
    from model.utils.modules import *   