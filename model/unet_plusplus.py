import torch
import torch.nn as nn
from torchinfo import summary

class ContinusParalleConv(nn.Module):
    # 一个连续的卷积模块，包含BatchNorm 在前 和 在后 两种模式
    def __init__(self, in_channels, out_channels, pre_Batch_Norm = True):
        super(ContinusParalleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
 
        if pre_Batch_Norm:
          self.Conv_forward = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
 
        else:
          self.Conv_forward = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU())
 
    def forward(self, x):
        x = self.Conv_forward(x)
        return x
 
class UnetPlusPlus(nn.Module):
    def __init__(self, in_channels, num_classes, base_channel=32, deep_supervision=False):
        super(UnetPlusPlus, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.filters = [base_channel, base_channel*2, base_channel*4, base_channel*8,  base_channel*16]
        
        self.CONV3_1 = ContinusParalleConv(base_channel*16, base_channel*8, pre_Batch_Norm = True)
 
        self.CONV2_2 = ContinusParalleConv(base_channel*4*3, base_channel*4, pre_Batch_Norm = True)
        self.CONV2_1 = ContinusParalleConv(base_channel*4*2, base_channel*4, pre_Batch_Norm = True)
 
        self.CONV1_1 = ContinusParalleConv(base_channel*2*2, base_channel*2, pre_Batch_Norm = True)
        self.CONV1_2 = ContinusParalleConv(base_channel*2*3, base_channel*2, pre_Batch_Norm = True)
        self.CONV1_3 = ContinusParalleConv(base_channel*2*4, base_channel*2, pre_Batch_Norm = True)
 
        self.CONV0_1 = ContinusParalleConv(base_channel*2, base_channel, pre_Batch_Norm = True)
        self.CONV0_2 = ContinusParalleConv(base_channel*3, base_channel, pre_Batch_Norm = True)
        self.CONV0_3 = ContinusParalleConv(base_channel*4, base_channel, pre_Batch_Norm = True)
        self.CONV0_4 = ContinusParalleConv(base_channel*5, base_channel, pre_Batch_Norm = True)
 
 
        self.stage_0 = ContinusParalleConv(in_channels,    base_channel,     pre_Batch_Norm = False)
        self.stage_1 = ContinusParalleConv(base_channel,   base_channel*2,   pre_Batch_Norm = False)
        self.stage_2 = ContinusParalleConv(base_channel*2, base_channel*4,   pre_Batch_Norm = False)
        self.stage_3 = ContinusParalleConv(base_channel*4, base_channel*8,   pre_Batch_Norm = False)
        self.stage_4 = ContinusParalleConv(base_channel*8, base_channel*16,  pre_Batch_Norm = False)
 
        self.pool = nn.MaxPool2d(2)
    
        self.upsample_3_1 = nn.ConvTranspose2d(in_channels=base_channel*16, out_channels=base_channel*8, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_2_1 = nn.ConvTranspose2d(in_channels=base_channel*8, out_channels=base_channel*4, kernel_size=4, stride=2, padding=1) 
        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=base_channel*8, out_channels=base_channel*4, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_1_1 = nn.ConvTranspose2d(in_channels=base_channel*4, out_channels=base_channel*2, kernel_size=4, stride=2, padding=1) 
        self.upsample_1_2 = nn.ConvTranspose2d(in_channels=base_channel*4, out_channels=base_channel*2, kernel_size=4, stride=2, padding=1) 
        self.upsample_1_3 = nn.ConvTranspose2d(in_channels=base_channel*4, out_channels=base_channel*2, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_0_1 = nn.ConvTranspose2d(in_channels=base_channel*2, out_channels=base_channel, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_2 = nn.ConvTranspose2d(in_channels=base_channel*2, out_channels=base_channel, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=base_channel*2, out_channels=base_channel, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_4 = nn.ConvTranspose2d(in_channels=base_channel*2, out_channels=base_channel, kernel_size=4, stride=2, padding=1) 
 
        
        # 分割头
        self.final_super_0_1 = nn.Sequential(
          nn.BatchNorm2d(base_channel),
          nn.ReLU(),
          nn.Conv2d(base_channel, self.num_classes, 3, padding=1),
        )        
        self.final_super_0_2 = nn.Sequential(
          nn.BatchNorm2d(base_channel),
          nn.ReLU(),
          nn.Conv2d(base_channel, self.num_classes, 3, padding=1),
        )        
        self.final_super_0_3 = nn.Sequential(
          nn.BatchNorm2d(base_channel),
          nn.ReLU(),
          nn.Conv2d(base_channel, self.num_classes, 3, padding=1),
        )        
        self.final_super_0_4 = nn.Sequential(
          nn.BatchNorm2d(base_channel),
          nn.ReLU(),
          nn.Conv2d(base_channel, self.num_classes, 3, padding=1),
        )        
 
        
    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool(x_0_0))
        x_2_0 = self.stage_2(self.pool(x_1_0))
        x_3_0 = self.stage_3(self.pool(x_2_0))
        x_4_0 = self.stage_4(self.pool(x_3_0))
        
        x_0_1 = torch.cat([self.upsample_0_1(x_1_0) , x_0_0], 1)
        x_0_1 =  self.CONV0_1(x_0_1)
        
        x_1_1 = torch.cat([self.upsample_1_1(x_2_0), x_1_0], 1)
        x_1_1 = self.CONV1_1(x_1_1)
        
        x_2_1 = torch.cat([self.upsample_2_1(x_3_0), x_2_0], 1)
        x_2_1 = self.CONV2_1(x_2_1)
        
        x_3_1 = torch.cat([self.upsample_3_1(x_4_0), x_3_0], 1)
        x_3_1 = self.CONV3_1(x_3_1)
 
        x_2_2 = torch.cat([self.upsample_2_2(x_3_1), x_2_0, x_2_1], 1)
        x_2_2 = self.CONV2_2(x_2_2)
        
        x_1_2 = torch.cat([self.upsample_1_2(x_2_1), x_1_0, x_1_1], 1)
        x_1_2 = self.CONV1_2(x_1_2)
        
        x_1_3 = torch.cat([self.upsample_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1)
        x_1_3 = self.CONV1_3(x_1_3)
 
        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_0, x_0_1], 1)
        x_0_2 = self.CONV0_2(x_0_2)
        
        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.CONV0_3(x_0_3)
        
        x_0_4 = torch.cat([self.upsample_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1)
        x_0_4 = self.CONV0_4(x_0_4)
    
    
        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            out_put4 = self.final_super_0_4(x_0_4)
            return [out_put1, out_put2, out_put3, out_put4]
        else:
            return self.final_super_0_4(x_0_4)
      
    def elastic_net(self, l1_lambda, l2_lambda):
      l1_loss = 0
      l2_loss = 0
      for param in self.parameters():
          l1_loss += torch.abs(param).sum()
          l2_loss += torch.pow(param, 2).sum()
          
      return l1_lambda * l1_loss + l2_lambda * l2_loss
  
 
if __name__ == "__main__":
    from utils import calculate_computation
    print("deep_supervision: False")
    deep_supervision = False
    device = torch.device('cpu')
    inputs = torch.randn((1, 3, 256, 256)).to(device)
    model = UnetPlusPlus(in_channels=3, num_classes=4, deep_supervision=deep_supervision).to(device)
    outputs = model(inputs)
    print(outputs.shape)    
    summary(model, (8, 3, 256, 256))
    calculate_computation(model, input_size=(3, 256, 256), device=device)

    print("deep_supervision: True")
    deep_supervision = True
    model = UnetPlusPlus(in_channels=3, num_classes=4, deep_supervision=deep_supervision).to(device)
    outputs = model(inputs)
    for out in outputs:
      print(out.shape)
 
 