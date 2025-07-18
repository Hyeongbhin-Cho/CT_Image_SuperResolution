# networks/dual_unet.py
import torch
import torch.nn as nn
"""
Total params: 2,300,983
Trainable params: 2,300,983
Non-trainable params: 0
"""
class ConvBlock(nn.Module):
    def __init__(self, c_in:int, c_out:int, kernel_size:int=3, num_group:int=16):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv3 = nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=1, padding=padding)
        self.act = nn.SiLU()
        self.norm = nn.GroupNorm(num_groups=num_group, num_channels=c_out)
        self.res_conv = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0) if c_in != c_out else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(0.01))
        
    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        
        '''r = self.res_conv(x)
        g = r * self.act(r)
        h += self.alpha * g'''
        h = self.norm(h)   
        
        return h


class DownBlock(nn.Module):
    def __init__(self, c_in:int, c_out:int, kernel_size:int=3, num_group:int=16):
        super(DownBlock, self).__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv3 = nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=1, padding=padding)
        self.act = nn.SiLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(num_groups=num_group, num_channels=c_out)
        self.res_conv = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0) if c_in != c_out else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(0.01))
        
    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        
        '''r = self.res_conv(x)
        g = r * self.act(r)
        h += self.alpha * g'''
        h = self.pool(h)
        h = self.norm(h)   
        
        return h
    

class UpBlock(nn.Module):
    def __init__(self, c_in:int, c_out:int, kernel_size:int=3, num_group:int=16):
        super(UpBlock, self).__init__()
        padding = kernel_size // 2
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv3 = nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=1, padding=padding)
        self.act = nn.SiLU()
        self.norm = nn.GroupNorm(num_groups=num_group, num_channels=c_out)
        self.res_conv = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0) if c_in != c_out else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(0.01))
        
    def forward(self, x):
        x_up = self.upsample(x)
        
        h = self.act(self.conv1(x_up))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        '''r = self.res_conv(x_up)
        g = r * self.act(r)
        h += self.alpha * g'''
        h = self.norm(h)
        
        return h
        
        
class DualUNet(nn.Module):
    def __init__(self):
        super(DualUNet, self).__init__()
        self.down_block1 = DownBlock(1, 64, num_group=16)
        self.down_block2 = DownBlock(64, 64, num_group=16)
        self.down_block3 = DownBlock(64, 128, num_group=32)
        self.up_block1 = UpBlock(128, 64, num_group=16)
        self.up_block2 = UpBlock(64, 64, num_group=16)
        self.up_block3 = UpBlock(64, 64, num_group=16)
        
        
        self.down_block4 = DownBlock(64, 64, num_group=16)
        self.down_block5 = DownBlock(64, 64, num_group=16)
        self.down_block6 = DownBlock(64, 128, num_group=32)
        self.up_block4 = UpBlock(128, 64, num_group=16)
        self.up_block5= UpBlock(64, 64, num_group=16)
        self.up_block6 = UpBlock(64, 1, num_group=1)
        
        self.act = nn.SiLU()
        
        
    def forward(self, x):
        # Unet_1
        # Encoder
        res1 = x
        h = self.down_block1(x)
        res2 = h
        h = self.down_block2(h)
        res3 = h
        h = self.down_block3(h)
        # Decoder
        h = self.up_block1(h)
        res4 = h
        h = self.up_block2(h)
        res5 = h
        h = self.up_block3(h)
        
        # Unet_2
        # Encoder
        h = self.down_block4(h)
        h += res2
        h = self.down_block5(h)
        h += res3
        h = self.down_block6(h)
        # Decoder
        h = self.up_block4(h)
        h += res4
        h = self.up_block5(h)
        h += res5
        h = self.up_block6(h)
        
        h += res1
        h = self.act(h)
        return h