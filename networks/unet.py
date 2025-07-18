# networks/unet.py
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
        
        
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down_block1 = DownBlock(32, 32, num_group=8)
        self.down_block2 = DownBlock(32, 32, num_group=8)
        self.down_block3 = DownBlock(64, 64, num_group=16)
        self.down_block4 = DownBlock(64, 64, num_group=16)
        self.down_block5 = DownBlock(128, 128, num_group=32)
        
        self.up_block1 = UpBlock(128, 128, num_group=32)
        self.up_block2 = UpBlock(64, 64, num_group=16)
        self.up_block3 = UpBlock(64, 32, num_group=8)
        self.up_block4 = UpBlock(32, 32, num_group=8)
        self.up_block5 = UpBlock(32, 32, num_group=8)
        self.act = nn.SiLU()
        
        
    def forward(self, x):
        # encoder
        r0 = x
        h = self.down_block1(h)
        r1 = h
        h = self.down_block2(h)
        r2 = h
        h = self.down_block3(h)
        r3 = h
        h = self.down_block4(h)
        r4 = h
        h = self.down_block5(h)

        # decoder
        h = self.up_block1(h)
        h += r4
        h = self.up_block2(h)        
        h += r3
        h = self.up_block3(h)
        h += r2
        h = self.up_block4(h)
        h += r1
        h = self.up_block5(h)
        h += r0
        h = self.act(h)
        return h