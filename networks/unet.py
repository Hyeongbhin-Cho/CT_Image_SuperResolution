# networks/unet.py
import torch
import torch.nn as nn
"""
Total params: 23,375,041
Trainable params: 23,375,041
Non-trainable params: 0
"""

# Block    
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, num_groups=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=kernel_size//2, stride=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=c_out)
        self.act = nn.SiLU()
        
    def forward(self, x):
        h = self.conv1(x)
        h = self.norm(h)
        h = self.act(h)
        
        return h    
        
class UNet(nn.Module):
    def __init__(self, nch_ker=64):
        super(UNet, self).__init__()

        self.nch_in = 1
        self.nch_out = 1
        self.nch_ker = nch_ker

        """
        Encoder part
        """

        self.enc1_1 = ConvBlock(1 * self.nch_in, 1 * self.nch_ker, kernel_size=3, num_groups=16)
        self.enc1_2 = ConvBlock(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, num_groups=16)
    
        self.pool1 = nn.AvgPool2d(2, 2)

        self.enc2_1 = ConvBlock(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, num_groups=32)
        self.enc2_2 = ConvBlock(2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, num_groups=32)

        self.pool2 = nn.AvgPool2d(2, 2)

        self.enc3_1 = ConvBlock(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, num_groups=64)
        self.enc3_2 = ConvBlock(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, num_groups=64)

        self.pool3 = nn.AvgPool2d(2, 2)

        self.enc4_1 = ConvBlock(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, num_groups=128)
        self.enc4_2 = ConvBlock(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, num_groups=128)

        self.pool4 = nn.AvgPool2d(2, 2)

        self.enc5_1 = ConvBlock(8 * self.nch_ker, 16 * self.nch_ker, kernel_size=3, num_groups=256)

        """
        Decoder part
        """
        self.dec5_1 = ConvBlock(16 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, num_groups=128)

        self.unpool4 = nn.ConvTranspose2d(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=2, stride=2)

        self.dec4_2 = ConvBlock(16 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, num_groups=128)
        self.dec4_1 = ConvBlock(8 * self.nch_ker,  4 * self.nch_ker, kernel_size=3, num_groups=64)

        self.unpool3 = nn.ConvTranspose2d(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=2, stride=2)

        self.dec3_2 = ConvBlock(8 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, num_groups=64)
        self.dec3_1 = ConvBlock(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, num_groups=32)

        self.unpool2 = nn.ConvTranspose2d(2 * self.nch_ker, 2 * self.nch_ker, kernel_size=2, stride=2)

        self.dec2_2 = ConvBlock(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, num_groups=32)
        self.dec2_1 = ConvBlock(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, num_groups=16)

        self.unpool1 = nn.ConvTranspose2d(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=2, stride=2)

        self.dec1_2 = ConvBlock(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, num_groups=16)
        self.dec1_1 = ConvBlock(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, num_groups=16)

        self.fc = nn.Conv2d(1 * self.nch_ker, 1 * self.nch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        """
        Encoder part
        """

        enc1 = self.enc1_2(self.enc1_1(x))
        pool1 = self.pool1(enc1)

        enc2 = self.enc2_2(self.enc2_1(pool1))
        pool2 = self.pool2(enc2)

        enc3 = self.enc3_2(self.enc3_1(pool2))
        pool3 = self.pool3(enc3)

        enc4 = self.enc4_2(self.enc4_1(pool3))
        pool4 = self.pool4(enc4)

        enc5 = self.enc5_1(pool4)

        """
        Encoder part
        """
        dec5 = self.dec5_1(enc5)

        unpool4 = self.unpool4(dec5)
        cat4 = torch.cat([enc4, unpool4], dim=1)
        dec4 = self.dec4_1(self.dec4_2(cat4))

        unpool3 = self.unpool3(dec4)
        cat3 = torch.cat([enc3, unpool3], dim=1)
        dec3 = self.dec3_1(self.dec3_2(cat3))

        unpool2 = self.unpool2(dec3)
        cat2 = torch.cat([enc2, unpool2], dim=1)
        dec2 = self.dec2_1(self.dec2_2(cat2))

        unpool1 = self.unpool1(dec2)
        cat1 = torch.cat([enc1, unpool1], dim=1)
        dec1 = self.dec1_1(self.dec1_2(cat1))

        x = self.fc(dec1)

        return x