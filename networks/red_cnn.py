# networks/red_cnn.py
import torch.nn as nn
"""
Total params: 1,848,865
Trainable params: 1,848,865
Non-trainable params: 0
"""
class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)

        self.act = nn.ReLU()

    def forward(self, x):
        # encoder
        residual_1 = x
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        residual_2 = out
        out = self.act(self.conv3(out))
        out = self.act(self.conv4(out))
        residual_3 = out
        out = self.act(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.act(out))
        out = self.tconv3(self.act(out))
        out += residual_2
        out = self.tconv4(self.act(out))
        out = self.tconv5(self.act(out))
        out += residual_1
        out = self.act(out)
        return out