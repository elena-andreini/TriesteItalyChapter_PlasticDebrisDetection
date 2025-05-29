import torch
from torch import nn
import torch.nn.functional as F

# defining default kwargs outside to avoid repetitive instantiation
conv_defaults = {
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "bias": True
    }

def conv(in_channels, out_channels, **kwargs):    
    return nn.Conv2d(in_channels, out_channels, **{**conv_defaults, **kwargs})


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels//ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//ratio, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return F.sigmoid(out)
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = conv(2, 1, kernel_size=kernel_size, padding=kernel_size//2)


    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True).values
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return F.sigmoid(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv(in_channels, out_channels, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv(out_channels, out_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.act = nn.ReLU(inplace=True)
        
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))

        out = self.conv2(out)
        out = self.act(self.bn2(out + residual))

        out = self.ca(out) * out
        out = self.sa(out) * out
        return out
    
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            conv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(.2),
            conv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(.2),
            nn.AvgPool2d(2)
        )

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.ca(x) * x
        return self.sa(x) * x
    


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            conv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(.2),
            conv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(.2)
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.conv(self.upsample(x))
        x = self.ca(x) * x
        return self.sa(x) * x
    


class ResAttenUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down1 = DownSample(in_channels, 32)
        self.down2 = DownSample(32, 64)
        self.down3 = DownSample(64, 128)
        self.down4 = DownSample(128, 256)
        self.down5 = DownSample(256, 512)

        self.btn1 = ResidualBlock(512, 512)
        self.btn2 = ResidualBlock(512, 512)
        self.btn3 = ResidualBlock(512, 512)

        self.up5 = UpSample(512, 256)
        self.up4 = UpSample(512, 128)
        self.up3 = UpSample(256, 64)
        self.up2 = UpSample(128, 32)
        self.up1 = UpSample(64, 32)

        self.head = nn.Conv2d(32, out_channels, kernel_size=1)
        self.cat = lambda a, b: torch.cat([a, b], dim=1)


    def forward(self, x):
        res1 = self.down1(x)
        res2 = self.down2(res1)
        res3 = self.down3(res2)
        res4 = self.down4(res3)
        res5 = self.down5(res4)

        lat = self.btn1(res5)
        lat = self.btn2(lat)
        lat = self.btn3(lat)

        out = self.up5(lat)
        out = self.up4(self.cat(out, res4))
        out = self.up3(self.cat(out, res3))
        out = self.up2(self.cat(out, res2))
        out = self.up1(self.cat(out, res1))
        
        return self.head(out)