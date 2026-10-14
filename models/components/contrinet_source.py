"""Verbatim MFM_0 and MFM from the official ConTriNet implementation.

Upstream: CSer-Tang-hao/ConTriNet_RGBT-SOD
Commit: 9af315d4a93d897cf0b9abb11e194d1ed1072ff9
File: models/ConTriNet.py
MIT license: third_party/contrinet/LICENSE
Only these classes are extracted; class bodies are unchanged.
The complete upstream file is archived at third_party/contrinet/ConTriNet.py.
"""
import torch
import torch.nn as nn

class MFM_0(nn.Module):
    def __init__(self,in_dim, out_dim):
        super(MFM_0, self).__init__()

        act_fn = nn.ReLU(inplace=True)


        self.layer_10 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.layer_11 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            act_fn,
        )

        self.layer_21 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            act_fn,
        )

        self.layer_ful1 = nn.Sequential(
            nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            act_fn,
        )

        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.channel_mul_conv1 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // 16, out_dim, kernel_size=1))
        
        self.channel_mul_conv2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // 16, out_dim, kernel_size=1))
        
    def forward(self, rgb, thermal):
        
        # Cross-guided Feature Enhancement
        x_rgb = self.layer_10(rgb)
        x_the = self.layer_20(thermal)

        rgb_w = self.sigmoid(x_rgb)
        the_w = self.sigmoid(x_the)

        x_rgb_w = rgb.mul(the_w)
        x_the_w = thermal.mul(rgb_w)

        x_rgb_r = x_rgb_w + rgb
        x_the_r = x_the_w + thermal

        x_rgb_r = self.layer_11(x_rgb_r)
        x_the_r = self.layer_21(x_the_r)

        x_rgb_r = x_rgb_r * torch.sigmoid(self.channel_mul_conv1(x_rgb_r))
        x_the_r = x_the_r * torch.sigmoid(self.channel_mul_conv2(x_the_r))
        ful_out = torch.cat((x_rgb_r, x_the_r), dim=1)

        # Attention-aware Feature Fusion
        avgout = torch.mean(ful_out, dim=1, keepdim=True)
        maxout, _ = torch.max(ful_out, dim=1, keepdim=True)
        mask = self.conv2d(torch.cat([avgout, maxout], dim=1))
        mask = self.sigmoid(mask)

        out1 = self.layer_ful1(ful_out) * mask

        return out1


class MFM(nn.Module):
    def __init__(self,in_dim, out_dim):
        super(MFM, self).__init__()

        act_fn = nn.ReLU(inplace=True)

        self.reduc_1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1), 
            act_fn
        )
        self.reduc_2 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1), 
            act_fn
        )

        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

        self.layer_11 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            act_fn,
        )
        self.layer_21 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            act_fn,
        )

        self.layer_ful1 = nn.Sequential(
            nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            act_fn,
        )
        self.layer_ful2 = nn.Sequential(
            nn.Conv2d(out_dim+out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            act_fn,
        )

        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.channel_mul_conv1 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // 16, out_dim, kernel_size=1)
        )
        self.channel_mul_conv2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // 16, out_dim, kernel_size=1)
        )
        
    def forward(self, rgb, thermal, xx):

        # Cross-guided Feature Enhancement
        x_rgb = self.reduc_1(rgb)
        x_the = self.reduc_2(thermal)

        x_rgb1 = self.layer_10(x_rgb)
        x_the1 = self.layer_20(x_the)

        rgb_w = self.sigmoid(x_rgb1)
        the_w = self.sigmoid(x_the1)

        x_rgb_w = x_rgb.mul(the_w)
        x_the_w = x_the.mul(rgb_w)

        x_rgb_r = x_rgb_w + x_rgb
        x_the_r = x_the_w + x_the

        x_rgb_r = self.layer_11(x_rgb_r)
        x_the_r = self.layer_21(x_the_r)

        x_rgb_r = x_rgb_r * torch.sigmoid(self.channel_mul_conv1(x_rgb_r))
        x_the_r = x_the_r * torch.sigmoid(self.channel_mul_conv2(x_the_r))
        ful_out = torch.cat((x_rgb_r, x_the_r), dim=1)

        # Attention-aware Feature Fusion
        avgout = torch.mean(ful_out, dim=1, keepdim=True)
        maxout, _ = torch.max(ful_out, dim=1, keepdim=True)
        mask = self.conv2d(torch.cat([avgout, maxout], dim=1))
        mask = self.sigmoid(mask)

        out1 = self.layer_ful1(ful_out) * mask
        out2 = self.layer_ful2(torch.cat([out1,xx],dim=1))

        return out2




