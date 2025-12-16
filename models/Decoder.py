"""
Author: Omid Nejati Manzari
Date: Jun  2023
Modified: Added Depthwise Separable Convolutions for efficiency
"""
import torch.nn as nn
from .depthwise_separable import DepthwiseSeparableConv2d

class ConvUpsample(nn.Module):
    def __init__(self, in_chans=384, out_chans=[128], upsample=True, use_depthwise_separable=True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.use_depthwise_separable = use_depthwise_separable
        
        self.conv_tower = nn.ModuleList()
        for i, out_ch in enumerate(self.out_chans):
            if i > 0: self.in_chans = out_ch
            
            # Use depthwise separable convolution for 3x3 convs (more efficient)
            if self.use_depthwise_separable:
                self.conv_tower.append(DepthwiseSeparableConv2d(
                    self.in_chans, out_ch,
                    kernel_size=3, stride=1,
                    padding=1, bias=False
                ))
            else:
                # Fallback to standard convolution
                self.conv_tower.append(nn.Conv2d(
                    self.in_chans, out_ch,
                    kernel_size=3, stride=1,
                    padding=1, bias=False
                ))
            
            self.conv_tower.append(nn.GroupNorm(32, out_ch))
            self.conv_tower.append(nn.ReLU(inplace=False))
            if upsample:
                self.conv_tower.append(nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False))
            
        self.convs_level = nn.Sequential(*self.conv_tower)
        
    def forward(self, x):
        return self.convs_level(x)


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_depthwise_separable=True):
        # Use depthwise separable for 3x3 convolutions
        if kernel_size == 3 and use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d(
                in_channels, out_channels, 
                kernel_size=kernel_size, 
                padding=kernel_size // 2
            )
        else:
            # For 1x1 or when depthwise separable is disabled, use standard conv
            conv2d = nn.Conv2d(
                in_channels, out_channels, 
                kernel_size=kernel_size, 
                padding=kernel_size // 2
            )
        super().__init__(conv2d)
