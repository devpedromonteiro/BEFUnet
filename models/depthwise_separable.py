"""
Depthwise Separable Convolution Module
Reduces parameters and MACs compared to standard convolution
"""
import torch.nn as nn


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution
    
    Replaces standard Conv2d with:
    1. Depthwise convolution (spatial filtering)
    2. Pointwise convolution (channel mixing)
    
    Reduces parameters by ~1/N where N is the number of output channels
    Reduces MACs significantly for typical kernel sizes (3x3, 5x5)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, dilation=1, bias=True):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        # Depthwise convolution: one filter per input channel
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # Each input channel gets its own filter
            bias=False
        )
        
        # Pointwise convolution: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

