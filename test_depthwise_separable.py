"""
Test script to verify Depthwise Separable Convolution implementation
"""
import torch
from models.depthwise_separable import DepthwiseSeparableConv2d
from models.Decoder import ConvUpsample, SegmentationHead

def test_depthwise_separable():
    """Test DepthwiseSeparableConv2d"""
    print("Testing DepthwiseSeparableConv2d...")
    
    # Create test input
    x = torch.randn(2, 64, 56, 56)  # batch=2, channels=64, H=56, W=56
    
    # Standard convolution
    conv_std = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
    out_std = conv_std(x)
    
    # Depthwise separable convolution
    conv_ds = DepthwiseSeparableConv2d(64, 128, kernel_size=3, padding=1, bias=False)
    out_ds = conv_ds(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Standard conv output: {out_std.shape}")
    print(f"Depthwise separable output: {out_ds.shape}")
    
    # Count parameters
    params_std = sum(p.numel() for p in conv_std.parameters())
    params_ds = sum(p.numel() for p in conv_ds.parameters())
    
    print(f"\nStandard Conv2d parameters: {params_std:,}")
    print(f"Depthwise Separable parameters: {params_ds:,}")
    print(f"Reduction: {((params_std - params_ds) / params_std * 100):.2f}%")
    
    assert out_std.shape == out_ds.shape, "Output shapes don't match!"
    print("✅ DepthwiseSeparableConv2d test passed!\n")


def test_conv_upsample():
    """Test ConvUpsample with depthwise separable"""
    print("Testing ConvUpsample...")
    
    x = torch.randn(2, 768, 7, 7)
    
    # With depthwise separable (default)
    conv_up_ds = ConvUpsample(in_chans=768, out_chans=[128, 128, 128], upsample=True, use_depthwise_separable=True)
    out_ds = conv_up_ds(x)
    
    # Without depthwise separable
    conv_up_std = ConvUpsample(in_chans=768, out_chans=[128, 128, 128], upsample=True, use_depthwise_separable=False)
    out_std = conv_up_std(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape (with DS): {out_ds.shape}")
    print(f"Output shape (standard): {out_std.shape}")
    
    params_ds = sum(p.numel() for p in conv_up_ds.parameters())
    params_std = sum(p.numel() for p in conv_up_std.parameters())
    
    print(f"\nConvUpsample parameters (DS): {params_ds:,}")
    print(f"ConvUpsample parameters (std): {params_std:,}")
    print(f"Reduction: {((params_std - params_ds) / params_std * 100):.2f}%")
    
    assert out_ds.shape == out_std.shape, "Output shapes don't match!"
    print("✅ ConvUpsample test passed!\n")


def test_segmentation_head():
    """Test SegmentationHead with depthwise separable"""
    print("Testing SegmentationHead...")
    
    x = torch.randn(2, 16, 224, 224)
    
    # With depthwise separable (default)
    seg_head_ds = SegmentationHead(in_channels=16, out_channels=9, kernel_size=3, use_depthwise_separable=True)
    out_ds = seg_head_ds(x)
    
    # Without depthwise separable
    seg_head_std = SegmentationHead(in_channels=16, out_channels=9, kernel_size=3, use_depthwise_separable=False)
    out_std = seg_head_std(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape (with DS): {out_ds.shape}")
    print(f"Output shape (standard): {out_std.shape}")
    
    params_ds = sum(p.numel() for p in seg_head_ds.parameters())
    params_std = sum(p.numel() for p in seg_head_std.parameters())
    
    print(f"\nSegmentationHead parameters (DS): {params_ds:,}")
    print(f"SegmentationHead parameters (std): {params_std:,}")
    print(f"Reduction: {((params_std - params_ds) / params_std * 100):.2f}%")
    
    assert out_ds.shape == out_std.shape, "Output shapes don't match!"
    print("✅ SegmentationHead test passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("Testing Depthwise Separable Convolution Implementation")
    print("="*60)
    print()
    
    try:
        test_depthwise_separable()
        test_conv_upsample()
        test_segmentation_head()
        
        print("="*60)
        print("✅ All tests passed!")
        print("="*60)
        print("\nThe depthwise separable convolution implementation is working correctly.")
        print("You can now use it in your BEFUnet model for reduced parameters and MACs.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

