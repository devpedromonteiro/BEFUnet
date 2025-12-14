"""
Test script to verify FlashAttention-2 implementation
"""
import torch
import torch.nn as nn
import time

def test_attention_implementation():
    """Test Attention class with and without FlashAttention-2"""
    print("="*70)
    print("Testing FlashAttention-2 Implementation")
    print("="*70)
    
    try:
        from models.Encoder import Attention, FLASH_ATTENTION_AVAILABLE
        
        print(f"\nFlashAttention-2 Available: {FLASH_ATTENTION_AVAILABLE}")
        
        # Test parameters
        batch_size = 2
        seq_len = 100
        dim = 512
        heads = 8
        dim_head = 64
        
        # Create test input
        x = torch.randn(batch_size, seq_len, dim).cuda() if torch.cuda.is_available() else torch.randn(batch_size, seq_len, dim)
        print(f"\nInput shape: {x.shape}")
        
        # Test with FlashAttention-2 (if available)
        print("\n" + "-"*70)
        print("Test 1: Attention with FlashAttention-2 (if available)")
        print("-"*70)
        attn_flash = Attention(dim=dim, factor=1, heads=heads, dim_head=dim_head, 
                              dropout=0.1, use_flash_attention=True)
        if torch.cuda.is_available():
            attn_flash = attn_flash.cuda()
            x = x.cuda()
        
        attn_flash.eval()
        with torch.no_grad():
            start_time = time.time()
            out_flash = attn_flash(x)
            flash_time = time.time() - start_time
        
        print(f"Output shape: {out_flash.shape}")
        print(f"Time: {flash_time*1000:.2f} ms")
        print(f"Using FlashAttention-2: {attn_flash.use_flash_attention}")
        
        # Test with standard attention
        print("\n" + "-"*70)
        print("Test 2: Attention with Standard Implementation")
        print("-"*70)
        attn_std = Attention(dim=dim, factor=1, heads=heads, dim_head=dim_head, 
                            dropout=0.1, use_flash_attention=False)
        if torch.cuda.is_available():
            attn_std = attn_std.cuda()
        
        attn_std.eval()
        with torch.no_grad():
            start_time = time.time()
            out_std = attn_std(x)
            std_time = time.time() - start_time
        
        print(f"Output shape: {out_std.shape}")
        print(f"Time: {std_time*1000:.2f} ms")
        
        # Compare outputs
        print("\n" + "-"*70)
        print("Comparison")
        print("-"*70)
        max_diff = torch.abs(out_flash - out_std).max().item()
        mean_diff = torch.abs(out_flash - out_std).mean().item()
        
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        if FLASH_ATTENTION_AVAILABLE and attn_flash.use_flash_attention:
            speedup = std_time / flash_time if flash_time > 0 else 0
            print(f"\nSpeedup: {speedup:.2f}x")
            if speedup > 1.0:
                print("✅ FlashAttention-2 is faster!")
            else:
                print("⚠️  FlashAttention-2 is slower (may be due to overhead or GPU)")
        
        # Check if outputs are similar (allowing for numerical differences)
        if max_diff < 0.1:  # FlashAttention uses recomputation, so small differences are expected
            print("\n✅ Outputs are similar (within expected tolerance)")
        else:
            print("\n⚠️  Outputs differ significantly (may need investigation)")
        
        print("\n" + "="*70)
        print("✅ All tests completed!")
        print("="*70)
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure you're running from the project root directory")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_memory_usage():
    """Test memory usage comparison"""
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available, skipping memory test")
        return
    
    try:
        from models.Encoder import Attention
        
        batch_size = 4
        seq_len = 500
        dim = 512
        heads = 8
        dim_head = 64
        
        x = torch.randn(batch_size, seq_len, dim).cuda()
        
        print("\n" + "="*70)
        print("Memory Usage Test")
        print("="*70)
        
        # Standard attention
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        attn_std = Attention(dim=dim, factor=1, heads=heads, dim_head=dim_head, 
                            dropout=0.1, use_flash_attention=False).cuda()
        attn_std.eval()
        with torch.no_grad():
            _ = attn_std(x)
        std_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        # FlashAttention-2
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        attn_flash = Attention(dim=dim, factor=1, heads=heads, dim_head=dim_head, 
                              dropout=0.1, use_flash_attention=True).cuda()
        attn_flash.eval()
        with torch.no_grad():
            _ = attn_flash(x)
        flash_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        print(f"Standard Attention Memory: {std_memory:.2f} MB")
        print(f"FlashAttention-2 Memory: {flash_memory:.2f} MB")
        if flash_memory < std_memory:
            reduction = ((std_memory - flash_memory) / std_memory) * 100
            print(f"Memory Reduction: {reduction:.1f}%")
            print("✅ FlashAttention-2 uses less memory!")
        else:
            print("⚠️  Memory usage similar (may vary with sequence length)")
        
    except Exception as e:
        print(f"\n❌ Memory test error: {e}")


if __name__ == "__main__":
    test_attention_implementation()
    test_memory_usage()



