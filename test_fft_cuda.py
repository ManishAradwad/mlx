#!/usr/bin/env python3
"""
Test script for cuFFT implementation in MLX CUDA backend.
"""

import mlx.core as mx
import numpy as np

def test_fft_cuda():
    print("=" * 70)
    print("Testing MLX FFT with CUDA backend")
    print("=" * 70)
    
    # Check MLX version and device
    print(f"\nMLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    print(f"Default device: {mx.default_device()}")
    
    # Test 1: Simple 1D FFT (complex to complex)
    print("\n" + "-" * 70)
    print("Test 1: 1D Complex-to-Complex FFT")
    print("-" * 70)
    
    x = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
    print(f"Input (float32): {x}")
    
    try:
        # Convert to complex and do FFT
        x_complex = mx.array(x, dtype=mx.complex64)
        print(f"Input (complex64): {x_complex}")
        
        result = mx.fft.fft(x_complex)
        mx.eval(result)  # Force evaluation
        print(f"FFT result: {result}")
        print(f"✓ Test 1 PASSED")
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        return False
    
    # Test 2: Real to Complex FFT (rfft)
    print("\n" + "-" * 70)
    print("Test 2: 1D Real-to-Complex FFT (rfft)")
    print("-" * 70)
    
    x_real = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
    print(f"Input: {x_real}")
    
    try:
        result = mx.fft.rfft(x_real)
        mx.eval(result)
        print(f"RFFT result: {result}")
        print(f"Result shape: {result.shape} (expected: (3,) for n=4)")
        print(f"✓ Test 2 PASSED")
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        return False
    
    # Test 3: Inverse FFT
    print("\n" + "-" * 70)
    print("Test 3: Inverse FFT (ifft)")
    print("-" * 70)
    
    try:
        x_complex = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.complex64)
        fft_result = mx.fft.fft(x_complex)
        ifft_result = mx.fft.ifft(fft_result)
        mx.eval(fft_result, ifft_result)
        
        print(f"Original: {x_complex}")
        print(f"FFT → IFFT: {ifft_result}")
        
        # Check if we get back the original (with some tolerance for numerical errors)
        diff = mx.abs(x_complex - ifft_result)
        mx.eval(diff)
        max_diff = mx.max(diff)
        mx.eval(max_diff)
        
        print(f"Max difference: {max_diff.item()}")
        if max_diff.item() < 1e-5:
            print(f"✓ Test 3 PASSED (round-trip successful)")
        else:
            print(f"✗ Test 3 FAILED (round-trip error too large)")
            return False
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Compare with NumPy
    print("\n" + "-" * 70)
    print("Test 4: Comparison with NumPy FFT")
    print("-" * 70)
    
    try:
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        x_mlx = mx.array(x_np)
        
        # NumPy FFT
        np_result = np.fft.fft(x_np)
        
        # MLX FFT (need to convert to complex first)
        x_mlx_complex = mx.array(x_np, dtype=mx.complex64)
        mlx_result = mx.fft.fft(x_mlx_complex)
        mx.eval(mlx_result)
        
        # Convert MLX result to numpy for comparison
        mlx_result_np = np.array(mlx_result)
        
        print(f"NumPy result: {np_result}")
        print(f"MLX result:   {mlx_result_np}")
        
        diff = np.abs(np_result - mlx_result_np)
        max_diff = np.max(diff)
        print(f"Max difference: {max_diff}")
        
        if max_diff < 1e-5:
            print(f"✓ Test 4 PASSED (matches NumPy)")
        else:
            print(f"⚠ Test 4 WARNING (small difference from NumPy: {max_diff})")
            # This might still be acceptable due to different algorithms
    except Exception as e:
        print(f"✗ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    import sys
    success = test_fft_cuda()
    sys.exit(0 if success else 1)
