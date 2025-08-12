#!/usr/bin/env python3
"""
GPU Memory Monitor and Optimization Helper
Use this script to diagnose CUDA memory issues and optimize settings.
"""

import torch
import gc
import os
import subprocess
import sys

def check_gpu_status():
    """Check basic GPU information and memory status"""
    print("=== GPU Status Check ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"🔧 CUDA version: {torch.version.cuda}")
    print(f"🏷️ PyTorch version: {torch.__version__}")
    
    # GPU device info
    device_count = torch.cuda.device_count()
    print(f"🎮 Number of GPUs: {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Multiprocessor count: {props.multi_processor_count}")
    
    return True

def check_memory_usage():
    """Check current GPU memory usage"""
    print("\n=== Memory Usage ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        
        # Current memory
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        cached = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        free = total - allocated
        
        print(f"  📊 Allocated: {allocated:.2f} GB ({allocated/total*100:.1f}%)")
        print(f"  🗂️ Cached: {cached:.2f} GB ({cached/total*100:.1f}%)")
        print(f"  🆓 Free: {free:.2f} GB ({free/total*100:.1f}%)")
        print(f"  📈 Total: {total:.2f} GB")
        
        # Memory stats if available
        try:
            peak = torch.cuda.max_memory_allocated(i) / 1024**3
            print(f"  🔝 Peak allocated: {peak:.2f} GB")
        except:
            pass

def clear_gpu_memory():
    """Clear GPU memory cache"""
    print("\n=== Clearing GPU Memory ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    # Clear PyTorch cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    print("✅ GPU memory cache cleared")
    check_memory_usage()

def check_system_memory():
    """Check system RAM usage"""
    print("\n=== System Memory ===")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"📊 Total RAM: {memory.total / 1024**3:.2f} GB")
        print(f"🆓 Available RAM: {memory.available / 1024**3:.2f} GB")
        print(f"📈 Used RAM: {memory.used / 1024**3:.2f} GB ({memory.percent:.1f}%)")
    except ImportError:
        print("⚠️ psutil not available. Install with: pip install psutil")

def get_nvidia_smi():
    """Get nvidia-smi output"""
    print("\n=== NVIDIA-SMI Output ===")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("❌ nvidia-smi not available or failed")
    except FileNotFoundError:
        print("❌ nvidia-smi command not found")

def memory_optimization_tips():
    """Print memory optimization suggestions"""
    print("\n=== Memory Optimization Tips ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"💡 For {total_memory:.0f}GB GPU memory:")
    
    if total_memory < 4:
        print("  🔴 Very Low Memory (<4GB):")
        print("    - Use 4-bit quantization")
        print("    - Set max_new_tokens=20-50")
        print("    - Consider CPU inference")
        print("    - Use model sharding")
        
    elif total_memory < 8:
        print("  🟡 Low Memory (4-8GB):")
        print("    - Use 8-bit quantization")
        print("    - Set max_new_tokens=50-100")
        print("    - Enable gradient checkpointing")
        print("    - Use device_map='auto'")
        
    elif total_memory < 16:
        print("  🟢 Medium Memory (8-16GB):")
        print("    - Use float16 precision")
        print("    - Set max_new_tokens=100-250")
        print("    - Enable memory optimizations")
        
    else:
        print("  🟢 High Memory (>16GB):")
        print("    - Can use full precision")
        print("    - Set max_new_tokens=250+")
        print("    - No special optimizations needed")
    
    print("\n📋 General optimizations:")
    print("  - Clear cache between inferences")
    print("  - Use torch.no_grad() context")
    print("  - Set use_cache=False")
    print("  - Enable attention slicing")
    print("  - Use low_cpu_mem_usage=True")

def test_memory_allocation():
    """Test memory allocation to find limits"""
    print("\n=== Memory Allocation Test ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    clear_gpu_memory()
    
    print("Testing memory allocation limits...")
    
    # Test different tensor sizes
    sizes_gb = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    
    for size_gb in sizes_gb:
        try:
            # Calculate tensor size for target GB
            elements = int(size_gb * 1024**3 / 4)  # 4 bytes per float32
            
            print(f"  Testing {size_gb:.1f}GB allocation... ", end="")
            tensor = torch.randn(elements, dtype=torch.float32, device='cuda')
            print("✅ Success")
            
            # Clean up
            del tensor
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print("❌ Failed")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

def main():
    """Main diagnostic function"""
    print("🔍 RoboBrain CUDA Memory Diagnostic Tool")
    print("=" * 50)
    
    # Basic checks
    if not check_gpu_status():
        print("\n❌ No CUDA GPU available. Please check your setup.")
        return
    
    # Memory status
    check_memory_usage()
    check_system_memory()
    
    # System info
    get_nvidia_smi()
    
    # Clear memory and recheck
    clear_gpu_memory()
    
    # Run memory test
    test_memory_allocation()
    
    # Provide recommendations
    memory_optimization_tips()
    
    print("\n" + "=" * 50)
    print("🎯 Diagnostic complete!")
    print("\nTo fix CUDA OOM issues:")
    print("1. Use the optimized inference_low_memory.py script")
    print("2. Reduce max_new_tokens parameter")
    print("3. Enable quantization (4-bit or 8-bit)")
    print("4. Clear GPU memory between runs")
    print("5. Close other GPU applications")

if __name__ == "__main__":
    main()
