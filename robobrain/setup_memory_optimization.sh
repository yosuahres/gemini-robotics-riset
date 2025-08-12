#!/bin/bash

# RoboBrain Memory Optimization Setup Script
# This script helps diagnose and fix CUDA out of memory issues

echo "🔧 RoboBrain Memory Optimization Setup"
echo "======================================="

# Check if we're in the right directory
if [ ! -f "inference.py" ]; then
    echo "❌ Error: Please run this script from the RoboBrain directory"
    exit 1
fi

echo "📋 Step 1: Checking current environment..."

# Check Python and CUDA
python3 -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f'GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)')
except ImportError:
    print('❌ PyTorch not installed')
"

echo ""
echo "📋 Step 2: Installing memory optimization packages..."

# Install additional packages for memory optimization
pip install bitsandbytes accelerate psutil

echo ""
echo "📋 Step 3: Running memory diagnostic..."

# Run the memory check script
python3 gpu_memory_check.py

echo ""
echo "📋 Step 4: Optimization recommendations..."

echo "🎯 To fix CUDA OOM issues, try in this order:"
echo ""
echo "1. 🔹 Use the optimized inference script:"
echo "   python3 inference.py"
echo ""
echo "2. 🔹 If still having issues, use low-memory version:"
echo "   python3 inference_low_memory.py"
echo ""
echo "3. 🔹 For severe memory constraints, modify parameters:"
echo "   - Reduce max_new_tokens to 50-100"
echo "   - Enable 4-bit quantization"
echo "   - Clear GPU cache between runs"
echo ""
echo "4. 🔹 Emergency fallback (CPU inference):"
echo "   python3 -c \"
from inference import SimpleInference
model = SimpleInference('BAAI/RoboBrain', device='cpu')
pred = model.inference('What is shown?', 'path/to/image.jpg', max_new_tokens=50)
print(pred)
\""

echo ""
echo "✅ Setup complete! Try running the optimized inference scripts."
