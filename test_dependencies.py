#!/usr/bin/env python3

"""
Test script to verify all dependencies are available for object tracking
"""

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

try:
    from scipy.spatial.distance import cdist
    print("✓ SciPy spatial distance imported successfully")
except ImportError as e:
    print(f"✗ SciPy import failed: {e}")

try:
    from collections import deque
    print("✓ Collections deque imported successfully")
except ImportError as e:
    print(f"✗ Collections deque import failed: {e}")

try:
    import cv2
    print("✓ OpenCV imported successfully")
except ImportError as e:
    print(f"✗ OpenCV import failed: {e}")

# Test basic functionality
try:
    # Test numpy array operations
    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[5, 6], [7, 8]])
    distance_matrix = cdist(arr1, arr2)
    print(f"✓ Distance calculation test successful: {distance_matrix.shape}")
    
    # Test deque operations
    history = deque(maxlen=5)
    history.append([1, 2])
    history.append([3, 4])
    print(f"✓ Deque operations test successful: {len(history)} items")
    
    print("\n🎉 All dependencies are working correctly!")
    print("You can now run the spatial understanding camera API with object tracking.")
    
except Exception as e:
    print(f"✗ Functionality test failed: {e}")
    print("Please install missing dependencies using: pip install -r requirements.txt")
