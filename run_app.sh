#!/bin/bash

# ISUKD Streamlit Application Launcher

echo "🚀 Starting ISUKD - Image Sharpening Using Knowledge Distillation"
echo "================================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import streamlit, torch, numpy, cv2, PIL" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Some required packages are missing. Installing..."
    pip3 install -r requirements.txt
fi

# Create sample image if it doesn't exist
if [ ! -f "static/sample_images/test_image.png" ]; then
    echo "🖼️  Creating sample test image..."
    python3 -c "
import numpy as np
from PIL import Image
import os
os.makedirs('static/sample_images', exist_ok=True)
img = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(0, 256, 32):
    img[i:i+16, :] = [255, 0, 0]
    img[:, i:i+16] = [0, 255, 0]
for i in range(64, 192, 32):
    for j in range(64, 192, 32):
        img[i:i+16, j:j+16] = [0, 0, 255]
Image.fromarray(img).save('static/sample_images/test_image.png')
print('Sample image created!')
"
fi

echo "✅ All checks passed!"
echo "🌐 Starting Streamlit application..."
echo "📱 Open your browser and go to: http://localhost:8501"
echo ""

# Run the Streamlit application
streamlit run app.py 