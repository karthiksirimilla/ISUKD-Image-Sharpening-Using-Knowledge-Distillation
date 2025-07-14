import cv2
import numpy as np
import torch
from PIL import Image

def load_image(uploaded_file):
    """Load image from file uploader or path"""
    if hasattr(uploaded_file, 'read'):
        image = Image.open(uploaded_file)
    else:
        image = Image.open(uploaded_file)
    img_np = np.array(image)
    
    # Convert to RGB if needed
    if len(img_np.shape) == 2:  # Grayscale
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:  # RGBA
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    return img_np

def preprocess_image(img_np, img_size=512):
    """Convert numpy image to normalized tensor"""
    # Resize if needed
    if img_size is not None:
        img_np = cv2.resize(img_np, (img_size, img_size))
    
    # Convert to tensor and normalize
    tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0)  # Add batch dimension

def postprocess_image(tensor, target_size=None):
    """Convert model output tensor to displayable image"""
    tensor = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    img = (tensor * 255).astype(np.uint8)
    
    # Resize to target dimensions if specified
    if target_size:
        img = cv2.resize(img, target_size)
    return img

def create_comparison(original, student, teacher):
    """Create side-by-side comparison image"""
    # Ensure all images have same height
    h = original.shape[0]
    student = cv2.resize(student, (original.shape[1], h))
    teacher = cv2.resize(teacher, (original.shape[1], h))
    
    # Create horizontal stack
    return np.hstack([original, student, teacher])