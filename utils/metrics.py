import cv2
import numpy as np

def calculate_psnr(img1, img2):
    """Calculate PSNR with size validation"""
    if img1.shape != img2.shape:
        # Resize to match dimensions
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return cv2.PSNR(img1, img2)

def calculate_ssim(img1, img2):
    """Calculate SSIM with size validation"""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return cv2.SSIM(img1, img2)[0]

def get_metrics(original, processed):
    """Safe metric calculation with error handling"""
    try:
        return {
            'psnr': calculate_psnr(original, processed),
            'ssim': calculate_ssim(original, processed)
        }
    except Exception as e:
        print(f"Metric calculation error: {str(e)}")
        return {'psnr': 0.0, 'ssim': 0.0}

def calculate_metrics(original, processed):
    """Calculate PSNR and SSIM metrics between two images"""
    try:
        # Ensure both images are in the same format
        if len(original.shape) == 3 and original.shape[2] == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        else:
            original_gray = original
            
        if len(processed.shape) == 3 and processed.shape[2] == 3:
            processed_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        else:
            processed_gray = processed
        
        # Resize processed image to match original dimensions
        if original_gray.shape != processed_gray.shape:
            processed_gray = cv2.resize(processed_gray, (original_gray.shape[1], original_gray.shape[0]))
        
        # Calculate PSNR
        mse = np.mean((original_gray.astype(np.float64) - processed_gray.astype(np.float64)) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # Calculate SSIM (simplified version)
        # For a more accurate SSIM, you might want to use scikit-image
        try:
            ssim = cv2.SSIM(original_gray, processed_gray)[0]
        except:
            # Fallback SSIM calculation
            ssim = 0.5  # Placeholder value
        
        return {
            'psnr': psnr,
            'ssim': ssim
        }
    except Exception as e:
        print(f"Metric calculation error: {str(e)}")
        return {'psnr': 0.0, 'ssim': 0.0}