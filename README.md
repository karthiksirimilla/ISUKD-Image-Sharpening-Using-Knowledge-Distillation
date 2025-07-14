# ISUKD - Image Sharpening Using Knowledge Distillation

A Streamlit web application that demonstrates image sharpening using knowledge distillation with teacher-student neural networks.

## Features

- **Image Upload**: Upload your own images for processing
- **Sample Images**: Test with pre-loaded sample images
- **Dual Model Processing**: Compare results from both student and teacher models
- **Performance Metrics**: View PSNR and SSIM metrics for quality assessment
- **Download Results**: Save processed images locally
- **Real-time Processing**: Fast inference with PyTorch models

## Installation

1. **Clone the repository**:
   ```bash
   git clone [<repository-url>](https://github.com/karthiksirimilla/ISUKD-Image-Sharpening-Using-Knowledge-Distillation)
   cd ISUKD-Image-Sharpening-Using-Knowledge-Distillation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch** (if not already installed):
   ```bash
   # For CPU only
   pip install torch torchvision
   
   # For CUDA support (if you have a GPU)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

## Usage

1. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. **Select input source**:
   - Upload an image from your device
   - Use one of the sample images provided

4. **Click "Enhance Image"** to process the image with both student and teacher models

5. **View results**:
   - Side-by-side comparison of input, student output, and teacher output
   - Performance metrics (PSNR and SSIM)
   - Download processed images

## Project Structure

```
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── models/                         # Neural network models
│   ├── student_model.py           # Student model architecture
│   ├── teacher_model.py           # Teacher model architecture
│   ├── student_model.pth          # Trained student model weights
│   └── teacher_model.pth          # Trained teacher model weights
├── utils/                          # Utility functions
│   ├── model_loader.py            # Model loading utilities
│   ├── image_processing.py        # Image preprocessing/postprocessing
│   └── metrics.py                 # Performance metrics calculation
├── static/                         # Static assets
│   └── sample_images/             # Sample images for testing
└── Restormer/                      # Restormer implementation (external)
```

## Model Architecture

### Student Model
- Lightweight CNN-based architecture
- Designed for real-time video processing
- Uses depthwise separable convolutions and attention mechanisms

### Teacher Model
- Transformer-based architecture
- Higher computational complexity but better quality
- Uses multi-head attention and residual connections

## Performance Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality in decibels
- **SSIM (Structural Similarity Index)**: Measures perceptual similarity (0-1 scale)

## Troubleshooting

### Common Issues

1. **Model files not found**: The application will use untrained models if `.pth` files are missing
2. **CUDA out of memory**: Try processing smaller images or use CPU mode
3. **Import errors**: Ensure all dependencies are installed correctly

### System Requirements

- Python 3.7+
- 4GB+ RAM (8GB+ recommended)
- GPU with CUDA support (optional, for faster processing)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the Restormer architecture for image restoration
- Uses knowledge distillation techniques for model compression
- Built with Streamlit for easy web deployment 
