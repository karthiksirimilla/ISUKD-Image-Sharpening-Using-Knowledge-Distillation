import torch
import numpy as np
from models.student_model import VideoSharpeningStudent
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = VideoSharpeningStudent()
state = torch.load('models/student_model.pth', map_location='cpu')
result = model.load_state_dict(state, strict=False)
print("Model load_state_dict result:", result)
model.eval()

# Load a test image
img = np.array(Image.open('static/sample_images/test_image.png').convert('RGB'))
img_tensor = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0) / 255.0

# Run model
with torch.no_grad():
    out = model(img_tensor)
out_np = out.squeeze(0).permute(1,2,0).cpu().numpy()
out_np = np.clip(out_np, 0, 1)

# Show input and output
plt.subplot(1,2,1); plt.imshow(img); plt.title('Input')
plt.subplot(1,2,2); plt.imshow(out_np); plt.title('Model Output')
plt.show()

# Print channel means
print("Input channel means:", img.mean(axis=(0,1)))
print("Output channel means:", out_np.mean(axis=(0,1))) 