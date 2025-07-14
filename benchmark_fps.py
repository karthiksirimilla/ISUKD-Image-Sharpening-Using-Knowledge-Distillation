import torch
import time
from models.student_model import VideoSharpeningStudent

# Device selection for Apple Silicon (MPS) or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon Metal (MPS) backend.")
else:
    device = torch.device("cpu")
    print("Using CPU backend.")

model = VideoSharpeningStudent().to(device)
model.eval()
dummy = torch.rand(1, 3, 256, 256).to(device)

# Warmup
for _ in range(5):
    with torch.no_grad():
        _ = model(dummy)

# Timing
N = 30  # Safe number of inferences
start = time.time()
with torch.no_grad():
    for _ in range(N):
        _ = model(dummy)
end = time.time()
fps = N / (end - start)
print(f"FPS: {fps:.2f}") 