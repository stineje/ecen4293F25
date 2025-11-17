import torch
import time

# Pick device (will be "cpu" on your machine)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device", device)

# Always create x on CPU first
x = torch.randn(5000, 5000)

# -------------------
# CPU version
# -------------------
start_time = time.time()
_ = torch.matmul(x, x)
end_time = time.time()
print(f"CPU time: {(end_time - start_time):6.5f}s")

# -------------------
# GPU version (only if CUDA is available)
# -------------------
if torch.cuda.is_available():
    # move to GPU
    x_gpu = x.to("cuda")

    # warm up GPU
    _ = torch.matmul(x_gpu, x_gpu)
    torch.cuda.synchronize()

    # CUDA is asynchronous, so use Events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    _ = torch.matmul(x_gpu, x_gpu)
    end.record()

    torch.cuda.synchronize()  # wait for GPU to finish

    gpu_time_s = 0.001 * start.elapsed_time(end)  # ms → s
    print(f"GPU time: {gpu_time_s:6.5f}s")
else:
    print("CUDA not available; skipping GPU timing.")
