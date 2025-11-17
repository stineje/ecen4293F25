import time
import torch

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type != "cuda":
    print("No CUDA device available — this script is meant for a GPU (A100).")
    exit(0)

# Try to enable TF32 (Ampere+)
tf32_available = hasattr(torch, "set_float32_matmul_precision")
if tf32_available:
    # "high" => uses TF32 on Ampere GPUs for float32 matmul
    torch.set_float32_matmul_precision("high")
    print("TF32 (float32 matmul on Tensor Cores) enabled where supported.")
else:
    print("TF32 API not available in this torch version; skipping TF32 mode.")

# Matrix sizes to sweep
sizes = [1024, 2048, 4096, 8192]  # adjust as you like

# Number of repeats for timing (per size / per mode)
REPEATS = 3

results = []

def time_matmul_cpu(n):
    x = torch.randn(n, n, dtype=torch.float32)
    # Warm-up
    _ = torch.matmul(x, x)
    start = time.time()
    for _ in range(REPEATS):
        _ = torch.matmul(x, x)
    end = time.time()
    return (end - start) / REPEATS

def time_matmul_gpu_fp32(n):
    x = torch.randn(n, n, dtype=torch.float32, device=device)
    torch.cuda.synchronize()
    # Warm-up
    _ = torch.matmul(x, x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(REPEATS):
        _ = torch.matmul(x, x)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) / REPEATS

def time_matmul_gpu_fp16(n):
    x = torch.randn(n, n, dtype=torch.float16, device=device)
    torch.cuda.synchronize()
    # Warm-up
    _ = torch.matmul(x, x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(REPEATS):
        _ = torch.matmul(x, x)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) / REPEATS

print("\nSize | CPU FP32 (s) | GPU FP32 (s) | GPU TF32 (s) | GPU FP16 (s) | "
      "Speedup FP32 | Speedup TF32 | Speedup FP16")
print("-"*100)

for n in sizes:
    # CPU FP32
    t_cpu = time_matmul_cpu(n)

    # GPU FP32
    t_gpu_fp32 = time_matmul_gpu_fp32(n)

    # GPU "TF32" – same as FP32 timing, but we conceptually call it TF32 on Ampere
    # If tf32_available is False, skip this and reuse FP32 time (or mark as None).
    if tf32_available:
        # With set_float32_matmul_precision('high'), this is TF32 under the hood.
        t_gpu_tf32 = time_matmul_gpu_fp32(n)
    else:
        t_gpu_tf32 = None

    # GPU FP16 (Tensor Cores)
    t_gpu_fp16 = time_matmul_gpu_fp16(n)

    # Speedups vs CPU FP32
    speedup_fp32 = t_cpu / t_gpu_fp32
    speedup_tf32 = t_cpu / t_gpu_tf32 if t_gpu_tf32 is not None else float('nan')
    speedup_fp16 = t_cpu / t_gpu_fp16

    results.append({
        "n": n,
        "cpu_fp32": t_cpu,
        "gpu_fp32": t_gpu_fp32,
        "gpu_tf32": t_gpu_tf32,
        "gpu_fp16": t_gpu_fp16,
        "speedup_fp32": speedup_fp32,
        "speedup_tf32": speedup_tf32,
        "speedup_fp16": speedup_fp16,
    })

    print(f"{n:4d} | "
          f"{t_cpu:11.5f} | "
          f"{t_gpu_fp32:11.5f} | "
          f"{(t_gpu_tf32 if t_gpu_tf32 is not None else 0):11.5f} | "
          f"{t_gpu_fp16:11.5f} | "
          f"{speedup_fp32:11.2f} | "
          f"{(speedup_tf32 if t_gpu_tf32 is not None else 0):11.2f} | "
          f"{speedup_fp16:11.2f}")

# Optional: Plot results with matplotlib
try:
    import matplotlib.pyplot as plt

    ns = [r["n"] for r in results]
    cpu_times = [r["cpu_fp32"] for r in results]
    gpu_fp32_times = [r["gpu_fp32"] for r in results]
    gpu_fp16_times = [r["gpu_fp16"] for r in results]
    if tf32_available:
        gpu_tf32_times = [r["gpu_tf32"] for r in results]

    plt.figure()
    plt.loglog(ns, cpu_times, marker='o', label="CPU FP32")
    plt.loglog(ns, gpu_fp32_times, marker='o', label="GPU FP32")
    if tf32_available:
        plt.loglog(ns, gpu_tf32_times, marker='o', label="GPU TF32")
    plt.loglog(ns, gpu_fp16_times, marker='o', label="GPU FP16")
    plt.xlabel("Matrix size n (n x n)")
    plt.ylabel("Time (s)")
    plt.title("CPU vs GPU matmul performance (A100)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

except ImportError:
    print("\nmatplotlib not installed; skipping plot.")
