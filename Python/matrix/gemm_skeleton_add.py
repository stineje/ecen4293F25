def gemm_metrics(n, dt, bytes_per_element=8):
    flops = 2 * (n**3)
    gflops_per_s = flops / dt / 1e9
    bytes_moved = 4 * (n**2) * bytes_per_element  # simple model
    oi = flops / bytes_moved
    return gflops_per_s, oi

sizes = [64, 128, 192, 256]
results_naive = []
for n in sizes:
    dt = time_gemm(gemm_naive, n)
    gflops, oi = gemm_metrics(n, dt)
    results_naive.append((n, gflops, oi))
    print(f"n={n}: {gflops:.3f} GF/s, OI={oi:.3f} FLOP/byte")
