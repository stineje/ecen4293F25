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


# build roofline curve
oi_vals = np.logspace(-2, 3, 200)
roof_vals = roofline(oi_vals)

plt.figure()
plt.loglog(oi_vals, roof_vals, label="Roofline")

# plot naive GEMM points
ois_naive = [oi for (_, _, oi) in results_naive]
gflops_naive = [g for (_, g, _) in results_naive]
plt.scatter(ois_naive, gflops_naive, marker="o", label="Naive GEMM")

# optional: NumPy GEMM
ois_np = [oi for (_, _, oi) in results_numpy]
gflops_np = [g for (_, g, _) in results_numpy]
plt.scatter(ois_np, gflops_np, marker="x", label="NumPy GEMM")

plt.xlabel("Operational Intensity (FLOPs/byte)")
plt.ylabel("Performance (GFLOP/s)")
plt.title("Roofline Model for GEMM")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.show()
