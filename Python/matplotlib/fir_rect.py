import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters (match your setup)
# ----------------------------
Fs = 8000
N = 67                 # number of taps (odd length recommended)
alpha = (N - 1) // 2    # group delay
n = np.arange(-alpha, alpha + 1)

# Normalized cutoff (to Nyquist): 0.3 == 0.3 * (Fs/2) = 0.15*Fs = 1200 Hz
fc = 0.3

# ----------------------------
# Ideal lowpass (rectangular truncation; no window)
# MATLAB's sinc = sin(pi x)/(pi x) => np.sinc uses same convention.
# h[n] = fc * sinc(fc * n)
# ----------------------------
b_rect = fc * np.sinc(fc * n)

# Normalize DC gain to 1 (optional but typical)
h = b_rect / np.sum(b_rect)

# Save taps (like writematrix)
np.savetxt("fir_rect.coe", h, fmt="%.10f")

# Print first 15 taps
print("First 15 taps of h (rect, DC-normalized):")
print(np.array2string(h[:15], precision=10, suppress_small=False))

# ----------------------------
# Frequency response (FFT)
# ----------------------------


def freq_response(h, fs, nfft=8192):
    H = np.fft.rfft(h, n=nfft)
    f = np.linspace(0, fs/2, H.size)
    mag_db = 20*np.log10(np.maximum(np.abs(H), 1e-12))
    return f, mag_db


f, Hdb = freq_response(h, Fs)

plt.figure(figsize=(8, 4.5))
plt.plot(f, Hdb, label="Rectangular-truncated ideal LPF")
plt.title("FIR Lowpass (No Window) — Magnitude Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True, ls=":")
plt.legend()
plt.tight_layout()
plt.show()

# Zoom near the cutoff (fc * Nyquist = 0.3 * (Fs/2) = 1200 Hz)
plt.figure(figsize=(8, 4.5))
plt.plot(f, Hdb, label="Rectangular")
plt.xlim(0, 2400)
plt.ylim(-100, 5)
plt.axvline(0.3*(Fs/2), linestyle="--", label="Cutoff ≈ 1200 Hz")
plt.title("Zoom Near Cutoff")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True, ls=":")
plt.legend()
plt.tight_layout()
plt.show()
