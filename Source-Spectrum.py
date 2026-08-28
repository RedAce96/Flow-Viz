import numpy as np
import matplotlib.pyplot as plt

# === PARAMETERS (from your inputs-laser) ===
E_pulse = 100.0e2          # erg/cm  (= 10,000)
pulse_dur = 10e-9          # s (FWHM)
sigma_t = pulse_dur / 2.355  # standard deviation
t_center = 0.5 * 1e-7      # s (= 50 ns, center of the single pulse)
dt = 2e-10                 # s (simulation timestep)
T_rec = 10e-6               # s (record length for FFT)
nyquist_freq = 0.5 / dt    # Hz

# === ANALYTICAL SPECTRUM ===
# The magnitude spectrum of total power P(t)
f = np.linspace(0, 200e6, 10000)  # 0 to 200 MHz
P_spectrum = E_pulse * np.exp(-2.0 * np.pi**2 * sigma_t**2 * f**2)

# === NUMERICAL VERIFICATION (sample the actual truncated pulse) ===
t = np.arange(0, T_rec, dt)
pulse = np.zeros_like(t)
# Active window: |t - t_center| <= 4*sigma_t
active = np.abs(t - t_center) <= 4.0 * sigma_t
pulse[active] = np.exp(-0.5 * ((t[active] - t_center) / sigma_t)**2)
# Normalize so integral = E_pulse
P_total = E_pulse * pulse / (sigma_t * np.sqrt(2.0 * np.pi))

# FFT
P_fft = np.fft.rfft(P_total)
freqs = np.fft.rfftfreq(len(t), d=dt)
P_fft_mag = np.abs(P_fft) * dt  # scale by dt for proper energy density

# === PLOT ===
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Time domain
axes[0].plot(t * 1e9, P_total)
axes[0].axvline(t_center * 1e9, color='r', ls='--', label='Pulse center')
axes[0].set_xlabel('Time [ns]')
axes[0].set_ylabel('Power [erg/(cm·s)]')
axes[0].set_title('Source Power vs Time (Single Pulse)')
axes[0].legend()

# Frequency domain
axes[1].semilogy(f * 1e-6, P_spectrum, 'b-', lw=2, label='Analytical (ideal Gaussian)')
axes[1].semilogy(freqs * 1e-6, P_fft_mag, 'r--', alpha=0.7, label='Numerical (truncated)')
axes[1].axvline(44, color='g', ls=':', label='~44 MHz (half-power)')
axes[1].set_xlabel('Frequency [MHz]')
axes[1].set_ylabel('Spectral Magnitude [erg/cm]')
axes[1].set_title('Source Spectrum (Single 10 ns Pulse)')
axes[1].set_xlim([0, 200])
axes[1].legend()

plt.tight_layout()
plt.savefig('source_spectrum_single_pulse.png', dpi=150)
plt.show()

# Print key numbers
print(f"Pulse energy:      {E_pulse:.2e} erg/cm")
print(f"Pulse FWHM:        {pulse_dur:.2e} s")
print(f"Sigma_t:           {sigma_t:.2e} s")
print(f"3dB bandwidth:     ~{np.sqrt(np.log(2)/(2*np.pi**2*sigma_t**2))*1e-6:.1f} MHz")
print(f"Nyquist frequency: {nyquist_freq*1e-6:.1f} MHz")
print(f"Temporal integral: {sigma_t*np.sqrt(2*np.pi):.2e} s")
