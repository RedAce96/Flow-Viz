"""Archived standalone FFT implementation.

FFT-Point-Probes.py
===================
Perform 1D FFT analysis on PeleC point-probe time series.

Works with the multi-probe system (probes.cpp) which writes each probe
location to its own file in a ``probes/`` subdirectory.

Usage
-----
Edit the INPUTS section below, then::

    python FFT-Point-Probes.py

Columns in each probe file (CGS units):
    0: time [s]
    1: rho  [g/cm^3]
    2: u    [cm/s]
    3: p    [dyne/cm^2]
    4: T    [K]
    5: x_sample [cm]
    6: y_sample [cm]
    7: level
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# ============================================================
#                          INPUTS
# ============================================================
# Path to the run folder containing inputs_flow / inputs_laser
run_dir = os.path.expanduser(
    "~/Documents/Programming/My-CFD/PeleC/SBLI-Driver/FP-Thermal-Source/TS-2"
)

# Subdirectory where probe files live (set by prob.probe_dir in inputs)
probe_dir = "probes"

# Probe file prefix (set by prob.probe_prefix in inputs)
probe_prefix = "flow_probe"   # or "laser_probe" for Phase 2

# Variable to analyse — 0-based column index in the probe file:
#   0=time, 1=rho, 2=u, 3=p, 4=T, 5=x_sample, 6=y_sample, 7=level
var_col = 4        # T (temperature)

# Skip the first N time steps (transient / spin-up)
nt_skip = 0

# Only analyse the first N probes.  Set to None for all.
max_probes = None   # e.g. 10 for quick testing

# --- Resampling for CFL-based (variable dt) simulations ---
# FFT requires evenly-spaced samples.  If your simulation used
#   pelec.fixed_dt = -1   (CFL-based adaptive timestepping),
# the probe time series will have non-uniform dt.  Set this to
# True to linearly interpolate onto a uniform grid first.
resample_to_uniform = True

# Target dt for resampling [s].  If None, use the median dt.
target_dt = None

# Plot the time trace and FFT of the LAST probe?
plot_last_probe = True

# ============================================================
#                     END OF INPUTS
# ============================================================

# --- helper: parse probe header to extract x,y location ---
def parse_probe_header(filepath):
    """Return (x_cm, y_cm) from the header of a probe file."""
    with open(filepath, "r") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = re.search(r"x=([\d\.eE+\-]+)\s*cm.*y=([\d\.eE+\-]+)\s*cm", line)
            if m:
                return float(m.group(1)), float(m.group(2))
    return None, None


# --- build file list ---
probe_path = os.path.join(run_dir, probe_dir)
pattern = os.path.join(probe_path, f"{probe_prefix}_*.dat")
files = sorted(glob.glob(pattern))

if not files:
    raise FileNotFoundError(f"No probe files found matching: {pattern}")

if max_probes is not None:
    files = files[:max_probes]

n_probes = len(files)
print(f"[I] Found {n_probes} probe files in {probe_path}/")

# --- read all probes ---
probe_data = []    # list of dicts: {x, y, time, signal, dt}
probe_x = []
probe_y = []

for i, fpath in enumerate(files):
    x, y = parse_probe_header(fpath)
    if x is None:
        print(f"[W] Could not parse header for {os.path.basename(fpath)}, skipping")
        continue

    # loadtxt with comments='#' skips header lines automatically
    data = np.loadtxt(fpath, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    time = data[nt_skip:, 0]
    signal = data[nt_skip:, var_col]

    # compute actual dt from the data (robust against adaptive timestepping)
    if len(time) >= 2:
        dt_actual = time[1] - time[0]
    else:
        dt_actual = 0.0

    probe_data.append({
        "x": x,
        "y": y,
        "time": time,
        "signal": signal,
        "dt": dt_actual,
        "filename": os.path.basename(fpath),
    })
    probe_x.append(x)
    probe_y.append(y)

n_valid = len(probe_data)
probe_x = np.array(probe_x)
probe_y = np.array(probe_y)

print(f"[I] Successfully loaded {n_valid} probes")

# --- Resample to uniform time grid (if needed) ---
if resample_to_uniform:
    # Determine common time span
    t_min = max(d["time"][0] for d in probe_data)
    t_max = min(d["time"][-1] for d in probe_data)

    # Use median dt across probes for the target
    dt_vals = [d["dt"] for d in probe_data if d["dt"] > 0]
    if target_dt is None:
        dt_use = np.median(dt_vals)
    else:
        dt_use = target_dt

    time_uniform = np.arange(t_min, t_max, dt_use)
    L = len(time_uniform)
    signal_matrix = np.zeros((L, n_valid))

    print(f"[I] Resampling to uniform grid: t=[{t_min:.3e}, {t_max:.3e}] s, "
          f"dt={dt_use:.3e} s, {L} samples")

    for ip, d in enumerate(probe_data):
        signal_matrix[:, ip] = np.interp(time_uniform, d["time"], d["signal"],
                                         left=d["signal"][0], right=d["signal"][-1])

    dt_actual = dt_use
else:
    # Use raw data — truncate all probes to the same length
    lengths = [d["time"].size for d in probe_data]
    L = int(np.median(lengths))
    signal_matrix = np.zeros((L, n_valid))
    dt_actual = np.median([d["dt"] for d in probe_data if d["dt"] > 0])

    print(f"[I] Using raw data: L={L}, dt={dt_actual:.3e} s (no resampling)")

    for ip, d in enumerate(probe_data):
        n = min(d["signal"].size, L)
        signal_matrix[:n, ip] = d["signal"][:n]

# --- FFT on each probe ---
print("[I] Performing FFT on each probe ...")

Fs = 1.0 / dt_actual
freq = Fs * np.arange(0, L // 2 + 1) / L
n_freq = len(freq)

P1 = np.zeros((n_freq, n_valid))

for ip in range(n_valid):
    sig = signal_matrix[:, ip]
    Y = np.fft.fft(sig)
    P2 = np.abs(Y / L)
    P1[:, ip] = P2[:n_freq]
    P1[1:-1, ip] = 2 * P1[1:-1, ip]

print(f"[I] FFT complete — {n_freq} frequency bins x {n_valid} probes")

# --- plots ---
if plot_last_probe:
    last = probe_data[-1]
    time_last = last["time"]
    sig_last = last["signal"][:L]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(time_last, sig_last)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(f"Signal (col {var_col})")
    ax1.set_title(f"Probe at x={last['x']:.3f} cm, y={last['y']:.4f} cm  ({last['filename']})")
    ax1.grid(True)

    ax2.loglog(freq, P1[:, -1])
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("|Amplitude|")
    ax2.set_title("Single-Sided Amplitude Spectrum")
    ax2.grid(True, which="both", ls=":", alpha=0.3)

    fig.tight_layout()
    plt.show()

print("[I] FINISHED")
