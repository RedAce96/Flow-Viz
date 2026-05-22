import os
import numpy as np
from matplotlib import pyplot as plt


RUNS_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../Runs-Folder')
DEFAULT_PROBE_CASES = [
    #{'case': 'Hyp-Lam-FPV4', 'label': 'MFC-FPV4'},
    {'case': 'Hyp-Lam-FPV5', 'label': 'MFC-FPV5-ISO'},
    {'case': 'Hyp-Lam-FPV5-Adia', 'label': 'MFC-FPV5-ADIA'},
]
REFERENCE_CASE = 'Hyp-Lam-FPV5'
FREESTREAM_TEMPERATURE = 125.0
REFERENCE_PRESSURE = 760.0
FLOW_TIMES = 0.1 / 1726


def GrabFolder(PN, db_dir=None):
    db_root = os.path.abspath(os.path.dirname(__file__))
    if db_dir is None:
        db_dir = os.path.join(db_root, '../Runs-Folder/Hyp-Lam-FPV5/D/')
    db_name = f'probe{PN}_prim.dat'
    db_path = os.path.join(db_dir, db_name)

    return db_path


def _case_dir(case_name, *parts):
    return os.path.join(RUNS_ROOT, case_name, *parts)


def _get_plot_dir(case_name=REFERENCE_CASE):
    plot_dir = _case_dir(case_name, 'Probe_Plots')
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def _load_probe_case(pn, case_config):
    case_name = case_config['case']
    db_path = GrabFolder(pn, db_dir=_case_dir(case_name, 'D'))
    db = np.loadtxt(db_path)

    keep_fraction = case_config.get('keep_fraction')
    if keep_fraction is not None:
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction!r} for {case_name}.")
        keep_count = max(1, int(len(db) * keep_fraction))
        db = db[:keep_count]

    time = db[:, 0]
    density = db[:, 1]
    velocity = db[:, 2]
    pressure = db[:, 3]
    temperature = pressure / (density * 287.05)

    return {
        'case': case_name,
        'label': case_config.get('label', case_name),
        'time': time,
        'density': density,
        'velocity': velocity,
        'pressure': pressure,
        'temperature': temperature,
        'pressure_ratio': pressure / REFERENCE_PRESSURE,
        'temperature_ratio': temperature / FREESTREAM_TEMPERATURE,
    }


def _load_probe_cases(pn, case_configs=None):
    if case_configs is None:
        case_configs = DEFAULT_PROBE_CASES
    return [_load_probe_case(pn, case_config) for case_config in case_configs]

def mfcpr(pn,skipNum: int = 0, AutoSkip: bool = True):
    plot_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../Runs-Folder/Hyp-Lam-FPV4/Probe_Plots')
    os.makedirs(plot_dir, exist_ok=True)

    db_path = GrabFolder(pn)
    db = np.loadtxt(db_path)

    if AutoSkip:
        Time_full = db[:, 0]
        for i in range(len(Time_full) - 1):
            # Look for significant time jump backwards (> 0.01s indicates a reset)
            if Time_full[i] - Time_full[i+1] > 0.01:  # Time went backwards significantly
                skipNum = i + 1
                print(f"Auto-skipping {skipNum} initial entries to start at t={Time_full[skipNum]:.7f}s")
                break
        else:
            print("No time reset detected. Using all data.")


    Time     = db[skipNum:, 0]
    Density  = db[skipNum:, 1] 
    Velocity = db[skipNum:, 2] 
    Pressure = db[skipNum:, 3] 

    # Calculate sampling rate and time step
    dt = np.mean(np.diff(Time))
    fs = 1.0 / dt  # Sampling frequency
    N = len(Time)

    
    # Subtract mean for FFT
    Density_fft  = Density - np.mean(Density)
    Velocity_fft = Velocity - np.mean(Velocity)
    Pressure_fft = Pressure - np.mean(Pressure)

    # FFT using numpy
    Den_fft = np.fft.fft(Density_fft)
    Vel_fft = np.fft.fft(Velocity_fft)
    Pres_fft = np.fft.fft(Pressure_fft)
    
    # Frequency array (only positive frequencies)
    freq = np.fft.fftfreq(N, dt)
    positive_freq_idx = freq > 0
    freq_positive = freq[positive_freq_idx]
    
    # Get magnitude of positive frequencies only
    Den_mag = 2.0 * np.abs(Den_fft[positive_freq_idx]) / N
    Vel_mag = 2.0 * np.abs(Vel_fft[positive_freq_idx]) / N
    Pres_mag = 2.0 * np.abs(Pres_fft[positive_freq_idx]) / N
    min_positive_freq = freq_positive[0] if len(freq_positive) > 0 else fs / max(N, 1)

    # Plotting the time series data
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Probe {pn} Time Series', fontsize=20)
    
    ax[0, 0].plot(Time, Density, color='blue', linewidth=0.5)
    ax[0, 1].plot(Time, Velocity, color='orange', linewidth=0.5)
    ax[1, 0].plot(Time, Pressure/760, color='green', linewidth=0.5)

    ax[0, 0].set_title('Density', fontsize=32)
    ax[0, 1].set_title('Velocity', fontsize=32)
    ax[1, 0].set_title('Pressure/p_inf', fontsize=32)
    ax[0, 0].set_xlabel('Time (s)', fontsize=24)
    ax[0, 1].set_xlabel('Time (s)', fontsize=24)
    ax[1, 0].set_xlabel('Time (s)', fontsize=24)
    ax[0, 0].set_ylabel('Density (kg/m³)', fontsize=24)
    ax[0, 1].set_ylabel('Velocity (m/s)', fontsize=24)
    ax[1, 0].set_ylabel('Pressure/p_inf', fontsize=24)
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 1].grid(True, alpha=0.3)
    ax[1, 0].grid(True, alpha=0.3)
    ax[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'fig_probe0{pn}.png'), dpi=150)
    plt.close()

    # FFT Plot
    fig, ax = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Probe {pn} Frequency Spectrum', fontsize=20)
    
    ax[0].plot(freq_positive, Den_mag, color='blue', linewidth=2)
    ax[0].set_ylabel('Density Amp', fontsize=24)
    ax[0].set_xlabel('Frequency (Hz)', fontsize=24)
    ax[0].set_title('Density FFT', fontsize=32)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlim(min_positive_freq, fs/2)
    ax[0].grid(True, alpha=0.3)
    
    ax[1].plot(freq_positive, Vel_mag, color='orange', linewidth=2)
    ax[1].set_ylabel('Velocity Amp', fontsize=24)
    ax[1].set_xlabel('Frequency (Hz)', fontsize=24)
    ax[1].set_title('Velocity FFT', fontsize=32)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlim(min_positive_freq, fs/2)
    ax[1].grid(True, alpha=0.3)
    
    ax[2].plot(freq_positive, Pres_mag, color='green', linewidth=2)
    ax[2].set_ylabel('Pressure Amp', fontsize=24)
    ax[2].set_xlabel('Frequency (Hz)', fontsize=24)
    ax[2].set_title('Pressure FFT', fontsize=32)
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlim(min_positive_freq, fs/2)
    ax[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'fig_probe0{pn}_fft.png'), dpi=150)
    plt.close()
    
    print(f"  Probe {pn}: Sampling rate = {fs:.1f} Hz, Duration = {Time[-1]-Time[0]:.6f} s, N = {N}")

def PT_Comp(pn):
    probe_cases = _load_probe_cases(pn)
    plot_dir = _get_plot_dir()

    Thesispress = np.loadtxt(os.path.join(plot_dir, 'Thesis_FP_Data.csv'), delimiter=',', skiprows=0)
    Thesistemp = np.loadtxt(os.path.join(plot_dir, 'Thesis_FP_DataT.csv'), delimiter=',', skiprows=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    for probe_case in probe_cases:
        ax.plot(
            probe_case['time'] / FLOW_TIMES,
            probe_case['pressure_ratio'],
            label=probe_case['label'],
            linewidth=1.5,
        )
    ax.plot(Thesispress[:,0], Thesispress[:,1], label='Thesis', color='black', linestyle='--')
    ax.set_xlabel(r'$\bar{t}$', fontsize=16)
    ax.set_ylabel(r'$P/P_\infty$', fontsize=16)
    ax.set_title(f'Probe {pn} Pressure Comparison', fontsize=20)
    ax.legend()
    #ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'fig_probe0{pn}_pressure_comparison.png'), dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    for probe_case in probe_cases:
        ax.plot(
            probe_case['time'] / FLOW_TIMES,
            probe_case['temperature_ratio'],
            label=probe_case['label'],
            linewidth=1.5,
        )
    ax.plot(Thesistemp[:,0], Thesistemp[:,1], label='Thesis', color='black', linestyle='--')
    ax.set_xlabel(r'$\bar{t}$', fontsize=16)
    ax.set_ylabel(r'$T/T_\infty$', fontsize=16)
    ax.set_title(f'Probe {pn} Temperature Comparison', fontsize=20)
    ax.legend()
    #ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'fig_probe0{pn}_temp_comparison.png'), dpi=200)
    plt.close()

def PT_PELEC_Comp(pn):
    probe_cases = _load_probe_cases(pn)
    plot_dir = _get_plot_dir()

    PeleCData = np.loadtxt(os.path.join(plot_dir, 'flatplate_probe.dat'))
    PeleCData_adia = np.loadtxt(os.path.join(plot_dir, 'flatplate_probe_adia.dat'))

    Thesispress = np.loadtxt(os.path.join(plot_dir, 'Thesis_FP_Data.csv'), delimiter=',', skiprows=0)
    Thesistemp = np.loadtxt(os.path.join(plot_dir, 'Thesis_FP_DataT.csv'), delimiter=',', skiprows=0)

    print(" Extracted ALL data for comparison. Now plotting with pelec...")

    fig, ax = plt.subplots(figsize=(10, 6))
    for probe_case in probe_cases:
        ax.plot(
            probe_case['time'] / FLOW_TIMES,
            probe_case['pressure_ratio'],
            label=probe_case['label'],
            linewidth=1.5,
        )
    ax.plot(PeleCData[:,0] / FLOW_TIMES, PeleCData[:,3] / 7600, label='PeleC-FPV1-ISO', color='green', linestyle='-.')
    ax.plot(PeleCData_adia[:,0] / FLOW_TIMES, PeleCData_adia[:,3] / 7600, label='PeleC-FPV2-ADIA', color='red', linestyle='-.')
    ax.plot(Thesispress[:,0], Thesispress[:,1], label='Thesis', color='black', linestyle='--')
    ax.set_xlabel(r'$\bar{t}$', fontsize=16)
    ax.set_ylabel(r'$P/P_\infty$', fontsize=16)
    ax.set_title(f'Probe {pn} Pressure Comparison with PeleC', fontsize=20)
    ax.set_ylim(1, 1.6)  # Adjust y-axis limits for better visibility
    ax.legend()
    #ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'fig_probe0{pn}_pressure_comparison_pelec.png'), dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    for probe_case in probe_cases:
        ax.plot(
            probe_case['time'] / FLOW_TIMES,
            probe_case['temperature_ratio'],
            label=probe_case['label'],
            linewidth=1.5,
        )
    ax.plot(PeleCData[:,0] / FLOW_TIMES, PeleCData[:,4] / FREESTREAM_TEMPERATURE, label='PeleC-FPV1-ISO', color='green', linestyle='-.')
    ax.plot(PeleCData_adia[:,0] / FLOW_TIMES, PeleCData_adia[:,4] / FREESTREAM_TEMPERATURE, label='PeleC-FPV2-ADIA', color='red', linestyle='-.')
    ax.plot(Thesistemp[:,0], Thesistemp[:,1], label='Thesis', color='black', linestyle='--')
    ax.set_xlabel(r'$\bar{t}$', fontsize=16)
    ax.set_ylabel(r'$T/T_\infty$', fontsize=16)
    ax.set_title(f'Probe {pn} Temperature Comparison', fontsize=20)
    ax.legend()
    #ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'fig_probe0{pn}_temp_comparison_pelec.png'), dpi=200)
    plt.close()




if __name__ == '__main__':
    # REMINDER of Probe order : Time, Density, Velocity, Pressure
    num_probes = 1

    for probe in range(1,num_probes+1):
        mfcpr(probe,skipNum=0, AutoSkip=True)

        PT_Comp(probe)
        PT_PELEC_Comp(probe)
