import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

c = 3e8
fc = 77e9
Ts = 0.5e-9
fs = 1 / Ts
pulse_width = 4e-9
PRI = 5e-6
PRF = 1 / PRI
N_pulses = 780

targets = {
    'Target 1': {'range': 100.0, 'velocity': 55.0},
    'Target 2': {'range': 150.0, 'velocity': -30.0}
}

attenuation = {
    name: 1 / (params['range'] ** 2)
    for name, params in targets.items()
}

SNR_dB = -10

R_max = c * PRI / 2
v_max = c * PRF / (4 * fc)


t_pulse = np.arange(0, pulse_width, Ts)
transmitted_pulse = np.ones(len(t_pulse), dtype=complex)

t_pri = np.arange(0, PRI, Ts)


received_signal = np.zeros(len(t_pri), dtype=complex)

for target_name, params in targets.items():
    R = params['range']
    v = params['velocity']

    tau = 2 * R / c
    delay_samples = int(tau * fs)

    fd = 2 * v * fc / c

    if delay_samples + len(t_pulse) < len(t_pri):
        echo = transmitted_pulse * np.exp(1j * 2 * np.pi * fd * t_pulse)
        echo *= attenuation[target_name]
        received_signal[delay_samples:delay_samples+len(echo)] += echo

signal_power = np.mean(np.abs(received_signal)**2)
noise_power = signal_power / (10**(SNR_dB/10))
noise = np.sqrt(noise_power/2) * (
    np.random.randn(len(received_signal)) +
    1j * np.random.randn(len(received_signal))
)
received_signal += noise

start_time = time.time()
mf_output = signal.correlate(received_signal, transmitted_pulse, mode='same')
mf_time = time.time() - start_time

mf_magnitude = np.abs(mf_output)
range_axis = t_pri * c / 2

start_time = time.time()

N_fft = len(received_signal)
R_fft = np.fft.fft(received_signal, N_fft)
S_fft = np.fft.fft(transmitted_pulse, N_fft)
MF_fft = R_fft * np.conj(S_fft)
fft_output = np.fft.ifft(MF_fft)

fft_time = time.time() - start_time
fft_magnitude = np.abs(fft_output)


P_fa = 1e-6

pulse_energy_gain = np.sum(np.abs(transmitted_pulse)**2)
noise_power_output = noise_power * pulse_energy_gain

scientific_threshold = np.sqrt(-noise_power_output * np.log(P_fa))

threshold_mf = scientific_threshold
threshold_fft = scientific_threshold

peaks_mf, _ = signal.find_peaks(mf_magnitude, height=threshold_mf,
                               distance=int(pulse_width*fs))
peaks_fft, _ = signal.find_peaks(fft_magnitude, height=threshold_fft,
                                distance=int(pulse_width*fs))

detected_ranges_mf = range_axis[peaks_mf]
detected_ranges_fft = range_axis[peaks_fft]

fig = plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(range_axis, mf_magnitude, linewidth=1.5, label='MF Output')
plt.plot(range_axis[peaks_mf], mf_magnitude[peaks_mf], 'ro', label='Detected Peaks')
plt.axhline(y=threshold_mf, color='r', linestyle='--', alpha=0.5, label='Threshold')
for params in targets.values():
    plt.axvline(x=params['range'], color='g', linestyle='--', linewidth=2)
plt.xlabel('Range (m)')
plt.ylabel('Magnitude')
plt.title('Method 1: Time-Domain Matched Filter Output')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 250])

plt.subplot(3, 1, 2)
plt.plot(range_axis, fft_magnitude, linewidth=1.5, label='FFT Output')
plt.plot(range_axis[peaks_fft], fft_magnitude[peaks_fft], 'ro', label='Detected Peaks')
plt.axhline(y=threshold_fft, color='r', linestyle='--', alpha=0.5, label='Threshold')
for params in targets.values():
    plt.axvline(x=params['range'], color='g', linestyle='--', linewidth=2)
plt.xlabel('Range (m)')
plt.ylabel('Magnitude')
plt.title('Method 2: FFT-Based Matched Filtering')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 250])

plt.subplot(3, 1, 3)
plt.plot(range_axis, mf_magnitude / np.max(mf_magnitude),
         label='Method 1 (Normalized)', alpha=0.7)
plt.plot(range_axis, fft_magnitude / np.max(fft_magnitude),
         '--', label='Method 2 (Normalized)', alpha=0.7)
plt.plot(range_axis[peaks_mf], mf_magnitude[peaks_mf]/np.max(mf_magnitude),
         'bo', markersize=8, label='Method 1 Peaks')
plt.plot(range_axis[peaks_fft], fft_magnitude[peaks_fft]/np.max(fft_magnitude),
         'ro', markersize=8, label='Method 2 Peaks')
for i, params in enumerate(targets.values()):
    label = 'True Range' if i == 0 else ''
    plt.axvline(x=params['range'], color='g', linestyle='-',
                linewidth=3, alpha=0.7, label=label)
plt.xlabel('Range (m)')
plt.ylabel('Normalized Magnitude')
plt.title('Comparison: Both Methods (Normalized)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([90, 160])

plt.tight_layout()
plt.savefig('./Q2/comparison_methods_radar_detection.png', dpi=300, bbox_inches='tight')
plt.show()
