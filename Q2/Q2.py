import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# ============================================================================
# PART B: RANGE DETECTION - METHOD COMPARISON (AUTOMOTIVE RADAR)
# ============================================================================

# Parameters from Part A (Q1) - UNIFIED
c = 3e8                    # Speed of light (m/s)
fc = 77e9                  # Carrier frequency (77 GHz)
Ts = 0.5e-9                # Sampling time (0.5 ns)
fs = 1 / Ts                # Sampling frequency (2 GHz)
pulse_width = 4e-9         # Pulse duration (4 ns)
PRI = 5e-6                 # Pulse Repetition Interval (5 µs)
PRF = 1 / PRI
N_pulses = 780             # Number of pulses (from Q1)

# Target parameters (AUTOMOTIVE SCENARIO - from Q1)
targets = {
    'Target 1': {'range': 100.0, 'velocity': 55.0},    # 100 m, approaching
    'Target 2': {'range': 150.0, 'velocity': -30.0}    # 150 m, receding
}

# SNR in dB 
SNR_dB = -10

# Calculate maximum unambiguous parameters
R_max = c * PRI / 2
v_max = c * PRF / (4 * fc)

print("=" * 70)
print("RADAR SYSTEM PARAMETERS (UNIFIED WITH PART A)")
print("=" * 70)
print(f"Carrier Frequency: {fc/1e9:.1f} GHz")
print(f"Sampling Frequency: {fs/1e9:.1f} GHz")
print(f"PRI: {PRI*1e6:.1f} μs")
print(f"Maximum Unambiguous Range: {R_max:.1f} m")
print(f"Maximum Unambiguous Velocity: {v_max:.2f} m/s")
print(f"Pulse Width: {pulse_width*1e9:.1f} ns")
print(f"SNR: {SNR_dB} dB")
print()

# ============================================================================
# GENERATE TRANSMITTED PULSE (RECTANGULAR - SAME AS PART A)
# ============================================================================

t_pulse = np.arange(0, pulse_width, Ts)
transmitted_pulse = np.ones(len(t_pulse), dtype=complex)

# Time vector for one PRI
t_pri = np.arange(0, PRI, Ts)

# ============================================================================
# GENERATE RECEIVED SIGNAL FOR FIRST PULSE
# ============================================================================

received_signal = np.zeros(len(t_pri), dtype=complex)

for target_name, params in targets.items():
    R = params['range']
    v = params['velocity']

    # Time delay
    tau = 2 * R / c
    delay_samples = int(tau * fs)

    # Doppler frequency
    fd = 2 * v * fc / c

    # Echo signal (single pulse)
    if delay_samples + len(t_pulse) < len(t_pri):
        echo = transmitted_pulse * np.exp(1j * 2 * np.pi * fd * t_pulse)
        received_signal[delay_samples:delay_samples+len(echo)] += echo

# Add AWGN
signal_power = np.mean(np.abs(received_signal)**2)
noise_power = signal_power / (10**(SNR_dB/10))
noise = np.sqrt(noise_power/2) * (
    np.random.randn(len(received_signal)) +
    1j * np.random.randn(len(received_signal))
)
received_signal += noise

# ============================================================================
# METHOD 1: MATCHED FILTER (TIME DOMAIN)
# ============================================================================

print("=" * 70)
print("METHOD 1: MATCHED FILTER (TIME DOMAIN)")
print("=" * 70)

start_time = time.time()
mf_output = signal.correlate(received_signal, transmitted_pulse, mode='same')
mf_time = time.time() - start_time

mf_magnitude = np.abs(mf_output)
range_axis = t_pri * c / 2

print(f"Processing time: {mf_time*1000:.4f} ms")

# ============================================================================
# METHOD 2: FFT-BASED PROCESSING (FREQUENCY DOMAIN)
# ============================================================================

print("\n" + "=" * 70)
print("METHOD 2: FFT-BASED PROCESSING (FREQUENCY DOMAIN)")
print("=" * 70)

start_time = time.time()

N_fft = len(received_signal)
R_fft = np.fft.fft(received_signal, N_fft)
S_fft = np.fft.fft(transmitted_pulse, N_fft)
MF_fft = R_fft * np.conj(S_fft)
fft_output = np.fft.ifft(MF_fft)

fft_time = time.time() - start_time
fft_magnitude = np.abs(fft_output)

print(f"Processing time: {fft_time*1000:.4f} ms")

# ============================================================================
# PEAK DETECTION
# ============================================================================

# 1. Define Desired Probability of False Alarm (Pfa)
# Standard radar Pfa is often between 1e-4 and 1e-6
P_fa = 1e-6  

# 2. Calculate Noise Power at Filter Output
# The matched filter (correlation) sums the energy over the pulse duration.
# This increases the noise variance by a factor equal to the pulse energy.
pulse_energy_gain = np.sum(np.abs(transmitted_pulse)**2)
noise_power_output = noise_power * pulse_energy_gain

# 3. Calculate Scientific Threshold (Rayleigh Distribution)
# Formula: V_t = sqrt( -2 * sigma^2 * ln(Pfa) )
# Note: In our complex noise model, total variance (noise_power_output) = 2 * sigma^2
# So the formula simplifies to: sqrt( -noise_power_output * ln(Pfa) )
scientific_threshold = np.sqrt(-noise_power_output * np.log(P_fa))

print("\n" + "-" * 40)
print("THRESHOLD CALCULATION")
print("-" * 40)
print(f"Target P_fa:          {P_fa}")
print(f"Input Noise Power:    {noise_power:.2e}")
print(f"Filter Gain:          {pulse_energy_gain:.2f}")
print(f"Calculated Threshold: {scientific_threshold:.4f}")

# Apply the calculated threshold to both methods
# (Both methods are mathematically equivalent, so they share the threshold)
threshold_mf = scientific_threshold
threshold_fft = scientific_threshold

peaks_mf, _ = signal.find_peaks(mf_magnitude, height=threshold_mf,
                               distance=int(pulse_width*fs))
peaks_fft, _ = signal.find_peaks(fft_magnitude, height=threshold_fft,
                                distance=int(pulse_width*fs))

detected_ranges_mf = range_axis[peaks_mf]
detected_ranges_fft = range_axis[peaks_fft]

# ============================================================================
# RESULTS COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("DETECTION RESULTS")
print("=" * 70)

print("\nTrue Target Ranges (m):")
for target_name, params in targets.items():
    print(f"  {target_name}: {params['range']:.1f} m")

print("\nMethod 1 (Time Domain) - Detected Ranges (m):")
for i, r in enumerate(detected_ranges_mf):
    print(f"  Peak {i+1}: {r:.2f} m")

print("\nMethod 2 (Frequency Domain) - Detected Ranges (m):")
for i, r in enumerate(detected_ranges_fft):
    print(f"  Peak {i+1}: {r:.2f} m")

print("\n" + "=" * 70)
print("PROCESSING TIME COMPARISON")
print("=" * 70)
print(f"Method 1 (Time Domain):      {mf_time*1000:.4f} ms")
print(f"Method 2 (Frequency Domain): {fft_time*1000:.4f} ms")
print(f"Speedup Factor:              {mf_time/fft_time:.2f}x")

# ============================================================================
# PLOTTING
# ============================================================================

fig = plt.figure(figsize=(14, 10))

# Plot 1: Time-Domain Matched Filter Output
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

# Plot 2: FFT-Based Output
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

# Plot 3: Normalized Comparison
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

# ============================================================================
# COMPARISON TABLE
# ============================================================================

print("\n" + "=" * 70)
print("DETAILED COMPARISON TABLE")
print("=" * 70)
print(f"{'Parameter':<30} {'Method 1 (Time)':<20} {'Method 2 (FFT)':<20}")
print("-" * 70)

true_ranges = [params['range'] for params in targets.values()]
print(f"{'True Ranges (m)':<30} {str(true_ranges):<20} {str(true_ranges):<20}")
print(f"{'Detected Ranges (m)':<30} {str([f'{r:.2f}' for r in detected_ranges_mf]):<20} {str([f'{r:.2f}' for r in detected_ranges_fft]):<20}")
print(f"{'Processing Time (ms)':<30} {mf_time*1000:<20.4f} {fft_time*1000:<20.4f}")
print(f"{'Number of Peaks Detected':<30} {len(peaks_mf):<20} {len(peaks_fft):<20}")

print("\n" + "=" * 70)
print("ANALYSIS ANSWERS")
print("=" * 70)
print("\n1. Are the results from both methods equivalent?")
print("   YES - Both methods produce identical results because:")
print("   - Convolution in time domain = Multiplication in frequency domain")
print("   - The matched filter is mathematically equivalent in both domains")
print("   - Any small numerical differences are due to floating-point precision")
print("\n2. Which method is computationally faster?")
if fft_time < mf_time:
    print(f"   FFT method is faster by a factor of {mf_time/fft_time:.2f}x")
    print("   - FFT has O(N log N) complexity")
    print("   - Direct convolution has O(N²) complexity")
    print("   - For large signals, FFT is significantly more efficient")
else:
    print(f"   Time domain method is faster by a factor of {fft_time/mf_time:.2f}x")
    print("   - This can happen for very short pulses")
    print("   - FFT overhead may exceed direct convolution for small N")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)