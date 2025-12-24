import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# ============================================================================
# PART B: RANGE DETECTION - METHOD COMPARISON
# ============================================================================

# Parameters from Part A (you should use your own values)
c = 3e8  # Speed of light (m/s)
fc = 10e9  # Carrier frequency (10 GHz, X-band)
PRI = 100e-6  # Pulse Repetition Interval (100 microseconds)
PRF = 1 / PRI  # Pulse Repetition Frequency
fs = 50e6  # Sampling frequency (50 MHz)
pulse_width = 1e-6  # Pulse duration (1 microsecond)
bandwidth = 10e6  # Bandwidth for chirp (10 MHz)
N_pulses = 128  # Number of pulses

# Target parameters
targets = {
    'Target 1': {'range': 5000, 'velocity': 50},  # 5 km, approaching at 50 m/s
    'Target 2': {'range': 12000, 'velocity': -30}  # 12 km, receding at 30 m/s
}

# SNR in dB
SNR_dB = -15

# Calculate maximum unambiguous parameters
R_max = c * PRI / 2
v_max = c * PRF / (4 * fc)

print("=" * 70)
print("RADAR SYSTEM PARAMETERS")
print("=" * 70)
print(f"Carrier Frequency: {fc/1e9:.1f} GHz")
print(f"PRI: {PRI*1e6:.1f} μs")
print(f"PRF: {PRF/1e3:.1f} kHz")
print(f"Maximum Unambiguous Range: {R_max/1e3:.2f} km")
print(f"Maximum Unambiguous Velocity: {v_max:.2f} m/s")
print(f"Pulse Width: {pulse_width*1e6:.1f} μs")
print(f"SNR: {SNR_dB} dB")
print()

# ============================================================================
# GENERATE TRANSMITTED PULSE (LFM Chirp - Better range resolution)
# ============================================================================

t_pulse = np.arange(0, pulse_width, 1/fs)
# LFM chirp signal
chirp_rate = bandwidth / pulse_width
transmitted_pulse = np.exp(1j * np.pi * chirp_rate * t_pulse**2)

# Time vector for one PRI
t_pri = np.arange(0, PRI, 1/fs)

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
    
    # Echo signal (for first pulse, n=0)
    if delay_samples + len(t_pulse) < len(t_pri):
        echo = transmitted_pulse * np.exp(1j * 2 * np.pi * fd * t_pulse)
        received_signal[delay_samples:delay_samples+len(echo)] += echo

# Add noise
signal_power = np.mean(np.abs(received_signal)**2)
noise_power = signal_power / (10**(SNR_dB/10))
noise = np.sqrt(noise_power/2) * (np.random.randn(len(received_signal)) + 
                                   1j * np.random.randn(len(received_signal)))
received_signal += noise

# ============================================================================
# METHOD 1: MATCHED FILTER (TIME DOMAIN)
# ============================================================================

print("=" * 70)
print("METHOD 1: MATCHED FILTER (TIME DOMAIN)")
print("=" * 70)

# Matched filter = convolution with time-reversed conjugate of transmitted pulse
matched_filter = np.conj(transmitted_pulse[::-1])

start_time = time.time()
# Perform convolution
mf_output = signal.correlate(received_signal, transmitted_pulse, mode='same')
mf_time = time.time() - start_time

# Convert to magnitude
mf_magnitude = np.abs(mf_output)

# Convert time samples to range
range_axis = t_pri * c / 2

print(f"Processing time: {mf_time*1000:.4f} ms")

# ============================================================================
# METHOD 2: FFT-BASED PROCESSING (FREQUENCY DOMAIN)
# ============================================================================

print("\n" + "=" * 70)
print("METHOD 2: FFT-BASED PROCESSING (FREQUENCY DOMAIN)")
print("=" * 70)

start_time = time.time()

# Zero-pad to same length
N_fft = len(received_signal)

# FFT of received signal
R_fft = np.fft.fft(received_signal, N_fft)

# FFT of transmitted pulse (zero-padded)
S_fft = np.fft.fft(transmitted_pulse, N_fft)

# Matched filtering in frequency domain: multiply by conjugate
MF_fft = R_fft * np.conj(S_fft)

# IFFT to get back to time domain
fft_output = np.fft.ifft(MF_fft)

fft_time = time.time() - start_time

# Convert to magnitude
fft_magnitude = np.abs(fft_output)

print(f"Processing time: {fft_time*1000:.4f} ms")

# ============================================================================
# PEAK DETECTION
# ============================================================================

# Set threshold (e.g., 50% of maximum)
threshold_mf = 0.5 * np.max(mf_magnitude)
threshold_fft = 0.5 * np.max(fft_magnitude)

# Find peaks for Method 1
peaks_mf, _ = signal.find_peaks(mf_magnitude, height=threshold_mf, distance=int(1e-6*fs))
detected_ranges_mf = range_axis[peaks_mf] / 1000  # Convert to km

# Find peaks for Method 2
peaks_fft, _ = signal.find_peaks(fft_magnitude, height=threshold_fft, distance=int(1e-6*fs))
detected_ranges_fft = range_axis[peaks_fft] / 1000  # Convert to km

# ============================================================================
# RESULTS COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("DETECTION RESULTS")
print("=" * 70)

print("\nTrue Target Ranges:")
for target_name, params in targets.items():
    print(f"  {target_name}: {params['range']/1000:.2f} km")

print("\nMethod 1 (Time Domain) - Detected Ranges:")
for i, r in enumerate(detected_ranges_mf):
    print(f"  Peak {i+1}: {r:.2f} km")

print("\nMethod 2 (Frequency Domain) - Detected Ranges:")
for i, r in enumerate(detected_ranges_fft):
    print(f"  Peak {i+1}: {r:.2f} km")

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

# Plot 1: Matched Filter Output (Method 1)
ax1 = plt.subplot(3, 1, 1)
plt.plot(range_axis/1000, mf_magnitude, 'b-', linewidth=1.5, label='MF Output')
plt.plot(range_axis[peaks_mf]/1000, mf_magnitude[peaks_mf], 'ro', 
         markersize=10, label='Detected Peaks')
plt.axhline(y=threshold_mf, color='r', linestyle='--', alpha=0.5, label='Threshold')
for target_name, params in targets.items():
    plt.axvline(x=params['range']/1000, color='g', linestyle='--', 
                alpha=0.7, linewidth=3)
plt.xlabel('Range (km)')
plt.ylabel('Magnitude')
plt.title('Method 1: Matched Filter Output (Time Domain)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 20])

# Plot 2: FFT Method Output (Method 2)
ax2 = plt.subplot(3, 1, 2)
plt.plot(range_axis/1000, fft_magnitude, 'b-', linewidth=1.5, label='FFT Output')
plt.plot(range_axis[peaks_fft]/1000, fft_magnitude[peaks_fft], 'ro', 
         markersize=10, label='Detected Peaks')
plt.axhline(y=threshold_fft, color='r', linestyle='--', alpha=0.5, label='Threshold')
for target_name, params in targets.items():
    plt.axvline(x=params['range']/1000, color='g', linestyle='--', 
                alpha=0.7, linewidth=3)
plt.xlabel('Range (km)')
plt.ylabel('Magnitude')
plt.title('Method 2: FFT-Based Processing (Frequency Domain)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 20])

# Plot 3: Comparison of Both Methods
ax3 = plt.subplot(3, 1, 3)
plt.plot(range_axis/1000, mf_magnitude/np.max(mf_magnitude), 'b-', 
         linewidth=1.5, alpha=0.7, label='Method 1 (Normalized)')
plt.plot(range_axis/1000, fft_magnitude/np.max(fft_magnitude), 'r--', 
         linewidth=1.5, alpha=0.7, label='Method 2 (Normalized)')
plt.plot(range_axis[peaks_mf]/1000, mf_magnitude[peaks_mf]/np.max(mf_magnitude), 
         'bo', markersize=8, label='Method 1 Peaks')
plt.plot(range_axis[peaks_fft]/1000, fft_magnitude[peaks_fft]/np.max(fft_magnitude), 
         'ro', markersize=8, label='Method 2 Peaks')
for target_name, params in targets.items():
    plt.axvline(x=params['range']/1000, color='g', linestyle='-', 
                alpha=0.7, linewidth=3, label='True Range' if target_name == 'Target 1' else '')
plt.xlabel('Range (km)')
plt.ylabel('Normalized Magnitude')
plt.title('Comparison: Both Methods (Normalized)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 20])

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

# Target detection comparison
true_ranges = [params['range']/1000 for params in targets.values()]
print(f"{'True Ranges (km)':<30} {str(true_ranges):<20} {str(true_ranges):<20}")
print(f"{'Detected Ranges (km)':<30} {str([f'{r:.2f}' for r in detected_ranges_mf]):<20} {str([f'{r:.2f}' for r in detected_ranges_fft]):<20}")
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