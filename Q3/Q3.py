import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# ============================================================================
# PART C: VELOCITY ESTIMATION USING DOPPLER ANALYSIS
# ============================================================================

# Parameters from Part A (use your own values)
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

print("=" * 70)
print("PART C: VELOCITY ESTIMATION USING DOPPLER ANALYSIS")
print("=" * 70)
print(f"Carrier Frequency: {fc/1e9:.1f} GHz")
print(f"PRI: {PRI*1e6:.1f} μs")
print(f"PRF: {PRF/1e3:.1f} kHz")
print(f"Number of Pulses: {N_pulses}")
print()

# ============================================================================
# GENERATE TRANSMITTED PULSE (LFM Chirp)
# ============================================================================

t_pulse = np.arange(0, pulse_width, 1/fs)
chirp_rate = bandwidth / pulse_width
transmitted_pulse = np.exp(1j * np.pi * chirp_rate * t_pulse**2)

# Time vector for one PRI
t_pri = np.arange(0, PRI, 1/fs)
range_axis = t_pri * c / 2

# ============================================================================
# GENERATE MULTI-PULSE RECEIVED SIGNAL
# ============================================================================

# Initialize received signal matrix: [N_pulses x samples_per_pulse]
received_signal_matrix = np.zeros((N_pulses, len(t_pri)), dtype=complex)

for pulse_idx in range(N_pulses):
    for target_name, params in targets.items():
        R = params['range']
        v = params['velocity']
        
        # Time delay
        tau = 2 * R / c
        delay_samples = int(tau * fs)
        
        # Doppler frequency
        fd = 2 * v * fc / c
        
        # Doppler phase accumulation across pulses
        doppler_phase = 2 * np.pi * fd * pulse_idx * PRI
        
        # Echo signal with Doppler phase
        if delay_samples + len(t_pulse) < len(t_pri):
            echo = transmitted_pulse * np.exp(1j * 2 * np.pi * fd * t_pulse) * np.exp(1j * doppler_phase)
            received_signal_matrix[pulse_idx, delay_samples:delay_samples+len(echo)] += echo
    
    # Add noise to each pulse
    signal_power = np.mean(np.abs(received_signal_matrix[pulse_idx, :])**2)
    noise_power = signal_power / (10**(SNR_dB/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(t_pri)) + 
                                       1j * np.random.randn(len(t_pri)))
    received_signal_matrix[pulse_idx, :] += noise

# ============================================================================
# PERFORM MATCHED FILTERING FOR ALL PULSES
# ============================================================================

print("Performing matched filtering for all pulses...")

# Matched filter output for all pulses
mf_output_matrix = np.zeros((N_pulses, len(t_pri)), dtype=complex)

for pulse_idx in range(N_pulses):
    mf_output_matrix[pulse_idx, :] = signal.correlate(
        received_signal_matrix[pulse_idx, :], 
        transmitted_pulse, 
        mode='same'
    )

# ============================================================================
# DETECT TARGET RANGE BINS
# ============================================================================

print("Detecting target ranges...")

# Use first pulse for range detection
mf_magnitude = np.abs(mf_output_matrix[0, :])
threshold = 0.5 * np.max(mf_magnitude)

# Find peaks
peaks, _ = signal.find_peaks(mf_magnitude, height=threshold, distance=int(1e-6*fs))
detected_ranges = range_axis[peaks]

print(f"\nDetected {len(peaks)} targets at ranges:")
for i, r in enumerate(detected_ranges):
    print(f"  Target {i+1}: {r/1000:.2f} km (Range bin: {peaks[i]})")

# ============================================================================
# DOPPLER ANALYSIS FOR EACH DETECTED TARGET
# ============================================================================

print("\n" + "=" * 70)
print("DOPPLER FREQUENCY ANALYSIS")
print("=" * 70)

# Doppler frequency axis (after FFT and fftshift)
doppler_freq_axis = np.fft.fftshift(np.fft.fftfreq(N_pulses, PRI))

# Convert Doppler frequency to velocity: v = (fd * c) / (2 * fc)
velocity_axis = (doppler_freq_axis * c) / (2 * fc)

# Store results
detected_velocities = []
velocity_errors = []
target_classifications = []

# Create figure for Doppler spectra
fig, axes = plt.subplots(len(peaks), 1, figsize=(12, 5*len(peaks)))
if len(peaks) == 1:
    axes = [axes]

for idx, peak_idx in enumerate(peaks):
    print(f"\n--- Processing Target {idx+1} at range bin {peak_idx} ---")
    
    # Extract signal across all pulses at this range bin
    slow_time_signal = mf_output_matrix[:, peak_idx]
    
    # Apply FFT to get Doppler spectrum
    doppler_spectrum = np.fft.fft(slow_time_signal, N_pulses)
    
    # Apply fftshift to center zero-Doppler
    doppler_spectrum_shifted = np.fft.fftshift(doppler_spectrum)
    
    # Get magnitude
    doppler_magnitude = np.abs(doppler_spectrum_shifted)
    
    # Find peak in Doppler spectrum
    peak_doppler_idx = np.argmax(doppler_magnitude)
    detected_fd = doppler_freq_axis[peak_doppler_idx]
    detected_velocity = velocity_axis[peak_doppler_idx]
    
    # Find corresponding true target
    true_velocity = None
    true_range = detected_ranges[idx]
    for target_name, params in targets.items():
        if abs(params['range'] - true_range) < 500:  # Within 500m
            true_velocity = params['velocity']
            break
    
    # Calculate error
    if true_velocity is not None:
        velocity_error = detected_velocity - true_velocity
        velocity_errors.append(velocity_error)
    else:
        velocity_error = None
        velocity_errors.append(None)
    
    detected_velocities.append(detected_velocity)
    
    # Classify target direction
    if detected_velocity > 0:
        classification = "Approaching"
    elif detected_velocity < 0:
        classification = "Receding"
    else:
        classification = "Stationary"
    target_classifications.append(classification)
    
    print(f"  Detected Doppler Frequency: {detected_fd:.2f} Hz")
    print(f"  Detected Velocity: {detected_velocity:.2f} m/s")
    if true_velocity is not None:
        print(f"  True Velocity: {true_velocity:.2f} m/s")
        print(f"  Velocity Error: {velocity_error:.2f} m/s")
    print(f"  Classification: {classification}")
    
    # Plot Doppler spectrum
    ax = axes[idx]
    ax.plot(velocity_axis, doppler_magnitude, 'b-', linewidth=1.5, label='Doppler Spectrum')
    ax.plot(detected_velocity, doppler_magnitude[peak_doppler_idx], 'ro', 
            markersize=12, label=f'Detected: {detected_velocity:.2f} m/s')
    
    if true_velocity is not None:
        ax.axvline(x=true_velocity, color='g', linestyle=':', linewidth=3, 
                   alpha=0.8, label=f'True: {true_velocity:.2f} m/s')
        if velocity_error is not None:
            ax.text(0.02, 0.95, f'Error: {velocity_error:.2f} m/s', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        if target_classifications is not None:
            ax.text(0.02, 0.88, f'Classification: {target_classifications[idx]} ', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    # if velocity_errors  is not None:
    #     ax.axvline(label=f'Error: {velocity_errors:.2f} m/s')
    # if target_classifications is not None:
    #     ax.axvline( label=f'Classification: {target_classifications:.2f} m/s')
    
    ax.set_xlabel('Velocity (m/s)', fontsize=11)
    ax.set_ylabel('Magnitude', fontsize=11)
    ax.set_title(f'Target {idx+1} - Doppler Spectrum (Range: {true_range/1000:.2f} km)', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set appropriate x-axis limits
    v_max_plot = c * PRF / (4 * fc)
    ax.set_xlim([-v_max_plot, v_max_plot])

plt.tight_layout()
plt.savefig('./Q3/Velocity_Estimation_using_Doppler_Analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# RESULTS TABLE
# ============================================================================

print("\n" + "=" * 70)
print("VELOCITY ESTIMATION RESULTS TABLE")
print("=" * 70)
print(f"{'Target':<10} {'Range (km)':<15} {'True Vel (m/s)':<18} {'Detected Vel (m/s)':<22} {'Error (m/s)':<15} {'Direction':<15}")
print("-" * 110)

for idx in range(len(peaks)):
    target_num = idx + 1
    range_km = detected_ranges[idx] / 1000
    
    # Find true velocity
    true_vel = None
    for target_name, params in targets.items():
        if abs(params['range'] - detected_ranges[idx]) < 500:
            true_vel = params['velocity']
            break
    
    detected_vel = detected_velocities[idx]
    error = velocity_errors[idx]
    direction = target_classifications[idx]
    
    if true_vel is not None and error is not None:
        print(f"{target_num:<10} {range_km:<15.2f} {true_vel:<18.2f} {detected_vel:<22.2f} {error:<15.2f} {direction:<15}")
    else:
        print(f"{target_num:<10} {range_km:<15.2f} {'N/A':<18} {detected_vel:<22.2f} {'N/A':<15} {direction:<15}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

valid_errors = [e for e in velocity_errors if e is not None]
if valid_errors:
    mean_error = np.mean(np.abs(valid_errors))
    max_error = np.max(np.abs(valid_errors))
    print(f"Mean Absolute Velocity Error: {mean_error:.2f} m/s")
    print(f"Maximum Velocity Error: {max_error:.2f} m/s")

print(f"\nVelocity Resolution: {(c * PRF) / (2 * fc * N_pulses):.2f} m/s")
print(f"Maximum Unambiguous Velocity: {(c * PRF) / (4 * fc):.2f} m/s")

approaching_count = sum(1 for c in target_classifications if c == "Approaching")
receding_count = sum(1 for c in target_classifications if c == "Receding")
stationary_count = sum(1 for c in target_classifications if c == "Stationary")

print(f"\nTarget Classification Summary:")
print(f"  Approaching Targets: {approaching_count}")
print(f"  Receding Targets: {receding_count}")
print(f"  Stationary Targets: {stationary_count}")

print("\n" + "=" * 70)
print("DOPPLER ANALYSIS COMPLETE")
print("=" * 70)