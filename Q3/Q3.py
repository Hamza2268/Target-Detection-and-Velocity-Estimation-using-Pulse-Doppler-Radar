import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ============================================================================
# PART C: VELOCITY ESTIMATION USING DOPPLER ANALYSIS
# ============================================================================

# Parameters from Part A (AUTOMOTIVE – UNIFIED)
c = 3e8                     # Speed of light (m/s)
fc = 77e9                   # Carrier frequency (77 GHz)
PRI = 5e-6                  # Pulse Repetition Interval (5 microseconds)
PRF = 1 / PRI               # Pulse Repetition Frequency
fs = 1 / 0.5e-9             # Sampling frequency (2 GHz)
pulse_width = 4e-9          # Pulse duration (4 ns)
bandwidth = 10e6            # (Kept but NOT used – no chirp)
N_pulses = 128              # Number of pulses

# Target parameters (AUTOMOTIVE SCENARIO)
targets = {
    'Target 1': {'range': 100.0, 'velocity': 55.0},
    'Target 2': {'range': 150.0, 'velocity': -30.0}
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
# GENERATE TRANSMITTED PULSE (RECTANGULAR – SAME AS PART A)
# ============================================================================

t_pulse = np.arange(0, pulse_width, 1/fs)
transmitted_pulse = np.ones(len(t_pulse), dtype=complex)

# Time vector for one PRI
t_pri = np.arange(0, PRI, 1/fs)
range_axis = t_pri * c / 2

# ============================================================================
# GENERATE MULTI-PULSE RECEIVED SIGNAL
# ============================================================================

received_signal_matrix = np.zeros((N_pulses, len(t_pri)), dtype=complex)

for pulse_idx in range(N_pulses):
    for target_name, params in targets.items():
        R = params['range']
        v = params['velocity']

        tau = 2 * R / c
        delay_samples = int(tau * fs)

        fd = 2 * v * fc / c
        doppler_phase = 2 * np.pi * fd * pulse_idx * PRI

        if delay_samples + len(t_pulse) < len(t_pri):
            echo = transmitted_pulse * np.exp(
                1j * 2 * np.pi * fd * t_pulse
            ) * np.exp(1j * doppler_phase)

            received_signal_matrix[pulse_idx,
                delay_samples:delay_samples + len(echo)] += echo

    signal_power = np.mean(np.abs(received_signal_matrix[pulse_idx])**2)
    noise_power = signal_power / (10**(SNR_dB/10))
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(len(t_pri)) +
        1j * np.random.randn(len(t_pri))
    )
    received_signal_matrix[pulse_idx] += noise

# ============================================================================
# PERFORM MATCHED FILTERING FOR ALL PULSES
# ============================================================================

print("Performing matched filtering for all pulses...")

mf_output_matrix = np.zeros((N_pulses, len(t_pri)), dtype=complex)

for pulse_idx in range(N_pulses):
    mf_output_matrix[pulse_idx] = signal.correlate(
        received_signal_matrix[pulse_idx],
        transmitted_pulse,
        mode='same'
    )

# ============================================================================
# DETECT TARGET RANGE BINS
# ============================================================================

print("Detecting target ranges...")

mf_magnitude = np.abs(mf_output_matrix[0])
threshold = 0.5 * np.max(mf_magnitude)

peaks, _ = signal.find_peaks(
    mf_magnitude,
    height=threshold,
    distance=int(pulse_width * fs)
)

detected_ranges = range_axis[peaks]

print(f"\nDetected {len(peaks)} targets at ranges:")
for i, r in enumerate(detected_ranges):
    print(f"  Target {i+1}: {r:.2f} m (Range bin: {peaks[i]})")

# ============================================================================
# DOPPLER ANALYSIS FOR EACH DETECTED TARGET (VISUALIZATION UPDATE)
# ============================================================================

print("\n" + "=" * 70)
print("DOPPLER FREQUENCY ANALYSIS")
print("=" * 70)

doppler_freq_axis = np.fft.fftshift(np.fft.fftfreq(N_pulses, PRI))
velocity_axis = (doppler_freq_axis * c) / (2 * fc)

detected_velocities = []
velocity_errors = []
target_classifications = []

# Increase figure size to match the wide look of the screenshot
fig, axes = plt.subplots(len(peaks), 1, figsize=(14, 8))
if len(peaks) == 1:
    axes = [axes]

# Properties for the text boxes (Beige background)
box_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

for idx, peak_idx in enumerate(peaks):
    slow_time_signal = mf_output_matrix[:, peak_idx]

    doppler_spectrum = np.fft.fftshift(
        np.fft.fft(slow_time_signal, N_pulses)
    )

    doppler_magnitude = np.abs(doppler_spectrum)
    peak_doppler_idx = np.argmax(doppler_magnitude)

    detected_velocity = velocity_axis[peak_doppler_idx]
    detected_velocities.append(detected_velocity)

    # Note: This assumes peaks are found in the same order as targets dict
    true_velocity = list(targets.values())[idx]['velocity']
    velocity_error = detected_velocity - true_velocity
    velocity_errors.append(velocity_error)

    classification = (
        "Approaching" if detected_velocity > 0 else
        "Receding" if detected_velocity < 0 else
        "Stationary"
    )
    target_classifications.append(classification)

    ax = axes[idx]
    
    # 1. Plot the Doppler Spectrum Line (Blue)
    ax.plot(velocity_axis, doppler_magnitude, color='blue', linewidth=1.5, label='Doppler Spectrum')
    
    # 2. Plot the Detected Peak (Red Dot)
    ax.plot(detected_velocity, doppler_magnitude[peak_doppler_idx], 'ro', markersize=12, 
            label=f'Detected: {detected_velocity:.2f} m/s', zorder=5)
    
    # 3. Plot the True Velocity (Green Dotted Line)
    ax.axvline(x=true_velocity, color='green', linestyle=':', linewidth=3, 
               label=f'True: {true_velocity:.2f} m/s', zorder=4)

    # 4. Title Styling (Bold, with Range in km)
    range_km = detected_ranges[idx] / 1000.0  # Convert m to km for the title style
    ax.set_title(f"Target {idx+1} - Doppler Spectrum (Range: {range_km:.2f} km)", 
                 fontsize=14, fontweight='bold')
    
    # 5. Text Boxes for Error and Classification
    # Error Box (Top Left, higher)
    ax.text(0.02, 0.90, f"Error: {velocity_error:.2f} m/s", transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', bbox=box_props)
    
    # Classification Box (Top Left, lower)
    ax.text(0.02, 0.78, f"Classification: {classification}", transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', bbox=box_props)

    # 6. Labels and Grid
    ax.set_xlabel("Velocity (m/s)", fontsize=12)
    ax.set_ylabel("Magnitude", fontsize=12)
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlim(-100, 150)
    
    # 7. Legend (Upper Right)
    ax.legend(loc='upper right', framealpha=1, fancybox=True, fontsize=10)

plt.tight_layout()
plt.savefig("./Q3/Velocity_Estimation_using_Doppler_Analysis.png")
plt.show()

print("\n" + "=" * 70)
print("DOPPLER ANALYSIS COMPLETE")
print("=" * 70)