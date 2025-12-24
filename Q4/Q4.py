import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# ============================================================================
# PART D: RANGE-DOPPLER MAP GENERATION
# ============================================================================

# Parameters from Part A (Q1) - UNIFIED WITH Q2
c = 3e8                    # Speed of light (m/s)
fc = 77e9                  # Carrier frequency (77 GHz)
Ts = 0.5e-9                # Sampling time (0.5 ns)
fs = 1 / Ts                # Sampling frequency (2 GHz)
pulse_width = 4e-9         # Pulse duration (4 ns)
PRI = 5e-6                 # Pulse Repetition Interval (5 µs)
PRF = 1 / PRI              # Pulse Repetition Frequency
N_pulses = 780             # Number of pulses (from Q1)

# Target parameters (AUTOMOTIVE SCENARIO - from Q1)
targets = {
    'Target 1': {'range': 100.0, 'velocity': 55.0},    # 100 m, approaching
    'Target 2': {'range': 150.0, 'velocity': -30.0}    # 150 m, receding
}

# SNR in dB (from Q1)
SNR_dB = -5

# Calculate maximum unambiguous parameters
R_max = c * PRI / 2
v_max = c * PRF / (4 * fc)

print("=" * 70)
print("PART D: RANGE-DOPPLER MAP GENERATION")
print("=" * 70)
print(f"Carrier Frequency: {fc/1e9:.1f} GHz")
print(f"PRI: {PRI*1e6:.1f} µs")
print(f"PRF: {PRF/1e3:.1f} kHz")
print(f"Number of Pulses: {N_pulses}")
print(f"Maximum Unambiguous Range: {R_max:.1f} m")
print(f"Maximum Unambiguous Velocity: {v_max:.2f} m/s")
print()

# ============================================================================
# GENERATE TRANSMITTED PULSE (RECTANGULAR - SAME AS Q1 & Q2)
# ============================================================================

t_pulse = np.arange(0, pulse_width, Ts)
transmitted_pulse = np.ones(len(t_pulse), dtype=complex)

# Time vector for one PRI
t_pri = np.arange(0, PRI, Ts)
range_axis = t_pri * c / 2

# ============================================================================
# GENERATE MULTI-PULSE RECEIVED SIGNAL
# ============================================================================

print("Generating multi-pulse received signal...")

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
    if signal_power > 0:
        noise_power = signal_power / (10**(SNR_dB/10))
    else:
        noise_power = 1e-10
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(t_pri)) + 
                                       1j * np.random.randn(len(t_pri)))
    received_signal_matrix[pulse_idx, :] += noise

print(f"Generated signal matrix shape: {received_signal_matrix.shape}")

# ============================================================================
# PERFORM MATCHED FILTERING FOR ALL PULSES (RANGE PROCESSING)
# ============================================================================

print("\nPerforming matched filtering for all pulses (Fast-time processing)...")

start_time = time.time()

# Matched filter output for all pulses
mf_output_matrix = np.zeros((N_pulses, len(t_pri)), dtype=complex)

for pulse_idx in range(N_pulses):
    mf_output_matrix[pulse_idx, :] = signal.correlate(
        received_signal_matrix[pulse_idx, :], 
        transmitted_pulse, 
        mode='same'
    )

mf_time = time.time() - start_time
print(f"Matched filtering completed in {mf_time:.4f} seconds")
print(f"Matched filter output matrix shape: {mf_output_matrix.shape}")

# ============================================================================
# PERFORM 2D FFT PROCESSING (DOPPLER PROCESSING)
# ============================================================================

print("\nPerforming Doppler processing (Slow-time FFT)...")

start_time = time.time()

# Apply FFT across pulses (slow-time dimension) for each range bin
range_doppler_matrix = np.fft.fft(mf_output_matrix, n=N_pulses, axis=0)

# Apply fftshift to center zero-Doppler frequency
range_doppler_matrix = np.fft.fftshift(range_doppler_matrix, axes=0)

doppler_time = time.time() - start_time
print(f"Doppler processing completed in {doppler_time:.4f} seconds")
print(f"Range-Doppler matrix shape: {range_doppler_matrix.shape}")

# ============================================================================
# CREATE RANGE AND VELOCITY AXES
# ============================================================================

# Range axis (from fast-time)
range_axis_km = range_axis / 1000  # Convert to km

# Doppler frequency axis (from slow-time FFT)
doppler_freq_axis = np.fft.fftshift(np.fft.fftfreq(N_pulses, PRI))

# Convert Doppler frequency to velocity: v = (fd * c) / (2 * fc)
velocity_axis = (doppler_freq_axis * c) / (2 * fc)

print(f"\nRange axis: {range_axis_km.min():.2f} to {range_axis_km.max():.2f} km")
print(f"Velocity axis: {velocity_axis.min():.2f} to {velocity_axis.max():.2f} m/s")

# ============================================================================
# CONVERT TO dB SCALE
# ============================================================================

# Get magnitude and convert to dB
range_doppler_magnitude = np.abs(range_doppler_matrix)
range_doppler_dB = 20 * np.log10(range_doppler_magnitude + 1e-10)  # Add small value to avoid log(0)

# Normalize dB scale
range_doppler_dB_normalized = range_doppler_dB - np.max(range_doppler_dB)

print(f"\ndB range: {range_doppler_dB_normalized.min():.2f} to {range_doppler_dB_normalized.max():.2f} dB")

# ============================================================================
# DETECT TARGETS IN RANGE-DOPPLER MAP
# ============================================================================

print("\n" + "=" * 70)
print("TARGET DETECTION IN RANGE-DOPPLER MAP")
print("=" * 70)

# Set threshold for detection (e.g., -20 dB below peak)
threshold_dB = np.max(range_doppler_dB_normalized) - 20

# Find local maxima above threshold
from scipy.ndimage import maximum_filter

# Apply local maximum filter
local_max = maximum_filter(range_doppler_dB_normalized, size=20)
detected_peaks = (range_doppler_dB_normalized == local_max) & (range_doppler_dB_normalized > threshold_dB)

# Get coordinates of detected targets
detected_coords = np.where(detected_peaks)
n_detected = len(detected_coords[0])

print(f"\nNumber of detected targets: {n_detected}")
print(f"Detection threshold: {threshold_dB:.2f} dB\n")

detected_targets = []
for i in range(n_detected):
    vel_idx = detected_coords[0][i]
    range_idx = detected_coords[1][i]
    
    detected_range = range_axis[range_idx]
    detected_velocity = velocity_axis[vel_idx]
    magnitude_dB = range_doppler_dB_normalized[vel_idx, range_idx]
    
    detected_targets.append({
        'range': detected_range,
        'velocity': detected_velocity,
        'magnitude_dB': magnitude_dB,
        'range_idx': range_idx,
        'vel_idx': vel_idx
    })
    
    print(f"Target {i+1}:")
    print(f"  Range: {detected_range:.2f} m ({detected_range/1000:.4f} km)")
    print(f"  Velocity: {detected_velocity:.2f} m/s")
    print(f"  Magnitude: {magnitude_dB:.2f} dB")
    print()

# ============================================================================
# FIGURE 1: RANGE-DOPPLER MAPS (Full and Zoomed)
# ============================================================================

print("=" * 70)
print("GENERATING RANGE-DOPPLER MAP VISUALIZATION")
print("=" * 70)

fig1 = plt.figure(figsize=(18, 8))

# Adjust subplot spacing for better clarity
plt.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.92, 
                   wspace=0.35)

# ============================================================================
# Plot 1: Full Range-Doppler Map
# ============================================================================
ax1 = plt.subplot(1, 2, 1)

# Create mesh grid for pcolormesh
extent = [range_axis_km.min(), range_axis_km.max(), 
          velocity_axis.min(), velocity_axis.max()]

im1 = ax1.imshow(range_doppler_dB_normalized, 
                 aspect='auto',
                 extent=extent,
                 origin='lower',
                 cmap='jet',
                 vmin=-60, vmax=0)

# Mark true target positions
for target_name, params in targets.items():
    ax1.plot(params['range']/1000, params['velocity'], 'g*', 
             markersize=20, markeredgecolor='white', markeredgewidth=2,
             label=f"{target_name} (True)")

# Mark detected targets
for i, det in enumerate(detected_targets):
    ax1.plot(det['range']/1000, det['velocity'], 'rx', 
             markersize=15, markeredgewidth=3,
             label=f"Detected {i+1}" if i == 0 else "")

ax1.set_xlabel('Range (km)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Velocity (m/s)', fontsize=13, fontweight='bold')
ax1.set_title('Range-Doppler Map (Full View)', fontsize=15, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper right', fontsize=10)

# Add colorbar
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Magnitude (dB)', fontsize=12, fontweight='bold')

# ============================================================================
# Plot 2: Zoomed Range-Doppler Map (Region of Interest)
# ============================================================================
ax2 = plt.subplot(1, 2, 2)

# Zoom to region around targets
range_zoom_min = 0.08  # km
range_zoom_max = 0.17  # km
vel_zoom_min = -60
vel_zoom_max = 80

extent_zoom = [range_zoom_min, range_zoom_max, vel_zoom_min, vel_zoom_max]

# Find indices for zoomed region
range_zoom_idx = (range_axis_km >= range_zoom_min) & (range_axis_km <= range_zoom_max)
vel_zoom_idx = (velocity_axis >= vel_zoom_min) & (velocity_axis <= vel_zoom_max)

# Extract zoomed data
zoomed_data = range_doppler_dB_normalized[np.ix_(vel_zoom_idx, range_zoom_idx)]

im2 = ax2.imshow(zoomed_data,
                 aspect='auto',
                 extent=extent_zoom,
                 origin='lower',
                 cmap='jet',
                 vmin=-60, vmax=0)

# Mark true target positions
for target_name, params in targets.items():
    ax2.plot(params['range']/1000, params['velocity'], 'g*', 
             markersize=25, markeredgecolor='white', markeredgewidth=2.5,
             label=f"{target_name}")
    # Add text annotation
    ax2.annotate(target_name, 
                xy=(params['range']/1000, params['velocity']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7))

# Mark detected targets
for i, det in enumerate(detected_targets):
    if (range_zoom_min <= det['range']/1000 <= range_zoom_max and 
        vel_zoom_min <= det['velocity'] <= vel_zoom_max):
        ax2.plot(det['range']/1000, det['velocity'], 'rx', 
                markersize=18, markeredgewidth=3.5)

ax2.set_xlabel('Range (km)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Velocity (m/s)', fontsize=13, fontweight='bold')
ax2.set_title('Range-Doppler Map (Zoomed)', fontsize=15, fontweight='bold')
ax2.grid(True, alpha=0.4, linestyle='--', linewidth=1.5)
ax2.legend(loc='upper right', fontsize=10)

# Add colorbar
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Magnitude (dB)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('./Q4/Range_Doppler_Map_2D.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# FIGURE 2: RANGE AND VELOCITY PROFILES
# ============================================================================

fig2 = plt.figure(figsize=(18, 7))

# Adjust subplot spacing
plt.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.90, 
                   wspace=0.35)

# ============================================================================
# Plot 3: Range Profile (summed across velocities)
# ============================================================================
ax3 = plt.subplot(1, 2, 1)

range_profile = np.sum(range_doppler_magnitude, axis=0)

# Add noise to range profile for realism
noise_level_range = 0.05 * np.max(range_profile)
range_noise = noise_level_range * np.random.randn(len(range_profile))
range_profile_noisy = range_profile + np.abs(range_noise)

range_profile_dB = 20 * np.log10(range_profile_noisy + 1e-10)
range_profile_dB -= np.max(range_profile_dB)

ax3.plot(range_axis_km, range_profile_dB, 'b-', linewidth=2.5, label='Range Profile', alpha=0.8)
ax3.fill_between(range_axis_km, range_profile_dB, -65, alpha=0.2, color='blue')

# Mark true ranges
for i, (target_name, params) in enumerate(targets.items()):
    label = f"{target_name}" if i < 2 else None
    ax3.axvline(x=params['range']/1000, color='g', linestyle='--', 
                linewidth=3, alpha=0.7, label=label)

ax3.set_xlabel('Range (km)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Magnitude (dB)', fontsize=13, fontweight='bold')
ax3.set_title('Range Profile (Integrated over Velocity)', fontsize=15, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle=':', linewidth=1)
ax3.legend(fontsize=11, loc='upper right')
ax3.set_xlim([0.05, 0.25])
ax3.set_ylim([-20, 10])

# ============================================================================
# Plot 4: Velocity Profile (summed across ranges)
# ============================================================================
ax4 = plt.subplot(1, 2, 2)

velocity_profile = np.sum(range_doppler_magnitude, axis=1)

# Add noise to velocity profile for realism
noise_level_vel = 0.05 * np.max(velocity_profile)
vel_noise = noise_level_vel * np.random.randn(len(velocity_profile))
velocity_profile_noisy = velocity_profile + np.abs(vel_noise)

velocity_profile_dB = 20 * np.log10(velocity_profile_noisy + 1e-10)
velocity_profile_dB -= np.max(velocity_profile_dB)

ax4.plot(velocity_axis, velocity_profile_dB, 'r-', linewidth=2.5, label='Velocity Profile', alpha=0.8)
ax4.fill_between(velocity_axis, velocity_profile_dB, -65, alpha=0.2, color='red')

# Mark true velocities
for i, (target_name, params) in enumerate(targets.items()):
    label = f"{target_name}" if i < 2 else None
    ax4.axvline(x=params['velocity'], color='g', linestyle='--', 
                linewidth=3, alpha=0.7, label=label)

ax4.set_xlabel('Velocity (m/s)', fontsize=13, fontweight='bold')
ax4.set_ylabel('Magnitude (dB)', fontsize=13, fontweight='bold')
ax4.set_title('Velocity Profile (Integrated over Range)', fontsize=15, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle=':', linewidth=1)
ax4.legend(fontsize=11, loc='upper right')
ax4.set_xlim([-50, 75])
ax4.set_ylim([-20, 10])

plt.tight_layout()
plt.savefig('./Q4/Range_Velocity_Profiles.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# COMPARISON TABLE: DETECTED VS TRUE TARGETS
# ============================================================================

print("\n" + "=" * 70)
print("DETECTION RESULTS COMPARISON TABLE")
print("=" * 70)
print(f"{'Target':<12} {'True Range':<15} {'True Vel':<15} {'Det Range':<15} {'Det Vel':<15} {'Range Err':<15} {'Vel Err':<15}")
print(f"{'      ':<12} {'(m)':<15} {'(m/s)':<15} {'(m)':<15} {'(m/s)':<15} {'(m)':<15} {'(m/s)':<15}")
print("-" * 110)

# Match detected targets to true targets
for target_name, params in targets.items():
    true_range = params['range']
    true_vel = params['velocity']
    
    # Find closest detected target
    min_dist = float('inf')
    closest_det = None
    
    for det in detected_targets:
        # Distance metric in normalized space
        range_dist = abs(det['range'] - true_range) / true_range
        vel_dist = abs(det['velocity'] - true_vel) / (abs(true_vel) + 1)
        dist = np.sqrt(range_dist**2 + vel_dist**2)
        
        if dist < min_dist:
            min_dist = dist
            closest_det = det
    
    if closest_det:
        range_error = closest_det['range'] - true_range
        vel_error = closest_det['velocity'] - true_vel
        
        print(f"{target_name:<12} {true_range:<15.2f} {true_vel:<15.2f} "
              f"{closest_det['range']:<15.2f} {closest_det['velocity']:<15.2f} "
              f"{range_error:<15.2f} {vel_error:<15.2f}")
    else:
        print(f"{target_name:<12} {true_range:<15.2f} {true_vel:<15.2f} "
              f"{'NOT DETECTED':<15} {'NOT DETECTED':<15} {'N/A':<15} {'N/A':<15}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(f"\nRange-Doppler Map Dimensions:")
print(f"  Range bins: {len(range_axis)}")
print(f"  Doppler bins: {N_pulses}")
print(f"  Total map size: {range_doppler_matrix.shape}")

print(f"\nResolution:")
range_resolution = c / (2 * fs)
velocity_resolution = (c * PRF) / (2 * fc * N_pulses)
print(f"  Range resolution: {range_resolution:.4f} m")
print(f"  Velocity resolution: {velocity_resolution:.4f} m/s")

print(f"\nProcessing Time:")
total_time = mf_time + doppler_time
print(f"  Range processing (matched filtering): {mf_time:.4f} seconds")
print(f"  Doppler processing (FFT): {doppler_time:.4f} seconds")
print(f"  Total processing time: {total_time:.4f} seconds")

print(f"\nDetection Performance:")
print(f"  Number of true targets: {len(targets)}")
print(f"  Number of detected targets: {n_detected}")
print(f"  Detection threshold: {threshold_dB:.2f} dB")

print("\n" + "=" * 70)
print("RANGE-DOPPLER MAP GENERATION COMPLETE")
print("=" * 70)