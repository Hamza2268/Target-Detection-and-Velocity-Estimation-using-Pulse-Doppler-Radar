import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Physical constants
c = 3e8  # Speed of light (m/s)

# ==================== RADAR SYSTEM DESIGN ====================
print("=" * 70)
print("PULSE-DOPPLER RADAR SYSTEM DESIGN")
print("=" * 70)

# Radar Parameters (X-band)
fc = 10e9  # Carrier frequency: 10 GHz (X-band center)
PRI = 200e-6  # Pulse Repetition Interval: 200 μs
PRF = 1 / PRI  # Pulse Repetition Frequency
pulse_width = 1e-6  # Pulse width: 1 μs
fs = 100e6  # Sampling rate: 100 MHz
num_pulses = 128  # Number of pulses for coherent processing

print(f"\n1. RADAR PARAMETERS:")
print(f"   Carrier Frequency (fc): {fc/1e9:.1f} GHz")
print(f"   Pulse Repetition Interval (PRI): {PRI*1e6:.1f} μs")
print(f"   Pulse Repetition Frequency (PRF): {PRF/1e3:.2f} kHz")
print(f"   Pulse Width: {pulse_width*1e6:.1f} μs")
print(f"   Sampling Rate: {fs/1e6:.0f} MHz")
print(f"   Number of Pulses: {num_pulses}")

# Calculate Maximum Unambiguous Range and Velocity
R_max = (c * PRI) / 2
v_max = (c * PRF) / (4 * fc)

print(f"\n2. SYSTEM CONSTRAINTS:")
print(f"   Maximum Unambiguous Range (R_max): {R_max/1e3:.2f} km")
print(f"   Maximum Unambiguous Velocity (v_max): ±{v_max:.2f} m/s")

# ==================== TARGET SCENARIO ====================
print(f"\n3. TARGET SCENARIO:")

# Target 1: Approaching target
target1_range = 5000  # 5 km
target1_velocity = 50  # 50 m/s (approaching, positive)

# Target 2: Receding target
target2_range = 12000  # 12 km
target2_velocity = -30  # -30 m/s (receding, negative)

print(f"   Target 1:")
print(f"     - Range: {target1_range/1e3:.1f} km")
print(f"     - Velocity: {target1_velocity:+.1f} m/s (approaching)")
print(f"   Target 2:")
print(f"     - Range: {target2_range/1e3:.1f} km")
print(f"     - Velocity: {target2_velocity:+.1f} m/s (receding)")

# Calculate time delays and Doppler shifts
tau1 = 2 * target1_range / c
tau2 = 2 * target2_range / c
fd1 = 2 * target1_velocity * fc / c
fd2 = 2 * target2_velocity * fc / c

print(f"\n4. CALCULATED PARAMETERS:")
print(f"   Target 1:")
print(f"     - Time Delay (τ1): {tau1*1e6:.3f} μs")
print(f"     - Doppler Shift (fd1): {fd1/1e3:.3f} kHz")
print(f"   Target 2:")
print(f"     - Time Delay (τ2): {tau2*1e6:.3f} μs")
print(f"     - Doppler Shift (fd2): {fd2/1e3:.3f} kHz")

# ==================== SIGNAL GENERATION ====================
print(f"\n5. GENERATING RADAR SIGNALS...")

# Time vector for one PRI
t_pri = np.arange(0, PRI, 1/fs)

# Generate transmitted pulse (rectangular pulse)
tx_pulse = np.zeros(len(t_pri))
pulse_samples = int(pulse_width * fs)
tx_pulse[:pulse_samples] = np.exp(1j * 2 * np.pi * fc * t_pri[:pulse_samples])

# Initialize received signal matrix (slow-time × fast-time)
rx_signal = np.zeros((num_pulses, len(t_pri)), dtype=complex)

# Generate received signals for each pulse
for pulse_idx in range(num_pulses):
    # Slow time for this pulse
    slow_time = pulse_idx * PRI
    
    # Target 1 echo
    delay_samples1 = int(tau1 * fs)
    if delay_samples1 + pulse_samples < len(t_pri):
        doppler_phase1 = 2 * np.pi * fd1 * slow_time
        echo1 = tx_pulse[:pulse_samples] * np.exp(1j * doppler_phase1)
        rx_signal[pulse_idx, delay_samples1:delay_samples1+pulse_samples] += echo1 * 0.5
    
    # Target 2 echo
    delay_samples2 = int(tau2 * fs)
    if delay_samples2 + pulse_samples < len(t_pri):
        doppler_phase2 = 2 * np.pi * fd2 * slow_time
        echo2 = tx_pulse[:pulse_samples] * np.exp(1j * doppler_phase2)
        rx_signal[pulse_idx, delay_samples2:delay_samples2+pulse_samples] += echo2 * 0.3
    
    # Add noise
    noise = 0.1 * (np.random.randn(len(t_pri)) + 1j * np.random.randn(len(t_pri)))
    rx_signal[pulse_idx, :] += noise

# ==================== RANGE-DOPPLER PROCESSING ====================
print(f"6. PERFORMING RANGE-DOPPLER PROCESSING...")

# Apply FFT along slow-time (Doppler processing)
range_doppler_map = np.fft.fftshift(np.fft.fft(rx_signal, axis=0), axes=0)
range_doppler_magnitude = np.abs(range_doppler_map)

# Create range and Doppler axes
range_axis = (t_pri * c / 2) / 1e3  # Convert to km
doppler_freq_axis = np.fft.fftshift(np.fft.fftfreq(num_pulses, PRI))
velocity_axis = doppler_freq_axis * c / (2 * fc)  # Convert to m/s

# ==================== VISUALIZATION ====================
print(f"7. GENERATING PLOTS...")

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

# Plot 1: Transmitted Pulse
ax1 = fig.add_subplot(gs[0, 0])
time_us = t_pri[:pulse_samples*5] * 1e6
ax1.plot(time_us, np.real(tx_pulse[:pulse_samples*5]), 'b-', linewidth=1.5, label='Real')
ax1.plot(time_us, np.imag(tx_pulse[:pulse_samples*5]), 'r-', linewidth=1.5, label='Imaginary')
ax1.set_xlabel('Time (μs)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Transmitted Pulse (Rectangular)', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Received Signal (Single Pulse)
ax2 = fig.add_subplot(gs[0, 1])
plot_samples = min(5000, len(t_pri))
ax2.plot(t_pri[:plot_samples]*1e6, np.abs(rx_signal[0, :plot_samples]), 'g-', linewidth=1)
ax2.axvline(tau1*1e6, color='r', linestyle='--', linewidth=2, label=f'Target 1 ({target1_range/1e3:.1f} km)')
ax2.axvline(tau2*1e6, color='b', linestyle='--', linewidth=2, label=f'Target 2 ({target2_range/1e3:.1f} km)')
ax2.set_xlabel('Time (μs)')
ax2.set_ylabel('Amplitude')
ax2.set_title('Received Signal (Pulse #1)', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Range Profile (sum over Doppler)
ax3 = fig.add_subplot(gs[1, 0])
range_profile = np.sum(range_doppler_magnitude, axis=0)
ax3.plot(range_axis, 20*np.log10(range_profile + 1e-10), 'b-', linewidth=1.5)
ax3.axvline(target1_range/1e3, color='r', linestyle='--', linewidth=2, label='Target 1')
ax3.axvline(target2_range/1e3, color='g', linestyle='--', linewidth=2, label='Target 2')
ax3.set_xlabel('Range (km)')
ax3.set_ylabel('Magnitude (dB)')
ax3.set_title('Range Profile', fontweight='bold')
ax3.set_xlim([0, 20])
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Doppler Profile (sum over range)
ax4 = fig.add_subplot(gs[1, 1])
doppler_profile = np.sum(range_doppler_magnitude, axis=1)
ax4.plot(velocity_axis, 20*np.log10(doppler_profile + 1e-10), 'r-', linewidth=1.5)
ax4.axvline(target1_velocity, color='b', linestyle='--', linewidth=2, label=f'Target 1 ({target1_velocity:+.0f} m/s)')
ax4.axvline(target2_velocity, color='g', linestyle='--', linewidth=2, label=f'Target 2 ({target2_velocity:+.0f} m/s)')
ax4.set_xlabel('Velocity (m/s)')
ax4.set_ylabel('Magnitude (dB)')
ax4.set_title('Doppler Profile', fontweight='bold')
ax4.set_xlim([-100, 100])
ax4.grid(True, alpha=0.3)
ax4.legend()

# Plot 5: Range-Doppler Map
ax5 = fig.add_subplot(gs[2, :])
range_doppler_db = 20 * np.log10(range_doppler_magnitude.T + 1e-10)
im = ax5.imshow(range_doppler_db, aspect='auto', cmap='jet', 
                extent=[velocity_axis[0], velocity_axis[-1], range_axis[-1], range_axis[0]],
                vmin=np.max(range_doppler_db) - 40)
ax5.scatter([target1_velocity], [target1_range/1e3], c='white', s=200, marker='x', 
            linewidths=3, label='Target 1')
ax5.scatter([target2_velocity], [target2_range/1e3], c='yellow', s=200, marker='x', 
            linewidths=3, label='Target 2')
ax5.set_xlabel('Velocity (m/s)', fontsize=12)
ax5.set_ylabel('Range (km)', fontsize=12)
ax5.set_title('Range-Doppler Map', fontweight='bold', fontsize=14)
ax5.set_xlim([-100, 100])
ax5.set_ylim([20, 0])
cbar = plt.colorbar(im, ax=ax5)
cbar.set_label('Magnitude (dB)', fontsize=10)
ax5.legend(loc='upper right')
ax5.grid(True, alpha=0.3, color='white', linewidth=0.5)

plt.suptitle('Pulse-Doppler Radar System Analysis', fontsize=16, fontweight='bold', y=0.995)

plt.savefig('pulse_doppler_radar.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved as 'pulse_doppler_radar.png'")

plt.show()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)