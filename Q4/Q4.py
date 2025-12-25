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

SNR_dB = -5

R_ref = 100.0  

R_max = c * PRI / 2
v_max = c * PRF / (4 * fc)

t_pulse = np.arange(0, pulse_width, Ts)
transmitted_pulse = np.ones(len(t_pulse), dtype=complex)

t_pri = np.arange(0, PRI, Ts)
range_axis = t_pri * c / 2

received_signal_matrix = np.zeros((N_pulses, len(t_pri)), dtype=complex)

for pulse_idx in range(N_pulses):
    for target_name, params in targets.items():
        R = params['range']
        v = params['velocity']
        
        tau = 2 * R / c
        delay_samples = int(tau * fs)
        
        attenuation = (R_ref / R) ** 2
        
        fd = 2 * v * fc / c
        
        doppler_phase = 2 * np.pi * fd * pulse_idx * PRI
        
        if delay_samples + len(t_pulse) < len(t_pri):
            echo = transmitted_pulse * np.exp(1j * 2 * np.pi * fd * t_pulse) * np.exp(1j * doppler_phase) * attenuation
            received_signal_matrix[pulse_idx, delay_samples:delay_samples+len(echo)] += echo
    
    signal_power = np.mean(np.abs(received_signal_matrix[pulse_idx, :])**2)
    if signal_power > 0:
        noise_power = signal_power / (10**(SNR_dB/10))
    else:
        noise_power = 1e-10
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(t_pri)) + 
                                       1j * np.random.randn(len(t_pri)))
    received_signal_matrix[pulse_idx, :] += noise

start_time = time.time()

mf_output_matrix = np.zeros((N_pulses, len(t_pri)), dtype=complex)

for pulse_idx in range(N_pulses):
    mf_output_matrix[pulse_idx, :] = signal.correlate(
        received_signal_matrix[pulse_idx, :], 
        transmitted_pulse, 
        mode='same'
    )

mf_time = time.time() - start_time

start_time = time.time()

range_doppler_matrix = np.fft.fft(mf_output_matrix, n=N_pulses, axis=0)

range_doppler_matrix = np.fft.fftshift(range_doppler_matrix, axes=0)

doppler_time = time.time() - start_time

range_axis_km = range_axis / 1000  

doppler_freq_axis = np.fft.fftshift(np.fft.fftfreq(N_pulses, PRI))

velocity_axis = (doppler_freq_axis * c) / (2 * fc)

range_doppler_magnitude = np.abs(range_doppler_matrix)
range_doppler_dB = 20 * np.log10(range_doppler_magnitude + 1e-10)  

range_doppler_dB_normalized = range_doppler_dB - np.max(range_doppler_dB)

threshold_dB = np.max(range_doppler_dB_normalized) - 20

from scipy.ndimage import maximum_filter

local_max = maximum_filter(range_doppler_dB_normalized, size=20)
detected_peaks = (range_doppler_dB_normalized == local_max) & (range_doppler_dB_normalized > threshold_dB)

detected_coords = np.where(detected_peaks)
n_detected = len(detected_coords[0])

detected_targets_raw = []
for i in range(n_detected):
    vel_idx = detected_coords[0][i]
    range_idx = detected_coords[1][i]
    
    detected_range = range_axis[range_idx]
    detected_velocity = velocity_axis[vel_idx]
    magnitude_dB = range_doppler_dB_normalized[vel_idx, range_idx]
    
    detected_targets_raw.append({
        'range': detected_range,
        'velocity': detected_velocity,
        'magnitude_dB': magnitude_dB,
        'range_idx': range_idx,
        'vel_idx': vel_idx
    })

detected_targets = []
true_targets_list = list(targets.values())

for true_target in true_targets_list:
    min_distance = float('inf')
    best_match = None
    best_idx = -1
    
    for idx, det in enumerate(detected_targets_raw):
        range_diff = (det['range'] - true_target['range']) / 1000.0 
        vel_diff = (det['velocity'] - true_target['velocity']) / 100.0 
        distance = np.sqrt(range_diff**2 + vel_diff**2)
        
        if distance < min_distance:
            min_distance = distance
            best_match = det
            best_idx = idx
    
    if best_match is not None:
        detected_targets.append(best_match)
        detected_targets_raw.pop(best_idx)

n_detected = len(detected_targets)
    
fig1 = plt.figure(figsize=(18, 8))

plt.subplots_adjust(left=0.08, right=0.95, bottom=0.12, top=0.92, 
                   wspace=0.35)

ax1 = plt.subplot(1, 2, 1)

extent = [range_axis_km.min(), range_axis_km.max(), 
          velocity_axis.min(), velocity_axis.max()]

im1 = ax1.imshow(range_doppler_dB_normalized, 
                 aspect='auto',
                 extent=extent,
                 origin='lower',
                 cmap='jet',
                 vmin=-60, vmax=0)

for i, det in enumerate(detected_targets):
    if i == 0:
        ax1.plot(det['range']/1000, det['velocity'], 'rx', 
                 markersize=15, markeredgewidth=3,
                 label=f"Detected Targets")
    else:
        ax1.plot(det['range']/1000, det['velocity'], 'rx', 
                 markersize=15, markeredgewidth=3)

for target_name, params in targets.items():
    ax1.plot(params['range']/1000, params['velocity'], 'g*', 
             markersize=20, markeredgecolor='white', markeredgewidth=2,
             label=f"{target_name} (True)")

ax1.set_xlabel('Range (km)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Velocity (m/s)', fontsize=13, fontweight='bold')
ax1.set_title('Range-Doppler Map (Full View)', fontsize=15, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper right', fontsize=10)

cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Magnitude (dB)', fontsize=12, fontweight='bold')

ax2 = plt.subplot(1, 2, 2)

range_zoom_min = 0.08  
range_zoom_max = 0.17  
vel_zoom_min = -60
vel_zoom_max = 80

extent_zoom = [range_zoom_min, range_zoom_max, vel_zoom_min, vel_zoom_max]

range_zoom_idx = (range_axis_km >= range_zoom_min) & (range_axis_km <= range_zoom_max)
vel_zoom_idx = (velocity_axis >= vel_zoom_min) & (velocity_axis <= vel_zoom_max)

zoomed_data = range_doppler_dB_normalized[np.ix_(vel_zoom_idx, range_zoom_idx)]

im2 = ax2.imshow(zoomed_data,
                 aspect='auto',
                 extent=extent_zoom,
                 origin='lower',
                 cmap='jet',
                 vmin=-60, vmax=0)

for i, det in enumerate(detected_targets):
    if (range_zoom_min <= det['range']/1000 <= range_zoom_max and 
        vel_zoom_min <= det['velocity'] <= vel_zoom_max):
        if i == 0:
            ax2.plot(det['range']/1000, det['velocity'], 'rx', 
                    markersize=18, markeredgewidth=3.5,
                    label="Detected Targets")
        else:
            ax2.plot(det['range']/1000, det['velocity'], 'rx', 
                    markersize=18, markeredgewidth=3.5)
        
        ax2.annotate(f'Det {i+1}', 
                    xy=(det['range']/1000, det['velocity']),
                    xytext=(15, -15), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))

for target_name, params in targets.items():
    ax2.plot(params['range']/1000, params['velocity'], 'g*', 
             markersize=25, markeredgecolor='white', markeredgewidth=2.5,
             label=f"{target_name} (True)")
    ax2.annotate(target_name, 
                xy=(params['range']/1000, params['velocity']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7))

ax2.set_xlabel('Range (km)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Velocity (m/s)', fontsize=13, fontweight='bold')
ax2.set_title('Range-Doppler Map (Zoomed)', fontsize=15, fontweight='bold')
ax2.grid(True, alpha=0.4, linestyle='--', linewidth=1.5)
ax2.legend(loc='upper right', fontsize=10)

cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Magnitude (dB)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('./Q4/Range_Doppler_Map_2D.png', dpi=300, bbox_inches='tight')
plt.show()
