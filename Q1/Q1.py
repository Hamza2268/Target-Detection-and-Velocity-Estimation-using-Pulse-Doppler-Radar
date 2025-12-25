import numpy as np
import matplotlib.pyplot as plt

SPEED_OF_LIGHT = 3e8  
carrier_frequency = 77e9        
sampling_interval = 0.5e-9      
sampling_rate = 1 / sampling_interval

pulse_width = 4e-9              
pulse_amplitude = 1.0

pulse_repetition_interval = 5e-6    
pulse_repetition_frequency = 1 / pulse_repetition_interval

num_pulses = 780

target_ranges = np.array([100.0, 150])     
target_velocities = np.array([55.0, -30.0])

round_trip_delays = 2 * target_ranges / SPEED_OF_LIGHT
doppler_frequencies = 2 * target_velocities * carrier_frequency / SPEED_OF_LIGHT

attenuation = 1 / (target_ranges ** 2)

fast_time = np.arange(0, pulse_repetition_interval, sampling_interval)
num_fast_samples = len(fast_time)

pulse_sample_count = int(pulse_width * sampling_rate)

transmit_pulse = np.zeros(num_fast_samples, dtype=complex)
transmit_pulse[:pulse_sample_count] = pulse_amplitude

received_signal = np.zeros((num_pulses, num_fast_samples), dtype=complex)

for pulse_index in range(num_pulses):
    slow_time = pulse_index * pulse_repetition_interval

    for target_index in range(len(target_ranges)):
        delay_samples = int(round_trip_delays[target_index] * sampling_rate)

        if delay_samples + pulse_sample_count < num_fast_samples:
            doppler_phase = np.exp(
                1j * 2 * np.pi * doppler_frequencies[target_index] * slow_time
            )
            received_signal[pulse_index,
                            delay_samples:delay_samples + pulse_sample_count] += \
                attenuation[target_index] * transmit_pulse[:pulse_sample_count] * doppler_phase

SNR_dB = -10

signal_power = np.mean(np.abs(received_signal) ** 2)
noise_power = signal_power / (10 ** (SNR_dB / 10))

noise = np.sqrt(noise_power / 2) * (
    np.random.randn(*received_signal.shape) +
    1j * np.random.randn(*received_signal.shape)
)

received_signal += noise


plt.figure(figsize=(10, 5))
plt.step(
    fast_time[:5 * pulse_sample_count] * 1e9,
    np.real(transmit_pulse[:5 * pulse_sample_count]),
    where='post',
    linewidth=2
)
plt.xlabel("Time (ns)", fontsize=12, fontweight='bold')
plt.ylabel("Amplitude", fontsize=12, fontweight='bold')
plt.title("Transmitted Rectangular Pulse (Real Part)", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./Q1/plot115_transmitted_pulse.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 5))

real_signal = np.real(received_signal[0])

plt.plot(
    fast_time * 1e6,
    real_signal,
    linewidth=1,
    color='blue',
    label='Received Signal'
)

max_amplitude = np.max(np.abs(real_signal))
plt.ylim(-max_amplitude * 1.8, max_amplitude * 1.8)

plt.axvline(
    round_trip_delays[0] * 1e6,
    linestyle='--',
    linewidth=1,
    color='red',
    label=f'Target 1 Echo (R = {target_ranges[0]:.1f} m)'
)

plt.axvline(
    round_trip_delays[1] * 1e6,
    linestyle='--',
    linewidth=1,
    color='green',
    label=f'Target 2 Echo (R = {target_ranges[1]:.1f} m)'
)

plt.xlabel("Time (µs)", fontsize=12, fontweight='bold')
plt.ylabel("Amplitude", fontsize=12, fontweight='bold')
plt.title("Received Signal – Single Pulse Period (Real Part)", fontsize=14, fontweight='bold')

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    './Q1/plot215_received_signal_single_pulse.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()

range_axis = fast_time * SPEED_OF_LIGHT / 2

max_range_m = 250
max_range_index = np.searchsorted(range_axis, max_range_m)

plt.figure(figsize=(12, 8))

plt.imshow(
    np.abs(received_signal[:, :max_range_index]),
    aspect='auto',
    extent=[0, max_range_m, num_pulses, 0],
    cmap='jet',
    interpolation='bilinear'
)

plt.xlabel("Range (m)", fontsize=12, fontweight='bold')
plt.ylabel("Pulse Index (Slow Time)", fontsize=12, fontweight='bold')
plt.title("Range–Time Diagram (All Pulses)", fontsize=14, fontweight='bold')

cbar = plt.colorbar(label="Magnitude")
cbar.set_label("Magnitude", fontsize=11, fontweight='bold')

plt.grid(True, alpha=0.2, color='white', linewidth=0.5)

plt.tight_layout()
plt.savefig('./Q1/plot315_range_time_diagram.png', dpi=300, bbox_inches='tight')
plt.show()

