import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

c = 3e8                     
fc = 77e9                   
PRI = 5e-6                  
PRF = 1 / PRI               
fs = 1 / 0.5e-9             
pulse_width = 4e-9          
bandwidth = 10e6            
N_pulses = 128             

targets = {
    'Target 1': {'range': 100.0, 'velocity': 55.0},
    'Target 2': {'range': 150.0, 'velocity': -30.0}
}

SNR_dB = -15

t_pulse = np.arange(0, pulse_width, 1/fs)
transmitted_pulse = np.ones(len(t_pulse), dtype=complex)

t_pri = np.arange(0, PRI, 1/fs)
range_axis = t_pri * c / 2

received_signal_matrix = np.zeros((N_pulses, len(t_pri)), dtype=complex)

for pulse_idx in range(N_pulses):
    for target_name, params in targets.items():
        R = params['range']
        v = params['velocity']

        tau = 2 * R / c
        delay_samples = int(tau * fs)

        # Range-dependent attenuation: 1/R^2
        # Normalize by a reference range to keep values reasonable
        R_ref = 100.0  # Reference range in meters
        attenuation = (R_ref / R) ** 2

        fd = 2 * v * fc / c
        doppler_phase = 2 * np.pi * fd * pulse_idx * PRI

        if delay_samples + len(t_pulse) < len(t_pri):
            echo = transmitted_pulse * np.exp(
                1j * 2 * np.pi * fd * t_pulse
            ) * np.exp(1j * doppler_phase) * attenuation

            received_signal_matrix[pulse_idx,
                delay_samples:delay_samples + len(echo)] += echo

    signal_power = np.mean(np.abs(received_signal_matrix[pulse_idx])**2)
    noise_power = signal_power / (10**(SNR_dB/10))
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(len(t_pri)) +
        1j * np.random.randn(len(t_pri))
    )
    received_signal_matrix[pulse_idx] += noise


mf_output_matrix = np.zeros((N_pulses, len(t_pri)), dtype=complex)

for pulse_idx in range(N_pulses):
    mf_output_matrix[pulse_idx] = signal.correlate(
        received_signal_matrix[pulse_idx],
        transmitted_pulse,
        mode='same'
    )


mf_magnitude = np.abs(mf_output_matrix[0])
P_fa = 1e-6

pulse_energy_gain = np.sum(np.abs(transmitted_pulse)**2)
noise_power_output = noise_power * pulse_energy_gain

threshold = np.sqrt(-noise_power_output * np.log(P_fa))


peaks, _ = signal.find_peaks(
    mf_magnitude,
    height=threshold,
    distance=int(pulse_width * fs)
)

detected_ranges = range_axis[peaks]


doppler_freq_axis = np.fft.fftshift(np.fft.fftfreq(N_pulses, PRI))
velocity_axis = (doppler_freq_axis * c) / (2 * fc)

detected_velocities = []
velocity_errors = []
target_classifications = []

fig, axes = plt.subplots(len(peaks), 1, figsize=(14, 8))
if len(peaks) == 1:
    axes = [axes]

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

    true_velocity = list(targets.values())[idx]['velocity']
    velocity_error = abs(detected_velocity - true_velocity)
    velocity_errors.append(velocity_error)

    classification = (
        "Approaching" if detected_velocity > 0 else
        "Receding" if detected_velocity < 0 else
        "Stationary"
    )
    target_classifications.append(classification)

    ax = axes[idx]
    
    ax.plot(velocity_axis, doppler_magnitude, color='blue', linewidth=1.5, label='Doppler Spectrum')
    
    ax.plot(detected_velocity, doppler_magnitude[peak_doppler_idx], 'ro', markersize=12, 
            label=f'Detected: {detected_velocity:.2f} m/s', zorder=5)
    
    ax.axvline(x=true_velocity, color='green', linestyle=':', linewidth=3, 
               label=f'True: {true_velocity:.2f} m/s', zorder=4)
               

    range_km = detected_ranges[idx] / 1000.0  
    ax.set_title(f"Target {idx+1} - Doppler Spectrum (Range: {range_km:.2f} km)", 
                 fontsize=14, fontweight='bold')
    
    ax.text(0.02, 0.90, f"Error: {velocity_error:.2f} m/s", transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', bbox=box_props)
    
    ax.text(0.02, 0.78, f"Classification: {classification}", transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', bbox=box_props)

    ax.set_xlabel("Velocity (m/s)", fontsize=12)
    ax.set_ylabel("Magnitude", fontsize=12)
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlim(-100, 150)
    
    ax.legend(loc='upper right', framealpha=1, fancybox=True, fontsize=10)

plt.tight_layout()
plt.savefig("./Q3/Velocity_Estimation_using_Doppler_Analysis.png")
plt.show()