# 🎯 Target Detection and Velocity Estimation using Pulse-Doppler Radar

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-Scientific_Computing-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/SciPy-Signal_Processing-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Domain-Radar_DSP-red?style=for-the-badge"/>
</p>

---

## 📖 Overview

This project implements a complete **Pulse-Doppler Radar signal processing pipeline** in Python, simulating the end-to-end chain from pulse transmission through to target detection, range estimation, velocity measurement, and Range-Doppler mapping.

The system models a **77 GHz millimeter-wave radar** — the same frequency band used in modern automotive radar and advanced driver assistance systems (ADAS) — and places two moving targets in a noisy environment. Each module progressively builds upon the previous, culminating in a 2D Range-Doppler map capable of simultaneously detecting and resolving multiple targets in both range and velocity.

The project is structured as four progressive questions (Q1–Q4), each tackling a distinct stage of the radar signal processing chain.

---

## 🌐 Radar System Specifications

| Parameter | Value |
|---|---|
| Carrier Frequency | 77 GHz |
| Sampling Interval | 0.5 ns |
| Sampling Rate | 2 GS/s |
| Pulse Width | 4 ns |
| Pulse Repetition Interval (PRI) | 5 µs |
| Pulse Repetition Frequency (PRF) | 200 kHz |
| Signal-to-Noise Ratio (SNR) | –5 to –15 dB (varying by module) |
| Number of Pulses | 128–780 |

### Simulated Targets

| Target | Range | Velocity | Direction |
|---|---|---|---|
| Target 1 | 100 m | +55 m/s | Approaching |
| Target 2 | 150 m | –30 m/s | Receding |

---

## 🗂️ Project Structure

```
Target-Detection-and-Velocity-Estimation-using-Pulse-Doppler-Radar/
│
├── Q1/
│   ├── Q1.py                                   # Pulse simulation & range-time diagram
│   ├── plot115_transmitted_pulse.png            # Output: Transmitted pulse waveform
│   ├── plot215_received_signal_single_pulse.png # Output: Received echo signal
│   └── plot315_range_time_diagram.png           # Output: Range–Time diagram
│
├── Q2/
│   ├── Q2.py                                   # Target detection via matched filtering
│   └── comparison_methods_radar_detection.png  # Output: Time-domain vs FFT comparison
│
├── Q3/
│   ├── Q3.py                                   # Velocity estimation via Doppler analysis
│   └── Velocity_Estimation_using_Doppler_Analysis.png  # Output: Doppler spectra
│
├── Q4/
│   ├── Q4.py                                   # 2D Range-Doppler map generation
│   └── Range_Doppler_Map_2D.png                # Output: Full + zoomed RD map
│
└── README.md
```

---

## 🔬 Module Breakdown

### Q1 — Pulse Simulation & Signal Visualization

**File:** `Q1/Q1.py`

This module lays the foundation of the radar system. It simulates the transmission of rectangular pulses and models the echoes returned from two moving targets embedded in additive complex Gaussian noise.

**Key steps:**
- Generates a rectangular baseband transmit pulse of width 4 ns.
- Computes round-trip time delays and Doppler phase shifts for both targets.
- Applies range-dependent attenuation proportional to `1/R²`.
- Adds complex white Gaussian noise at SNR = –10 dB.
- Builds a (780 pulses × fast-time samples) received signal matrix.

**Outputs:**
- `plot115_transmitted_pulse.png` — A step-plot of the transmitted pulse in the time domain.
- `plot215_received_signal_single_pulse.png` — The real part of a single received pulse with annotated target echo positions.
- `plot315_range_time_diagram.png` — A 2D heatmap showing echo magnitude across all pulses (slow time) vs. range (fast time), revealing the Doppler-induced modulation across pulses.

---

### Q2 — Range Detection via Matched Filtering

**File:** `Q2/Q2.py`

This module implements two matched filtering approaches for range detection and benchmarks their performance side by side.

**Methods implemented:**
1. **Time-Domain Matched Filter** — Direct cross-correlation of the received signal with the transmitted pulse replica using `scipy.signal.correlate`.
2. **FFT-Based Matched Filter** — Frequency-domain implementation using pointwise multiplication of the signal spectrum with the conjugate of the pulse spectrum, followed by IFFT.

**Detection logic:**
- A **CFAR-inspired threshold** is derived analytically from the false alarm probability `P_fa = 1e-6` and the estimated noise power at the matched filter output.
- Peak detection (`scipy.signal.find_peaks`) identifies detected targets above the threshold.

**Outputs:**
- `comparison_methods_radar_detection.png` — A 3-panel figure showing:
  - Method 1 output with detected peaks and threshold.
  - Method 2 output with detected peaks and threshold.
  - Normalized overlay of both methods zoomed into the target range window (90–160 m) for visual comparison.

---

### Q3 — Velocity Estimation via Doppler Analysis

**File:** `Q3/Q3.py`

Building on the range detection from Q2, this module estimates the radial velocity of each detected target by analyzing phase variations across the slow-time (pulse) dimension — the classic **Doppler processing** step.

**Processing chain:**
1. Constructs a full (N_pulses × fast-time) signal matrix with per-pulse Doppler phase accumulation.
2. Applies a matched filter to each pulse to produce a matched-filter output matrix.
3. For each range-detected target, extracts the slow-time signal at that range bin.
4. Applies an FFT along the slow-time axis to obtain the **Doppler spectrum**.
5. Identifies the spectral peak to estimate velocity.
6. Classifies each target as **Approaching**, **Receding**, or **Stationary** based on the sign of the Doppler shift.

**Outputs:**
- `Velocity_Estimation_using_Doppler_Analysis.png` — Per-target subplots showing the Doppler spectrum, the detected velocity peak, the true velocity reference line, estimation error, and motion classification label.

---

### Q4 — 2D Range-Doppler Map

**File:** `Q4/Q4.py`

The most comprehensive module, Q4 combines range and Doppler processing into a unified **2D Range-Doppler Map** — the standard visualization used in operational radar systems to simultaneously resolve targets in both range and velocity.

**Processing pipeline:**
1. Simulates a full multi-pulse received signal matrix (780 pulses × fast-time).
2. Applies matched filtering along the fast-time axis to achieve range compression.
3. Applies FFT along the slow-time (pulse index) axis to achieve Doppler compression.
4. `fftshift` recenters zero Doppler at the map center.
5. Converts to dB scale and normalizes relative to the peak.
6. Applies a **2D local maximum filter** (`scipy.ndimage.maximum_filter`) with a –20 dB threshold for peak detection.
7. Associates detected peaks with ground-truth targets using a normalized Euclidean distance metric in range-velocity space.

**Outputs:**
- `Range_Doppler_Map_2D.png` — A side-by-side figure showing:
  - **Full RD Map** — Jet colormap over the complete unambiguous range and velocity space, with detected peaks (red ×) and true target locations (green ★).
  - **Zoomed RD Map** — Magnified view around the target region (80–170 m, –60 to +80 m/s) with annotated detection and ground-truth labels.

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip

### Install Dependencies

```bash
pip install numpy scipy matplotlib
```

Or, if you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

> No additional hardware or radar datasets are needed — all signals are fully simulated in software.

---

## ▶️ How to Run

Each question is a standalone script. Run them in order for the full pipeline experience:

```bash
# Q1 — Simulate pulse transmission and visualize echoes
python Q1/Q1.py

# Q2 — Detect targets via matched filtering (time-domain and FFT)
python Q2/Q2.py

# Q3 — Estimate target velocities via Doppler analysis
python Q3/Q3.py

# Q4 — Generate 2D Range-Doppler map
python Q4/Q4.py
```

Each script will display plots interactively and save `.png` output files to its respective folder.

> **Note:** Scripts save output images relative to the project root. Run them from the project root directory or adjust the output paths in the code accordingly.

---

## 📊 Key Concepts Explained

### Pulse-Doppler Radar
A pulse-Doppler radar emits a series of coherent pulses and processes the echoes in both the fast-time (range) and slow-time (Doppler) dimensions. The fast-time axis resolves target range via time delay, while the slow-time axis resolves radial velocity via the Doppler effect.

### Matched Filtering
The matched filter is the optimal linear filter for detecting a known signal in additive white Gaussian noise. It maximizes SNR at the moment of detection and is implemented either as a direct cross-correlation or an FFT-based frequency-domain multiplication.

### Doppler Effect in Radar
A moving target introduces a phase shift between successive pulses proportional to its radial velocity. By taking the FFT across pulses at a given range bin, this phase progression manifests as a spectral peak at the Doppler frequency `f_d = 2v·fc/c`, from which velocity is recovered directly.

### Range-Doppler Map
A 2D matrix obtained by applying matched filtering in fast-time and FFT in slow-time. It provides a joint range-velocity representation of all targets in the scene, enabling simultaneous resolution of targets that are close in range but differ in velocity, or vice versa.

---

## 📈 Results Summary

| Module | Targets Detected | Method | SNR |
|---|---|---|---|
| Q2 | 2 / 2 | Time-Domain MF & FFT MF | –10 dB |
| Q3 | 2 / 2 | Doppler FFT per range bin | –15 dB |
| Q4 | 2 / 2 | 2D Range-Doppler + 2D CFAR | –5 dB |

Both targets are correctly detected and their velocities recovered with high fidelity across all modules, even under negative SNR conditions — demonstrating the effectiveness of coherent pulse-Doppler processing.

---

## 🛠️ Dependencies

| Library | Purpose |
|---|---|
| `numpy` | Array operations, FFT, signal math |
| `scipy.signal` | Matched filtering, peak detection |
| `scipy.ndimage` | 2D local maximum filter for RD map CFAR |
| `matplotlib` | All plots and visualizations |
| `time` | Execution timing for method comparison |

---

## 🔭 Potential Extensions

- **Chirp / LFM Pulse** — Replace the rectangular pulse with a linear frequency-modulated waveform for improved range resolution via pulse compression.
- **CFAR Detection** — Implement a proper Cell-Averaging CFAR (CA-CFAR) for adaptive threshold detection in non-uniform clutter.
- **Clutter Cancellation** — Add a Moving Target Indicator (MTI) filter to suppress stationary ground clutter before Doppler processing.
- **Ambiguity Function** — Plot the radar ambiguity function to visualize the range-Doppler resolution tradeoffs for the chosen waveform.
- **Multiple PRIs / Staggered PRF** — Address range and velocity ambiguities using staggered pulse repetition frequencies.
- **3D Scenario** — Extend to multiple targets with different angles and implement beamforming or angle estimation.

---
<p align="center">
  <i>Built with Python · Radar Signal Processing · Doppler Analysis · Scientific Visualization</i>
</p>
