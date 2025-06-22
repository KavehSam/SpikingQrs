import neurokit2 as nk
import numpy as np
import torch
import snntorch.spikegen as spikegen


def generate_spike_input(x, num_steps):
    """
    Generate Poisson-distributed spike trains for the input.
    :param x: Input tensor of shape [batch_size, num_inputs]
    :param num_steps: Number of time steps for spike generation
    :return: Spike trains of shape [num_steps, batch_size, num_inputs]
    """
    # Ensure x is a tensor and handle numpy arrays properly
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x) if hasattr(x, "__array__") else torch.tensor(x)

    x = x.float()

    # Normalize input to range [0, 1]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)

     # Replace NaNs with 0
    if torch.isnan(x_norm).any():
        print("NaNs detected. Cleaning data...")
        x_norm = torch.nan_to_num(x_norm, nan=0.0) 

    # Generate Poisson-distributed spike trains
    spk_input = spikegen.rate(x_norm, num_steps=num_steps)
    return spk_input, x_norm


def generate_challenging_ecg(
    duration=10, sampling_rate=500, noise=0.9, heart_rate=75, heart_rate_std=20
):
    """
    Generate an ECG signal with beats of varying amplitudes.

    Parameters:
    - duration (float): Length of the signal in seconds.
    - sampling_rate (int): Sampling frequency in Hz.

    Returns:
    - ecg_signal (array): ECG signal with varying beat amplitudes.
    """
    # Generate baseline ECG signal
    ecg_signal = nk.ecg_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        noise=noise,
        heart_rate=heart_rate,
        heart_rate_std=heart_rate_std,
    )

    # varying amplitude modulation
    time = np.linspace(0, duration, len(ecg_signal))
    modulation = 1 + 0.8 * np.sin(
        2 * np.pi * 0.5 * time
    ) 
    ecg_signal = ecg_signal * modulation

    # radomly reduce some beats' amplitude
    _, info = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
    r_peaks_indices = info["ECG_R_Peaks"]

    if r_peaks_indices.size > 0:
        for peak in r_peaks_indices:
            if np.random.rand() < 0.7:  # 30% chance to reduce a beat's amplitude
                start = max(0, peak - 20)
                end = min(len(ecg_signal), peak + 20)
                ecg_signal[start:end] *= np.random.uniform(0.3, 0.5)

    return ecg_signal
