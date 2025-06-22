"""
Set of functions for feature extraction for burst detection in raster plot of spiking activity.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import scipy.signal as signal


def streamingAvg(x: float, n: int, sum: float) -> float:
    sum = sum + x
    return float(sum) / n


def update_streaming_avg(x: float, n: int, streaming_sum: float) -> tuple[float, float]:
    streaming_avg = streamingAvg(x, n, streaming_sum)
    streaming_sum = streaming_avg * n

    return streaming_avg, streaming_sum


def is_burst_block_fp(prev_r_peak_fex: dict, cur_fex: dict) -> bool:
    """
    Check if the current burst block is a false positive compared to the previous r peak burst block.

    Args:
        prev_r_peak_fex (dict): Features of the previous r peak burst block.
        cur_fex (dict): Features of the current burst block.

    Returns:
        bool: True if the current burst block is a false positive, False otherwise.
    """
    # false positive if duration is neuron participation is less then last r peak burst
    if (cur_fex["Burst Duration"] < prev_r_peak_fex["Burst Duration"]) and (
        cur_fex["Neuron Participation"] < prev_r_peak_fex["Neuron Participation"]
    ):
        return True

    # false positive if 'ISI CV' is less than 1 and mean ISI is greater than last r peak burst
    if cur_fex["ISI CV"] < 1:
        if cur_fex["Mean ISI"] > prev_r_peak_fex["Mean ISI"]:
            return True

    return False


def is_burst_block_outlier(fex_avg: dict, cur_fex: dict) -> bool:
    """
    Check if the current burst block features are outliers compared to the average of burst block features.

    Args:
        fex_avg (dict): Average burst features.
        cur_fex (dict): Current burst features.

    Returns:
        bool: True if the current burst is an outlier, False otherwise.
    """

    # outlier if 'ISI CV' is less than 1 and mean ISI is greater than avg
    if cur_fex["ISI CV"] < 1:
        if cur_fex["Mean ISI"] > fex_avg["Mean ISI"]:
            return True

    # outlier if duration is neuron participation is less than avg
    if (
        (cur_fex["Burst Duration"] > fex_avg["Burst Duration"])
        and (cur_fex["Spike Rate"] < fex_avg["Spike Rate"])
        and (cur_fex["Neuron Participation"] < fex_avg["Neuron Participation"])
    ):
        return True

    # outlier if duration is neuron participation is less than avg
    if (cur_fex["Burst Duration"] < fex_avg["Burst Duration"]) and (
        cur_fex["ISI CV"] < fex_avg["ISI CV"]
    ):
        return True

    return False


def extract_features_for_burst(
    spike_data: np.ndarray, start: int, end: int, time_step: float
) -> dict:
    """
    Extracts features for a single burst block.

    Parameters:
    - spike_data: 2D NumPy array (neurons x time points).
    - start, end: Start and end indices of the burst.
    - time_step: Time resolution per bin.

    Returns:
    - Dictionary containing extracted burst features.
    """
    burst_spikes = spike_data[:, start:end]
    burst_duration = (end - start) * time_step
    total_spikes = np.sum(burst_spikes)
    spike_rate = total_spikes / (burst_spikes.shape[0] * burst_duration)
    active_neurons = np.sum(np.any(burst_spikes > 0, axis=1))
    neuron_participation = active_neurons / burst_spikes.shape[0]

    # Inter-Spike Interval (ISI)
    isi_list = []
    for neuron in range(burst_spikes.shape[0]):
        spike_times = np.where(burst_spikes[neuron, :] > 0)[0] * time_step
        if len(spike_times) > 1:
            isi_list.extend(np.diff(spike_times))

    mean_isi = np.mean(isi_list) if isi_list else 0
    cv_isi = np.std(isi_list) / mean_isi if mean_isi else 0

    # Power Spectral Density (PSD)
    spike_counts = np.sum(burst_spikes, axis=0)

    dominant_freq = 0
    if len(spike_counts) >= 8:
        nperseg = min(256, len(spike_counts) // 2)
        if nperseg >= 4:
            freqs, psd = signal.welch(spike_counts, fs=1 / time_step, nperseg=nperseg)
            dominant_freq = freqs[np.argmax(psd)] if len(freqs) > 0 else 0

    return {
        "Start": start,
        "End": end,
        "Burst Duration": burst_duration,
        "Total Spikes": total_spikes,
        "Spike Rate": spike_rate,
        "Neuron Participation": neuron_participation,
        "Mean ISI": mean_isi,
        "ISI CV": cv_isi,
        "Dominant Frequency": dominant_freq,
    }


def detect_spiking_periods(
    spike_data: np.ndarray,
    threshold_factor: float = 3,
    smoothing: float = 5,
    min_duration: int = 5,
) -> tuple[list[tuple[int, int]], np.ndarray, float]:
    """
    Detect spiking periods in raster plot data and compute average spiking rate.

    Parameters:
    - spike_data: 2D numpy array (rows: neurons, columns: time points).
    - threshold_factor: Factor for spike detection, based on MAD.
    - smoothing: Gaussian filter smoothing parameter.
    - min_duration: Minimum duration for a detected spike period.

    Returns:
    - spiking_periods: List of tuples (start, end) for detected spike windows.
    - avg_spike_rates: List of average spiking rates for each detected period.
    - smoothed_spikes: Smoothed spike activity for visualization.
    - threshold: Adaptive threshold used for spike detection.
    """

    # Compute total spike activity at each time point
    spike_counts = np.sum(spike_data, axis=0)

    # Smooth spike counts using Gaussian filter
    smoothed_spikes = gaussian_filter1d(spike_counts, sigma=smoothing)

    # Adaptive threshold using median absolute deviation (MAD)
    median_spike = np.median(smoothed_spikes)
    mad = np.median(np.abs(smoothed_spikes - median_spike))
    threshold = median_spike + threshold_factor * mad

    # Detect peak regions using the threshold
    spiking_regions = smoothed_spikes > threshold

    # Extract contiguous spiking periods and compute average spiking rate
    spiking_periods = []
    start = None

    for i in range(len(spiking_regions)):
        if spiking_regions[i] and start is None:
            start = i
        elif not spiking_regions[i] and start is not None:
            if i - start >= min_duration:
                spiking_periods.append((start, i))

            start = None

    return spiking_periods, smoothed_spikes, threshold
