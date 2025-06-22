"""
Preprocessing of ECG signals for SpikingQRS detector.
"""

import neurokit2 as nk
import numpy as np


def preprocess_ecg_signal(ecg_signal, sampling_rate, do_filter: bool = True):
    
    if do_filter:
        order = 2
        filtered_signal = nk.signal.signal_filter(
            signal=ecg_signal,
            sampling_rate=sampling_rate,
            lowcut=8,
            highcut=28,
            method="butterworth_zi",
            order=order,
        )
    else:
        filtered_signal = ecg_signal

    # Compute the Squared Signal
    squared = filtered_signal ** 2
    
    # Smooth the squared signal
    window_size = int(0.12 * sampling_rate)  # 120 ms window
    smoothed = np.convolve(squared, np.ones(window_size) / window_size, mode="same")
    
   
    return filtered_signal, squared, smoothed

