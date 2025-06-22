"""
This module contains set of functions for evaluating the performance of the SpikingQRS detector
on ECG signals and datasets.
"""

import neurokit2 as nk
import numpy as np
import os
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List, TYPE_CHECKING

from snn_models.adaptive_homesatic_two_layers import AdaptiveHomeostasisTwoLayers
from utils.bxb_comparison import BxbEvaluator
from spiking_qrs.online_spiking_qrs_detector import OnlineSpikingQrsDetector
from utils.config_loader import ConfigLoader


def evaluate_homeostasis(
    ecg_signal: np.ndarray,
    sampling_rate: int,
    log_dir: Optional[str],
    config: ConfigLoader,
    true_peaks: Optional[np.ndarray] = None,
    plotting: bool = False,
    return_spikes: bool = False,
    is_ecg_raw: bool = True,
    record_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[List], Optional[List], Dict[str, Any]]:
    """
    Evaluate the SpikingQRS detector on a single ECG signal.
    
    Args:
        ecg_signal: ECG signal array
        sampling_rate: Sampling rate in Hz
        log_dir: Directory for logging (optional)
        config: Configuration loader
        true_peaks: True R-peak locations (optional, will be computed if None)
        plotting: Enable plotting
        return_spikes: Return spike trains
        is_ecg_raw: Treat input as raw ECG signal
        record_name: Name of the record for logging
        
    Returns:
        Tuple of (true_peaks, spiking_peaks, spike_trains_1, spike_trains_2, spiking_results)
    """
    true_peaks_array: np.ndarray
    if true_peaks is None:
        _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
        r_peaks_list = rpeaks.get("ECG_R_Peaks", [])
        true_peaks_array = np.array(r_peaks_list)
    else:
        true_peaks_array = true_peaks

    # Create an adaptive homeostasis spiking network with config
    net_snn = AdaptiveHomeostasisTwoLayers(config=config)

    # Get data processing parameters
    data_config = config.get_data_config()
    processing_config = config.get_processing_config()
    network_config = config.get_network_config()

    # Create visualizer if plotting is enabled
    visualizer = None
    if plotting or processing_config["enable_plotting"]:
        from plotting import spiking_visualizer
        if not spiking_visualizer.PYTQTGRAPH_AVAILABLE:
            print("PyQtGraph not available or installed. Real-time visualization is disabled.")
            plotting = False

        if plotting:
            from plotting.spiking_visualizer import SpikingVisualizer
            visualizer = SpikingVisualizer(
                sampling_rate=sampling_rate,
                num_neurons=network_config['input_size']
            )

    spiking_detector = OnlineSpikingQrsDetector(
        sampling_rate,
        net_snn,
        chunk_size=data_config["chunk_size"],
        overlap=data_config["overlap"],
        plotting=plotting,
        visualizer=visualizer,
        return_spikes=return_spikes or processing_config["return_spikes"],
        is_ecg_raw=is_ecg_raw or processing_config["is_ecg_raw"],
        record_name=record_name,
    )

    if log_dir is not None:
        spiking_detector.set_log_dir(log_dir)

    spike_trains_1 = spike_trains_2 = None
    spiking_peaks, spike_trains_1, spike_trains_2 = (
        spiking_detector.detect_beat_single_lead(ecg_signal)
    )

    # Evaluate Beat Detection Performance
    evaluation_config = config.get_evaluation_config()
    evaluator = BxbEvaluator(tolerance=evaluation_config["tolerance"])

    data_config = config.get_data_config()
    spiking_results = evaluator.evaluate(
        true_peaks_array,
        np.unique(spiking_peaks) + data_config["detection_delay"],
        sampling_rate=sampling_rate,
    )

    for metric, value in spiking_results.items():
        if log_dir is not None and spiking_detector.writer is not None:
            spiking_detector.writer.add_scalar(f"spiking/{metric}", value, 0)

    return true_peaks_array, np.array(spiking_peaks), spike_trains_1, spike_trains_2, spiking_results


def process_record(
    record_name: str, 
    config: ConfigLoader, 
    ds_folder: str = "mitbih_arrhythmia_database"
) -> Optional[Tuple[str, Tuple]]:
    """
    Process a single record from a dataset.
    
    Args:
        record_name: Name of the record to process
        config: Configuration loader
        ds_folder: Dataset folder path
        
    Returns:
        Tuple of (record_name, results) or None if processing failed
    """
    from utils.mit_data_handler import load_mitbih_data
    
    if os.path.exists(os.path.join(ds_folder, f"{record_name}.hea")):
        data_config = config.get_data_config()

        exclude_artifacts = data_config.get("exclude_artifacts", True)
        if "sinus" in ds_folder:
            exclude_artifacts = False

        ecg_signal, r_true, sampling_rate = load_mitbih_data(
            record_name=record_name,
            path=ds_folder,
            exclude_artitacts=exclude_artifacts,
        )

        if ecg_signal is None or r_true is None or sampling_rate is None:
            return None

        return (
            record_name,
            evaluate_homeostasis(
                ecg_signal[:, 0],
                sampling_rate,
                log_dir=None,
                config=config,
                true_peaks=r_true,
                plotting=False,
                return_spikes=False,
                is_ecg_raw=True,
                record_name=record_name,
            ),
        )
    
    return None


def process_single_record(
    record_name: str, 
    config: ConfigLoader, 
    dataset: str,
    log_dir: Optional[str] = None,
    plotting: bool = False,
    return_spikes: bool = False,
    is_ecg_raw: bool = True
) -> Optional[Tuple[str, Tuple]]:
    """
    Process a single record with custom parameters.
    
    Args:
        record_name: Name of the record to process
        config: Configuration loader
        dataset: Dataset name
        log_dir: Custom log directory
        plotting: Enable plotting
        return_spikes: Return spike trains
        is_ecg_raw: Treat input as raw ECG signal
        
    Returns:
        Tuple of (record_name, results) or None if processing failed
    """
    from utils.mit_data_handler import load_mitbih_data
    
    ds_folder = f"data/{dataset}"
    
    if not os.path.exists(os.path.join(ds_folder, f"{record_name}.hea")):
        print(f"Record {record_name} not found in {ds_folder}")
        return None
    
    data_config = config.get_data_config()
    exclude_artifacts = data_config.get("exclude_artifacts", True)
    if "sinus" in dataset:
        exclude_artifacts = False

    ecg_signal, r_true, sampling_rate = load_mitbih_data(
        record_name=record_name,
        path=ds_folder,
        exclude_artitacts=exclude_artifacts,
    )

    if ecg_signal is None or r_true is None or sampling_rate is None:
        print(f"Failed to load record {record_name}")
        return None

    # Create custom log directory for single record
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if log_dir:
        log_dir = f"{log_dir}/{record_name}_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)

    print(f"Processing record: {record_name}")
    if ecg_signal is not None:
        print(f"Signal length: {len(ecg_signal)} samples")
    print(f"Sampling rate: {sampling_rate}")
    if r_true is not None:
        print(f"True R-peaks: {len(r_true)}")

    result = evaluate_homeostasis(
        ecg_signal[:, 0],
        sampling_rate,
        log_dir=log_dir,
        config=config,
        true_peaks=r_true,
        plotting=plotting,
        return_spikes=return_spikes,
        is_ecg_raw=is_ecg_raw,
        record_name=record_name,
    )

    if result is None:
        return None

    _, _, _, _, spiking_results = result

    print(f"\nResults for record: {record_name}")
    print("-" * 50)
    for metric, value in spiking_results.items():
        if "Rpeaks" not in metric and "Times" not in metric and "method" not in metric:
            print(f"  {metric}: {value:.4f}")

    return record_name, result 