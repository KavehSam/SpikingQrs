"""
SpikingQRS package for ECG QRS detection using spiking neural networks.

This package provides tools for:
- Online QRS detection using spiking neural networks
- ECG signal preprocessing
- Feature extraction from spike trains
- Evaluation and benchmarking
- Dataset processing
"""

from .online_spiking_qrs_detector import OnlineSpikingQrsDetector
from .preprocessor import preprocess_ecg_signal
from .raster_fex import *
from .evaluator import evaluate_homeostasis, process_record, process_single_record
from .processor import run_dataset, run_dataset_with_args

__all__ = [
    'OnlineSpikingQrsDetector',
    'preprocess_ecg_signal',
    'evaluate_homeostasis',
    'process_record', 
    'process_single_record',
    'run_dataset',
    'run_dataset_with_args',
] 