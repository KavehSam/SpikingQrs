import argparse
import tracemalloc

from utils.bxb_comparison import *
from utils.config_loader import ConfigLoader

from spiking_qrs.evaluator import process_single_record
from spiking_qrs.processor import run_dataset_with_args


def parse_arguments():
    """Parse command line arguments for the SpikingQRS detector."""
    parser = argparse.ArgumentParser(
        description="Spiking Neural Network QRS Detection with Homeostasis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="toy_dataset",
        help="Dataset to use for evaluation"
    )
    
    parser.add_argument(
        "--dataset-name", 
        type=str, 
        default="TWADB",
        help="Name for the dataset (used in logging and file naming)"
    )
    
    # Configuration file
    parser.add_argument(
        "--config", 
        type=str, 
        default="snn_models/config.yaml",
        help="Path to YAML configuration file"
    )
    
    # Processing options
    parser.add_argument(
        "--enable-plotting", 
        action="store_true",
        help="Enable real-time plotting of ECG signals and spike data."
    )
    
    parser.add_argument(
        "--return-spikes", 
        action="store_true",
        help="Return spike data."
    )
    
    parser.add_argument(
        "--is-ecg-raw", 
        action="store_true",
        help="Treat input as raw ECG signal."
    )
    
    # Parallel processing
    parser.add_argument(
        "--parallel-jobs", 
        type=int, 
        default=-1,
        help="Number of parallel jobs (-1 for all available processors, default: -1)"
    )
    
    # Single record processing
    parser.add_argument(
        "--record", 
        type=str, 
        default=None,
        help="Process single record instead of full dataset"
    )
    
    # Logging and output
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default=None,
        help="Custom log directory (overrides config)"
    )
    
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default=None,
        help="Custom results directory (overrides config)"
    )
    
    parser.add_argument(
        "--runs-dir", 
        type=str, 
        default=None,
        help="Custom runs directory (overrides config)"
    )
    
    # Data processing parameters
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=None,
        help="Chunk size for processing (overrides config)"
    )
    
    parser.add_argument(
        "--overlap", 
        type=int, 
        default=None,
        help="Overlap between chunks (overrides config)"
    )
    
    parser.add_argument(
        "--exclude-artifacts", 
        action="store_true",
        default=None,
        help="Exclude artifacts from processing (overrides config)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--tolerance", 
        type=int, 
        default=None,
        help="Tolerance for R-peak detection evaluation in samples (overrides config)"
    )
    
    parser.add_argument(
        "--detection-delay", 
        type=int, 
        default=None,
        help="Detection delay compensation in samples (overrides config)"
    )
    
    # Verbosity
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    # Dry run
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show configuration without running experiments"
    )
    
    return parser.parse_args()


def update_config_with_args(config: ConfigLoader, args):
    """Update configuration with command line arguments."""

    config.set("processing.parallel_jobs", args.parallel_jobs)
    
    if args.chunk_size is not None:
        config.set("data.chunk_size", args.chunk_size)
    
    if args.overlap is not None:
        config.set("data.overlap", args.overlap)
    
    if args.exclude_artifacts is not None:
        config.set("data.exclude_artifacts", args.exclude_artifacts)
    
    if args.tolerance is not None:
        config.set("evaluation.tolerance", args.tolerance)
    
    if args.detection_delay is not None:
        config.set("data.detection_delay", args.detection_delay)
    
    if args.enable_plotting:
        config.set("processing.enable_plotting", True)
    
    if args.return_spikes:
        config.set("processing.return_spikes", True)
    
    if args.is_ecg_raw:
        config.set("processing.is_ecg_raw", True)
    
    if args.log_dir is not None:
        config.set("logging.log_dir", args.log_dir)
    
    if args.results_dir is not None:
        config.set("logging.results_dir", args.results_dir)
    
    if args.runs_dir is not None:
        config.set("logging.runs_dir", args.runs_dir)


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # pasre configuration
    try:
        config = ConfigLoader(args.config)
    except FileNotFoundError:
        config = ConfigLoader()
        print("config.yaml not found, using default configuration.")

    # Set configuration from command-line arguments
    if args.chunk_size:
        config.set("data.chunk_size", args.chunk_size)
    if args.overlap:
        config.set("data.overlap", args.overlap)
    if args.detection_delay:
        config.set("data.detection_delay", args.detection_delay)
    
    if args.enable_plotting:
        config.set("processing.enable_plotting", True)
    
    if args.return_spikes:
        config.set("processing.return_spikes", True)
    
    if args.is_ecg_raw:
        config.set("processing.is_ecg_raw", True)
    
    if args.log_dir is not None:
        config.set("logging.log_dir", args.log_dir)
    
    # Check if plotting is enabled for a dataset run and disable it
    if args.enable_plotting and not args.record:
        print("Warning: Plotting is only available when processing a single record.")
        print("Disabling plotting for the dataset run.")
        args.enable_plotting = False
    
    # If plotting for a single record is set, override chunking for better visualization
    if args.enable_plotting and args.record:
        print("Plotting enabled. Overriding chunk_size to 2s and overlap to 0.2s for visualization.")
        config.set("data.chunk_size", 2)
        config.set("data.overlap", 0.2)
    
    print("--------------------Configuration--------------------")
    print(f"Dataset: {args.dataset}")
    
    # update configuration with command line arguments
    update_config_with_args(config, args)
    
    # printout configuration if dry run
    if args.dry_run:
        print("Configuration (after argument overrides):")
        print("-" * 50)
        print(f"Dataset: {args.dataset}")
        print(f"Dataset name: {args.dataset_name}")
        print(f"Chunk size: {config.get('data.chunk_size', 'Not set')}")
        print(f"Overlap: {config.get('data.overlap', 'Not set')}")
        print(f"Parallel jobs: {config.get('processing.parallel_jobs', 'Not set')}")
        print(f"Enable plotting: {config.get('processing.enable_plotting', 'Not set')}")
        print(f"Return spikes: {config.get('processing.return_spikes', 'Not set')}")
        print(f"Is ECG raw: {config.get('processing.is_ecg_raw', 'Not set')}")
        print(f"Tolerance: {config.get('evaluation.tolerance', 'Not set')}")
        print(f"Detection delay: {config.get('data.detection_delay', 'Not set')}")
        exit(0)
    
    # single record mode, if specified
    if args.record:
        result = process_single_record(
            record_name=args.record,
            config=config,
            dataset=args.dataset,
            log_dir=args.log_dir,
            plotting=args.enable_plotting,
            return_spikes=args.return_spikes,
            is_ecg_raw=args.is_ecg_raw
        )
        if result is None:
            exit(1)
        exit(0)
    
    # Run SpikingQrs beat detector on all records of all selected datasets
    run_dataset_with_args(
        config=config,
        dataset=args.dataset,
        dataset_name=args.dataset_name,
        verbose=args.verbose
    )
