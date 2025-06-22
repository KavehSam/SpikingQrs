"""
Dataset processing module for SpikingQRS detector.

This module contains functions for processing entire datasets and managing
experiment runs.
"""

import os
import pickle
from datetime import datetime
from typing import List, Tuple, Dict, Any
from joblib import Parallel, delayed

from utils.mit_data_handler import list_records
from utils.bxb_comparison import parse_bxb_results, calculate_overall_statistics, append_overall_statistics_to_file
from utils.config_loader import ConfigLoader
from spiking_qrs.evaluator import process_record


def run_dataset(
    config: ConfigLoader, 
    ds_folder: str = "mitbih_arrhythmia_database", 
    ds_name: str = "MITBIH_ARRHY"
) -> None:
    """
    Run experiments on an entire dataset.
    
    Args:
        config: Configuration loader
        ds_folder: Dataset folder name
        ds_name: Dataset name for logging
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    data_config = config.get_data_config()
    logging_config = config.get("logging", {}) or {}
    ds_folder = f"data/{ds_folder}"

    log_dir_name = f"{ds_name}_{timestamp}_cs{data_config['chunk_size']}_ol{data_config['overlap']}"
    log_dir = f"{logging_config.get('runs_dir', 'runs')}/{log_dir_name}"
    os.makedirs(log_dir, exist_ok=True)

    record_names = list_records(ds_folder)

    processing_config = config.get_processing_config()
    results = Parallel(n_jobs=processing_config["parallel_jobs"])(
        delayed(process_record)(_record_name, config, ds_folder)
        for _record_name in record_names
    )

    # Filter out None results
    results = [result for result in results if result is not None]

    # Save results to a pickle file
    results_dir = logging_config.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    with open(
        os.path.join(
            results_dir,
            f"results_{timestamp}_cs{data_config['chunk_size']}_ol{data_config['overlap']}.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(results, f)

    # iterate over results and print out per record metrics
    for record_name, record_results in results:
        print(f"\n Performance Comparison record: {record_name}")

        # parse the SpikingQrs method results 
        method_results = record_results[-1]
        method = "SpikingQrs"

        print(f"\n{method} Results:")
        for metric, value in method_results.items():
            if (
                "Rpeaks" not in metric
                and "Times" not in metric
                and "method" not in metric
            ):
                print(f"  {metric}: {value:.4f}")

        # Write results to a text file
        with open(f"bxb_{log_dir_name}.txt", "a+") as f:
            f.write(f"\n Performance Comparison record: {record_name}\n")
            method_results["method"] = [method]
            for metric, value in method_results.items():
                if isinstance(value, list):
                    f.write(f"{metric}: {', '.join(map(str, value))}\n")
                else:
                    f.write(f"{metric}: {value:.4f}\n")

    parsed_data = parse_bxb_results(f"bxb_{log_dir_name}.txt")
    overall_stats = calculate_overall_statistics(parsed_data)

    # Print overall and average statistics for each method
    for method, stats in overall_stats.items():
        print("\n\n")
        print("-" * 100)
        print(f"Overall Performance metrics for Method: {method}")
        print("-" * 100)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()

    # Append overall statistics to the file
    append_overall_statistics_to_file(f"bxb_{log_dir_name}.txt", overall_stats)
    print(f"Overall statistics appended to bxb_{log_dir_name}.txt")


def run_dataset_with_args(
    config: ConfigLoader, 
    dataset: str,
    dataset_name: str,
    verbose: bool = False
) -> None:
    """
    Run dataset experiments with argument-based configuration.
    
    Args:
        config: Configuration loader
        dataset: Dataset folder name
        dataset_name: Dataset name for logging
        verbose: Enable verbose output
    """
    print(f"Running experiments with dataset: {dataset}")
    print(f"Dataset name: {dataset_name}")
    
    if verbose:
        print(f"Chunk size: {config.get('data.chunk_size', 'Not set')}")
        print(f"Overlap: {config.get('data.overlap', 'Not set')}")
        print(f"Parallel jobs: {config.get('processing.parallel_jobs', 'Not set')}")
    
    run_dataset(config, ds_folder=dataset, ds_name=dataset_name) 