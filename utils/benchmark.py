import numpy as np
import neurokit2 as nk
import os
from datetime import datetime
from joblib import Parallel, delayed

from utils.bxb_comparison import BxbEvaluator


def evaluate_methods(
    ecg_signal: np.ndarray, sampling_rate: int, true_peaks: np.ndarray | None = None
) -> tuple[np.ndarray, dict[str, dict]]:
    # Ground Truth: Simulated R-peaks
    if true_peaks is None:
        _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
        true_peaks = rpeaks["ECG_R_Peaks"]

    results = {}
    for method in ["hamilton2002", "elgendi2010", "gamboa2008", "kalidas2017"]:
        _smoothed = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method=method)
        _, info = nk.ecg_peaks(_smoothed, sampling_rate=sampling_rate, method=method)
        baseline_peaks = info["ECG_R_Peaks"]

        # Evaluate Performance
        evaluator = BxbEvaluator(tolerance=0.15)
        results[method] = evaluator.evaluate(
            true_peaks, np.unique(baseline_peaks), sampling_rate=sampling_rate
        )

    return true_peaks, results


def benchmark_record(
    record_name: str, ds_folder: str = "mitbih_arrhythmia_database"
) -> tuple[str, tuple[np.ndarray, dict[str, dict]]] | None:
    if os.path.exists(os.path.join(ds_folder, f"{record_name}.hea")):
        # print(f"Processing record: {record_name}")

        exclude_artitacts = True
        if "sinus" in ds_folder:
            exclude_artitacts = False

        ecg_signal, r_true, sampling_rate = load_mitbih_data(
            record_name=f"{record_name}",
            path=ds_folder,
            exclude_artitacts=exclude_artitacts,
        )

        # ecg_signal = nk.ecg_clean(ecg_signal[:, 0], sampling_rate=sampling_rate, method="neurokit")  # Other methods: 'biosppy', 'pamtompkins' 'elgendi2010
        if ecg_signal is None:
            return None

        return (
            record_name,
            evaluate_methods(ecg_signal[:, 0], sampling_rate, true_peaks=r_true),
        )


def benchmark_dataset(ds_folder="mitbih_arrhythmia_database", ds_name="MITBIH_ARRHY"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_name = f"{ds_name}_{timestamp}"
    log_dir = f"runs/{log_dir_name}"
    os.makedirs(log_dir, exist_ok=True)

    record_names = list_records(ds_folder)
    results = Parallel(n_jobs=-1)(
        delayed(benchmark_record)(_record_name, ds_folder)
        for _record_name in record_names
    )

    # Filter out None results
    results = [result for result in results if result is not None]

    # Unpack results if needed
    for record_name, record_results in results:
        # # Print Results
        print(f"\n Performance Comparison record: {record_name}")
        for method in record_results[1].keys():
            method_results = record_results[1][method]
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
                    if isinstance(value, list):  # Check if the value is a list
                        f.write(f"{metric}: {', '.join(map(str, value))}\n")
                    else:
                        f.write(f"{metric}: {value:.4f}\n")

    parsed_data = parse_bxb_results(f"bxb_{log_dir_name}.txt")
    overall_stats = calculate_overall_statistics(parsed_data)

    # Print overall and average statistics for each method
    for method, stats in overall_stats.items():
        print(f"Method: {method}")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()

    # Append overall statistics to the file
    append_overall_statistics_to_file(f"bxb_{log_dir_name}.txt", overall_stats)

    print("Overall statistics appended to the file.")


if __name__ == "__main__":
    datasets = {
        "MITBIH_ARRHY": "mitbih_arrhythmia_database",
        "NSTDB": "mit-bih-noise-stress-test-database-1.0.0",
        "QTDB": "qt-database-1.0.0",  # slow
        "TWADB": "t-wave-alternans-challenge-database-1.0.0",
        "SVDB": "mit-bih-supraventricular-arrhythmia-database-1.0.0",
        "LAFDB": "mit-bih-atrial-fibrillation-database-1.0.0",  # big
        "NSRDB": "mit-bih-normal-sinus-rhythm-database-1.0.0",  # big
    }

    ds_names = [
        "MITBIH_ARRHY",
        "TWADB",
        "NSTDB",
        "QTDB",
        "SVDB",
        "NSRDB",
    ]  # "LAFDB",  "LAFDB",

    for ds_name in ds_names:
        print(f"Running dataset: {ds_name}")
        ds_folder = f"data/{datasets[ds_name]}"

        benchmark_dataset(ds_folder, ds_name)
