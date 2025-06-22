import re
import numpy as np


class BxbEvaluator:
    def __init__(self, tolerance=0.15):
        """
        Initialize the evaluator.

        Args:
            tolerance (float): Maximum allowable difference (in seconds) between reference
                               and detected annotations for a match.
        """
        self.tolerance = tolerance

    def evaluate(self, reference_annotations, detected_annotations, sampling_rate=250):
        """
        Compare detected annotations with reference annotations.

        Args:
            reference_annotations (list or np.array): List of reference beat positions (in samples).
            detected_annotations (list or np.array): List of detected beat positions (in samples).
            sampling_rate (int): Sampling rate of the signal (default: 250 Hz).

        Returns:
            dict: Dictionary containing performance metrics and times of FPs and FNs.
        """

        # Convert tolerance to samples
        tolerance_samples = int(self.tolerance * sampling_rate)

        # Sort annotations
        reference_annotations = np.sort(reference_annotations)
        detected_annotations = np.sort(detected_annotations)

        # Initialize counters and storage
        true_positives = 0
        false_positives = []
        false_negatives = []

        # Track used reference annotations
        matched_reference = np.zeros(len(reference_annotations), dtype=bool)

        for detected in detected_annotations:
            # Find the closest reference annotation
            differences = np.abs(reference_annotations - detected)
            closest_index = np.argmin(differences)

            # Check if the closest reference annotation is within tolerance
            if (
                differences[closest_index] <= tolerance_samples
                and not matched_reference[closest_index]
            ):
                true_positives += 1
                matched_reference[closest_index] = True
            else:
                false_positives.append(detected)  # Add FP (in samples)

        # Remaining unmatched reference annotations are false negatives
        for i, matched in enumerate(matched_reference):
            if not matched:
                false_negatives.append(reference_annotations[i])  # Add FN (in samples)

        # Convert FPs and FNs to times (in seconds)
        false_positives_times = [fp / sampling_rate for fp in false_positives]
        false_negatives_times = [fn / sampling_rate for fn in false_negatives]

        # Compute performance metrics
        sensitivity = (
            true_positives / (true_positives + len(false_negatives))
            if (true_positives + len(false_negatives)) > 0
            else 0
        )
        precision = (
            true_positives / (true_positives + len(false_positives))
            if (true_positives + len(false_positives)) > 0
            else 0
        )
        f1_score = (
            2 * (sensitivity * precision) / (sensitivity + precision)
            if (sensitivity + precision) > 0
            else 0
        )

        # Round sensitivity and precision to four decimal places
        sensitivity = round(sensitivity, 4)
        precision = round(precision, 4)
        f1_score = round(f1_score, 4)

        return {
            "True Positives": true_positives,
            "False Positives": len(false_positives),
            "False Negatives": len(false_negatives),
            "False Positives Rpeaks": false_positives,
            "False Negatives Rpeaks": false_negatives,
            "False Positives Times (s)": false_positives_times,
            "False Negatives Times (s)": false_negatives_times,
            "Sensitivity": sensitivity,
            "Precision": precision,
            "F1 Score": f1_score,
        }


def parse_bxb_results(file_path):
    """
    Parse a text file containing BxB results into a structured list of dictionaries.

    Args:
        file_path (str): Path to the BxB results text file.

    Returns:
        list: List of dictionaries, each representing results for one record and method.
    """
    records = []
    with open(file_path, "r") as file:
        content = file.read()

    # Split by performance comparison blocks
    blocks = content.strip().split("\n\n")

    for block in blocks:
        record_data = {}

        # Extract record number
        match = re.search(r"Performance Comparison record: (\d+)", block)
        if match:
            record_data["record"] = int(match.group(1))

        # Extract true positives, false positives, and false negatives
        for key in ["True Positives", "False Positives", "False Negatives"]:
            match = re.search(rf"{key}: ([\d\.]+)", block)
            if match:
                record_data[key] = float(match.group(1))

        # Extract lists of false positives and false negatives R-peaks
        for key in ["False Positives Rpeaks", "False Negatives Rpeaks"]:
            match = re.search(rf"{key}: ([\d, ]+)", block)
            if match:
                record_data[key] = list(map(int, match.group(1).split(", ")))

        # Extract times of false positives and false negatives
        for key in ["False Positives Times \(s\)", "False Negatives Times \(s\)"]:
            match = re.search(rf"{key}: ([\d\., ]+)", block)
            if match:
                record_data[key.replace(" \(s\)", "")] = list(
                    map(float, match.group(1).split(", "))
                )

        # Extract sensitivity, precision, and F1 score
        for key in ["Sensitivity", "Precision", "F1 Score"]:
            match = re.search(rf"{key}: ([\d\.]+)", block)
            if match:
                record_data[key] = float(match.group(1))

        # Extract method
        match = re.search(r"method: (\w+)", block)
        if match:
            record_data["method"] = match.group(1)

        # Add the parsed record data
        records.append(record_data)

    return records


def calculate_overall_statistics(parsed_data):
    """
    Calculate overall gross and average statistics for methods across all records.

    Args:
        parsed_data (list): List of dictionaries, each containing BxB results for one record.

    Returns:
        dict: Dictionary containing overall and average statistics for each method.
    """
    # Initialize dictionaries to accumulate data for each method
    overall_stats = {}

    for record in parsed_data:
        method = record["method"]
        if method not in overall_stats:
            # Initialize gross and average statistics for this method
            overall_stats[method] = {
                "True Positives": 0,
                "False Positives": 0,
                "False Negatives": 0,
                "Sensitivity": [],
                "Precision": [],
                "F1 Score": [],
            }

        # Accumulate gross statistics
        overall_stats[method]["True Positives"] += record["True Positives"]
        overall_stats[method]["False Positives"] += record["False Positives"]
        overall_stats[method]["False Negatives"] += record["False Negatives"]

        # Append metrics for averaging
        overall_stats[method]["Sensitivity"].append(record["Sensitivity"])
        overall_stats[method]["Precision"].append(record["Precision"])
        overall_stats[method]["F1 Score"].append(record["F1 Score"])

    # Calculate averages for each method
    for method, stats in overall_stats.items():
        stats["Average Sensitivity"] = sum(stats["Sensitivity"]) / len(
            stats["Sensitivity"]
        )
        stats["Average Precision"] = sum(stats["Precision"]) / len(stats["Precision"])
        stats["Average F1 Score"] = sum(stats["F1 Score"]) / len(stats["F1 Score"])
        stats["Total Detection Error"] = (
            stats["False Positives"] + stats["False Negatives"]
        ) / (stats["True Positives"] + stats["False Negatives"])

        # Remove lists after averaging
        del stats["Sensitivity"]
        del stats["Precision"]
        del stats["F1 Score"]

    return overall_stats


def append_overall_statistics_to_file(file_path, overall_stats):
    """
    Append overall statistics to the end of the BxB results text file.

    Args:
        file_path (str): Path to the BxB results text file.
        overall_stats (dict): Overall statistics to append.
    """
    with open(file_path, "a") as file:
        file.write("\n\nOverall Performance Statistics\n")
        for method, stats in overall_stats.items():
            file.write(f"Method: {method}\n")
            for key, value in stats.items():
                file.write(f"  {key}: {value:.4f}\n")
            file.write("\n")
