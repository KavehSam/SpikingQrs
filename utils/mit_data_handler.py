import os
import numpy as np
import wfdb


def list_records(ds_folder="mitbih_arrhythmia_database"):
    records = [f.split(".")[0] for f in os.listdir(ds_folder) if f.endswith(".hea") and any(
        char.isdigit() for char in f) and (os.path.exists(os.path.join(ds_folder, f.replace(".hea", ".atr"))) or os.path.exists(os.path.join(ds_folder, f.replace(".hea", ".qrs"))))]
    return records


def exclude_non_qrs_annotation(annotations, exclude_artitacts=True):
    """
    filter reference annotations.

    Args:
        annotation (array): array of annotations.

    Returns:
        np.array: Array of reference R-peak positions (in samples).
    """

    # Filter annotations to include only QRS annotations
    qrs_types = [
        'N',  # Normal beat
        'L',  # Left bundle branch block beat
        'R',  # Right bundle branch block beat
        'e',  # Atrial escape beat
        'j',  # Nodal (junctional) escape beat
        'J',  # Nodal (junctional) premature beat
        'A',  # Atrial premature beat
        'a',  # Aberrated atrial premature beat
        'S',  # Supraventricular premature beat
        'E',  # Ventricular escape beat
        'Q',  # Unclassifiable beat
        '/',  # Paced beat
        '.',  # Ventricular flutter wave
        'V',  # Premature ventricular contraction
        'F',  # Fusion of ventricular and normal beat
        'f',  # Fusion of paced and normal beat
        'P',  # Paced beat
        'T',  # Ventricular tachycardia beat
        '!',  # Ventricular flutter wave
        
    ]
    
    if not exclude_artitacts:
        qrs_types += [ '|']  
        
    r_peaks = np.array([sample for sample, symbol in zip(annotations.sample, annotations.symbol) if symbol in qrs_types])

    return r_peaks


def load_mitbih_data(record_name="100", path="mitbih_arrhythmia_database", exclude_artitacts=True):
    """
    Loads the ECG signal and annotations from the MIT-BIH Arrhythmia Database.

    Args:
        record_name (str, optional): The name of the record to load. Defaults to "100".
        path (str, optional): Dataset folder name. Defaults to "mitbih_arrhythmia_database".
        exclude_artitacts (bool, optional): Exclude non-QRS annotations. Defaults to True.

    Returns:
        ecg_signal (np.array): The ECG signal.
        r_peaks (np.array): R-peaks time stamps.
        fs (int): The sampling frequency.
    """
    try:
        record = wfdb.rdrecord(os.path.join(path, record_name))
    except (ValueError, FileNotFoundError):
        return None, None, None
    
    if not isinstance(record, wfdb.Record):
        print(f"Warning: Record {record_name} is not a single-segment record. Skipping.")
        return None, None, None

    if os.path.exists(os.path.join(path, f"{record_name}.atr")):
        annotations = wfdb.rdann(os.path.join(path, record_name), "atr")
    elif os.path.exists(os.path.join(path, f"{record_name}.qrs")):
        annotations = wfdb.rdann(os.path.join(path, record_name), "qrs")        
    
    ecg_signal = record.p_signal  
    r_peaks =  exclude_non_qrs_annotation(annotations, exclude_artitacts=exclude_artitacts)
    return ecg_signal, r_peaks, record.fs 


def download_mitbih(db_code="mitdb", folder_name=None):
    """
    Downloads a PhysioNet database into the 'data' directory.

    Parameters:
    - db_code (str): The PhysioNet database code (e.g., 'mitdb').
    - folder_name (str, optional): The name of the subfolder inside 'data'. 
      If None, it defaults to '{db_code}_database'.

    Returns:
    - None
    """
    if folder_name is None:
        folder_name = f"{db_code}_database"

    base_dir = "data"
    target_folder = os.path.join(base_dir, folder_name)

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created target folder: {target_folder}")

    print(f"Downloading {db_code} database from PhysioNet...")
    wfdb.dl_database(
        db_code,
        target_folder,
        overwrite=True 
    )
    print(f"Download complete! Data saved in: {target_folder}")