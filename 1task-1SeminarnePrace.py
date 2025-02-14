import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# === FUNKCE ===
def load_ecg(file_path):
    """Načte data EKG"""
    record = wfdb.rdrecord(file_path)
    ecg_signal = record.p_signal[:, 0]  # První kanál
    fs = record.fs  # Frekvence vzorkování
    return ecg_signal, fs

def bandpass_filter(signal, fs, lowcut=0.5, highcut=30, order=4):
    """Pásmový filtr (nastavitelný pro MIT-BIH)"""
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_r_peaks(ecg_signal, fs, height_factor=0.8, distance_factor=None):
    """Detekce R-špičky"""
    threshold = np.percentile(ecg_signal, 98)  # Dynamický práh
    if distance_factor is None:
        distance_factor = fs // 2 

    r_peaks, _ = find_peaks(ecg_signal, height=threshold * height_factor, distance=distance_factor)
    return r_peaks

def compute_bpm(r_peaks, fs):
    """Výpočet tepové frekvence (BPM)"""
    rr_intervals = np.diff(r_peaks) / fs  # Intervaly v sekundách
    bpm_values = 60 / rr_intervals
    return bpm_values

def process_ecg(file_path):
    """Základní zpracování EKG"""
    # === 1. Načítání dat ===
    ecg_signal, fs = load_ecg(file_path)
    
    # === 2. Korekce amplitudy pro MIT-BIH ===
    if fs == 128:
        ecg_signal *= 1000  # Převádíme na mikrovolty

    time = np.linspace(0, len(ecg_signal) / fs, len(ecg_signal))

    # === 3. Filtrace ===
    if fs == 128:
        filtered_ecg = bandpass_filter(ecg_signal, fs, lowcut=0.5, highcut=20, order=2)
    else:
        filtered_ecg = bandpass_filter(ecg_signal, fs)

    # === 4. Vyhledávání R-špičky ===
    r_peaks = detect_r_peaks(filtered_ecg, fs)



    # === 5. Výpočet BPM ===
    bpm_values = compute_bpm(r_peaks, fs)
    avg_bpm = np.mean(bpm_values) if len(bpm_values) > 0 else 0

    # === VIZUALIZACE ===
    plt.figure(figsize=(12, 4))
    plt.plot(time, filtered_ecg, color='blue')
    plt.scatter(time[r_peaks], filtered_ecg[r_peaks], color='red', label="R-peaks")
    plt.title(f"Tepová frekvence {avg_bpm:.2f} bpm")
    plt.xlabel("Čas (s)")
    plt.ylabel("Voltage (uV)")
    plt.legend()
    plt.grid()
    plt.show()


    # === 7. Výběr 30sekundového úseku ===
    start_time = 30 
    end_time = 40
    start_index = int(fs * start_time)
    end_index = int(fs * end_time)

    # Přerušení signálu
    segment_time = time[start_index:end_index]
    segment_signal = filtered_ecg[start_index:end_index]

    # V případě MIT-BIH změňte parametry vyhledávání R-peaku
    if fs == 128:
        r_peaks_segment = detect_r_peaks(segment_signal, fs, height_factor=0.5, distance_factor=fs // 4)
    else:
        r_peaks_segment = detect_r_peaks(segment_signal, fs)
    
    r_peaks_segment = r_peaks_segment[r_peaks_segment < len(segment_time)]   # Indexování

    # === 9. Výpočet BPM pro 30sekundový úsek ===
    bpm_values_segment = compute_bpm(r_peaks_segment, fs)
    avg_bpm_segment = np.mean(bpm_values_segment) if len(bpm_values_segment) > 0 else 0

    # === 10. Vizualizace 30sekundového úseku ===
    plt.figure(figsize=(12, 4))
    plt.plot(segment_time, segment_signal, color='blue')
    plt.scatter(segment_time[r_peaks_segment], segment_signal[r_peaks_segment], color='red', label="R-peaks")
    plt.title(f"Tepová frekvence {avg_bpm_segment:.2f} bpm (30 sekund)")
    plt.xlabel("Čas (s)")
    plt.ylabel("Voltage (uV)")
    plt.legend()
    plt.grid()
    plt.show()

# === SEZNAM SOUBORŮ ===
file_paths = [
    "C:/LocalFiles_my/PZS/My_seminarne_prace/brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0/100001/100001_ECG",
    "C:/LocalFiles_my/PZS/My_seminarne_prace/mit_bih/16265",
    "C:/LocalFiles_my/PZS/My_seminarne_prace/brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0/100002/100002_ECG"
]

# === ZPRACOVÁNÍ SOUBORŮ ===
for file_path in file_paths:
    print(f"Processing file: {file_path}")
    process_ecg(file_path)
