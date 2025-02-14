import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd

## downloadong and vizualization of data
record = wfdb.rdrecord("C:/LocalFiles_my/PZS/My_seminarne_prace/brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0/100001/100001_ECG")

ecg_signal = record.p_signal[:, 0]

plt.figure(figsize=(10, 4))
plt.plot(ecg_signal, label="ECG Signal")
plt.title("ECG Signal")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

## filter of the signal (and deleting)
def bandpass_filter(signal, lowcut=0.5, highcut=50.0, fs=360, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

filtered_ecg = bandpass_filter(ecg_signal)

plt.figure(figsize=(10, 4))
plt.plot(ecg_signal, label="Raw ECG", alpha=0.6)
plt.plot(filtered_ecg, label="Filtered ECG", color="red")
plt.title("ECG Signal Before and After Filtering")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

## Detekce anomálií 
# Metoda náhlých změn amplitudy
diff_signal = np.diff(filtered_ecg)
threshold = 3 * np.std(diff_signal)
anomaly_indices = np.where(np.abs(diff_signal) > threshold)[0]

plt.figure(figsize=(10, 4))
plt.plot(filtered_ecg, label="Filtered ECG")
plt.scatter(anomaly_indices, filtered_ecg[anomaly_indices], color="red", label="Anomalies")
plt.title("ECG Anomalies Detection (Amplitude Changes)")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()


## Metoda prudkých odchylek frekvence signálu (HRV)
peaks, _ = find_peaks(filtered_ecg, height=np.mean(filtered_ecg) + 0.5*np.std(filtered_ecg), distance=50)
rr_intervals = np.diff(peaks) / 360 
rr_diff = np.diff(rr_intervals)

# Práh anomálie: prudké skoky > 3σ
rr_threshold = 2 * np.std(rr_diff)
anomaly_rr = np.where(np.abs(rr_diff) > rr_threshold)[0]

### 4. Optimalizovaný plán intervalů RR ###
num_points = len(rr_intervals)
max_points = 10000

if num_points > max_points:
    step = num_points // max_points
    rr_intervals_downsampled = rr_intervals[::step]

    # Oprava indexů anomálií a ponechání pouze těch, které spadají do pole se sníženým vzorkováním.
    anomaly_rr_downsampled = [i // step for i in anomaly_rr if i // step < len(rr_intervals_downsampled)]
else:
    rr_intervals_downsampled = rr_intervals
    anomaly_rr_downsampled = anomaly_rr

plt.figure(figsize=(10, 5)) #was 4 
plt.plot(rr_intervals_downsampled, label="RR Intervals", linewidth=0.8, zorder=1)
if len(anomaly_rr_downsampled) > 0:
    plt.scatter(anomaly_rr_downsampled, 
                np.array(rr_intervals_downsampled)[anomaly_rr_downsampled], 
                color="red", label="Anomalies", s=10, zorder=2)
plt.title("RR Interval Anomalies")
plt.xlabel("Beat Number")
plt.ylabel("Interval (s)")
plt.legend()
plt.grid(True)
plt.show()

## Statistiky anomálií
total_points = len(filtered_ecg)
num_anomalies = len(anomaly_rr)
percentage_anomalies = (num_anomalies / total_points) * 100

# Vytvoření DataFrame s výsledky
df_stats = pd.DataFrame({
    "Total Points": [total_points],
    "Anomalous Points": [num_anomalies],
    "Percentage Anomalies": [percentage_anomalies]
})

print(df_stats)

########################################## extensions

#### that's for local extrem / Detekce vln R, P a T

### Dodatečná filtrace signálu před detekcí ###
# Průměrování signálu pomocí klouzavého průměru (odstranění ostrých šumů)
def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

filtered_ecg_smoothed = moving_average(filtered_ecg, window_size=5)

# vyhledávání R-špičky (již dříve implementováno)
r_threshold = np.mean(filtered_ecg) + 0.7 * np.std(filtered_ecg)  # Minimální výška
peaks_R, _ = find_peaks(filtered_ecg, height=r_threshold, distance=50)  # Detekce R-špičky


# Hledání vln P 
peaks_P = []
for r in peaks_R:
    search_window = filtered_ecg[max(0, r-100):r-10]
    if len(search_window) > 0:
        p_peak = np.argmax(search_window) + (r-100)  # Local max
        if filtered_ecg_smoothed[p_peak] > np.mean(search_window) + 0.3 * np.std(search_window):  # Zesílený filtr
            peaks_P.append(p_peak)

# Vyhledejte vlny T
peaks_T = []
for r in peaks_R:
    search_window = filtered_ecg_smoothed[r:r+80]  # Okno pro vyhledávání vln T
    if len(search_window) > 0:
        t_peak = r + np.argmax(search_window)
        if filtered_ecg_smoothed[t_peak] > np.mean(search_window) + 0.5 * np.std(search_window):
            peaks_T.append(t_peak)


fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Graf R-špičky
axs[0].plot(filtered_ecg, label="Filtered ECG", alpha=0.6, color="deepskyblue")
axs[0].set_title("R-peaks Detection")
axs[0].set_ylabel("Amplitude")
axs[0].legend()
axs[0].grid(True)

# Graf vlny P
axs[1].plot(filtered_ecg, label="Filtered ECG", alpha=0.6, color="deepskyblue")
axs[1].scatter(peaks_P, filtered_ecg[peaks_P], color="orange", label="P-waves", s=15)
axs[1].set_title("P-waves Detection")
axs[1].set_ylabel("Amplitude")
axs[1].legend()
axs[1].grid(True)

# Graf vlny T
axs[2].plot(filtered_ecg, label="Filtered ECG", alpha=0.6, color="deepskyblue")
axs[2].scatter(peaks_T, filtered_ecg[peaks_T], color="lime", label="T-waves", s=15)
axs[2].set_title("T-waves Detection")
axs[2].set_xlabel("Time (samples)")
axs[2].set_ylabel("Amplitude")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()

#### Analýza anomálií vln P a T

if len(peaks_P) > 0:
    mean_p, std_p = np.mean(filtered_ecg[peaks_P]), np.std(filtered_ecg[peaks_P])
    anomalous_p = np.array(peaks_P)[np.abs(filtered_ecg[peaks_P] - mean_p) > 2 * std_p]
else:
    anomalous_p = []

if len(peaks_T) > 0:
    mean_t, std_t = np.mean(filtered_ecg[peaks_T]), np.std(filtered_ecg[peaks_T])
    t_threshold = 0.5 * std_t 
    anomalous_t = np.array(peaks_T)[np.abs(filtered_ecg[peaks_T] - mean_t) > t_threshold]
else:
    anomalous_t = []

# Vizualizace abnormalit vln P a T
plt.figure(figsize=(12, 5))
plt.plot(filtered_ecg, label="Filtered ECG", alpha=0.6, color="deepskyblue")

plt.scatter(anomalous_p, filtered_ecg[anomalous_p], color="cyan", label="Anomalous P-waves", s=15)
plt.scatter(anomalous_t, filtered_ecg[anomalous_t], color="magenta", label="Anomalous T-waves", s=15)

plt.title("Anomalies in P- and T-waves")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()


####### Sčítání statistik vln P a T
df_p_t_stats = pd.DataFrame({
    "Total P-waves": [len(peaks_P)],
    "Anomalous P-waves": [len(anomalous_p)],
    "Total T-waves": [len(peaks_T)],
    "Anomalous T-waves": [len(anomalous_t)]
})

print(df_p_t_stats)
