from scipy import signal
import torch
import numpy as np
from scipy.signal import lfilter, welch
from scipy.stats import skew, kurtosis
import pandas as pd

def initialize_filter_coefficients(fsamp):
    b = {}
    a = {}
    nyquist = 0.5 * fsamp
    f_high = 250
    b['high'], a['high'] = signal.butter(4, f_high / nyquist, btype='highpass')
    f_low = 5000
    b['low'], a['low'] = signal.butter(8, f_low / nyquist, btype='lowpass')
    notch_freqs = np.arange(150, 5000, 50)
    for freq in notch_freqs:
        b[f'notch_{freq}Hz'], a[f'notch_{freq}Hz'] = signal.iirnotch(freq, 50, fsamp)
    for key in b:
        b[key] = torch.tensor(b[key], dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for key in a:
        a[key] = torch.tensor(a[key], dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return b, a

def filter_data(data, b, a):
    # data = data.cpu().numpy()
    filtered_data = data
    for key in b:
        filtered_data = lfilter(b[key].cpu().numpy(), a[key].cpu().numpy(), filtered_data)
    filtered_data = torch.tensor(filtered_data, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return filtered_data

def remove_artifact(data, fsamp):
    L = data.shape[0]
    params = {'w': int(100e-3 * fsamp), 'threshold': 1.44}
    artMask = torch.zeros(L, dtype=torch.bool, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    Wbis = 0
    t = 0
    start = 0
    stop = params['w']
    pvar = 1e14
    first_it = True    
    s = params['w'] // 2
    w = params['w']
    while start < L:
        epoch = data[start:stop]
        nvar = torch.var(epoch)
        if nvar >= params['threshold'] * pvar:
            W = [1,1]
        else:
            W = [0,0]
            pvar = nvar
        if first_it:
            Wbis = W[0]
            first_it = False

        if stop - start < s:
            s = stop - start

        cls = torch.mean(torch.tensor([W[0], Wbis], dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        artMask[t:t + s] = cls >= 0.5

        Wbis = W[1]

        t += s
        start += s
        stop = start + w
        if stop > L:
            stop = L
            w = stop - start

    clearSignal = data[~artMask]
    return clearSignal, artMask

def extract_segment_features_min_max(data, fs):
    segment_length = fs
    # num_segments = data.shape[0] // segment_length
    all_features = []

    # normalization intervals computed from the training set
    max_vals = [8.707813,0.016264,0.073682,0.133387,0.16357,2769,0.435164,0.936993,2.680869]
    min_vals = [1.357211,0.002564,0.019996,-0.250695,0.041139,1644.88,0.153334,0.833393,-0.096321]

    overlap = 0.5
    step = int(segment_length * (1 - overlap))

    for start in range(0, data.shape[0] - segment_length + 1, step):
        segment = data[start:start + segment_length]

        # PSD calculation
        freqs, Pxx = welch(segment, fs, nperseg=fs // 10)
        P_tot = np.sum(Pxx)
        PSD = Pxx / P_tot
        freqsA, PxxA = welch(np.abs(segment), fs, nperseg=fs // 10)
        P_totA = np.sum(PxxA)
        PSDA = PxxA / P_totA
       
        # Feature extraction
        avgAbsDiff = np.mean(np.abs(segment - np.mean(segment)))
        pr_8_13Hz = np.sum(PSDA[(freqsA >= 8) & (freqsA <= 13)])
        pr_30_70Hz = np.sum(PSDA[(freqsA >= 30) & (freqsA <= 70)])
        skewness = skew(segment)
        PSDratio = np.sum(Pxx[(freqs >= 3000) & (freqs <= 4000)]) / np.sum(Pxx[(freqs >= 1600) & (freqs <= 2200)])
        zc_count = ((segment[:-1] * segment[1:]) < 0).sum()
        pr_1_2kHz = np.sum(PSD[(freqs >= 1000) & (freqs <= 2000)])
        PSDindex = 0
        for j in range(22):
            PSDindex += np.sum(PSD[(freqs >= 255 + j * 100) & (freqs <= 245 + (j + 1) * 100)])   
        kurt = kurtosis(segment)        

        features = np.array([avgAbsDiff, pr_8_13Hz, pr_30_70Hz, skewness, PSDratio, zc_count, pr_1_2kHz, PSDindex, kurt])
        all_features.append(features)

    all_features = np.stack(all_features)

    for i in range(all_features.shape[1]):
        all_features[:, i] = (all_features[:, i] - min_vals[i]) / (max_vals[i] - min_vals[i])
    all_features = np.clip(all_features, 0, 1)
    return all_features

def plot_signals(patient, side, data, meta, fsamp):
    import matplotlib.pyplot as plt

    # Filter the metadata for the given patient and side
    patient_meta = meta[(meta['patient'] == patient) & (meta['side'] == side)]
    
    # Sort by depth
    patient_meta = patient_meta.sort_values(by='depth', ascending=True)
    
    # Extract the signals
    signals = []
    for idx, row in patient_meta.iterrows():
        signal = data[idx, :row['length']]
        signals.append(signal)
    
    # Normalize signals with respect to the median standard deviation
    all_signals = np.concatenate(signals)
    median_std = np.median(np.std(all_signals))
    normalized_signals = [signal / (15*median_std) for signal in signals]

    # Define a time vector for plotting
    time_vector = np.arange(len(normalized_signals[0])) / fsamp
    
    # Plot the signals
    fig, ax = plt.subplots(figsize=(15, 10))
    for i, signal in enumerate(normalized_signals):
        c = '#212e47'
        if patient_meta.iloc[i]['class'] == 1: c = '#e26d0e'
        ax.plot(time_vector, signal - i, linewidth=0.3, color=c)  # Offset each signal for clarity
    
    ax.set_yticks(-np.arange(len(normalized_signals)))
    ax.set_yticklabels(patient_meta['depth']/1000)  # Convert depth to mm
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Estimated Distance from Target (mm)')
    ax.set_title(f'Signals for {patient} - {side}')
    plt.show()