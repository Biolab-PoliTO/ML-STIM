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
    print("Filter coefficients initialized")
    return b, a

def filter_data(data, b, a):
    data = data.cpu().numpy()
    filtered_data = data
    for key in b:
        filtered_data = lfilter(b[key].cpu().numpy(), a[key].cpu().numpy(), filtered_data)
    filtered_data = torch.tensor(filtered_data, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return filtered_data

def remove_artifact_50overlap(data, fsamp):
    L = data.shape[0]
    params = {'w': int(100e-3 * fsamp), 'threshold': 1.33}
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

    artMask[-1] = True
    clearSignal = data[~artMask]
    return clearSignal

def extract_segment_features_min_max(data, fs, max_vals, min_vals):
    segment_length = fs
    # num_segments = data.shape[0] // segment_length
    all_features = []

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