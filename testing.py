#%% import libraries

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import lib

#%% Check for GPU availability

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print("Running on GPU!")
else:
    print("GPU not detected. Running on CPU.")

#%% Load data
def load_data(filepath, metapath):
    with np.load(filepath) as npfh:
        data = npfh['data']
        meta = pd.read_feather(metapath)
        lens = meta['length'].to_numpy()
        clss = meta['class'].to_numpy()

    print(f"Data loaded with shape: {data.shape}")
    return torch.tensor(data, dtype=torch.float32).to(device), clss, lens, meta

filepath = "C:/Users/fabri/Desktop/scripts_final/dataset/raw/ciecierski/raw_test.npz"
metapath = "C:/Users/fabri/Desktop/scripts_final/dataset/raw/ciecierski/raw_test.feather"
print('Loading data...')
raw_data, clss, lens, meta = load_data(filepath, metapath)

# Display the first 3 rows of the table meta
print(meta.head(3))

# Plot the first n signals of raw_data
fsamp = 24000 # Hz
fig, axs = plt.subplots(1, 1, figsize=(10, 4))
time_vector = np.arange(0, lens[0]/fsamp, 1/fsamp)
axs.plot(time_vector,raw_data[0, :lens[0]].cpu().numpy())
axs.set_xlabel('Time (s)')
axs.set_ylabel('Amplitude (uV)')
plt.show()

#%% Load trained model
from trained_model.MLP_architecture import MLP_STIM

def load_model(model_path):
    norm_intervals = np.loadtxt(os.path.join(model_path, 'normalization_intervals.txt'),skiprows=1, comments=None)
    model = MLP_STIM(9, 1).to(device)
    mapped_state_dict = torch.load(os.path.join(model_path,'MLP_parameters.pth'), weights_only=True, map_location=device)
    model.load_state_dict(mapped_state_dict)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model, norm_intervals

model_path = os.path.join(os.getcwd(),'trained_model')
fsamp = 24000
b, a = lib.initialize_filter_coefficients(fsamp)

model, norm_intervals = load_model(model_path)

max_vals_np = norm_intervals[:,1]
min_vals_np = norm_intervals[:,0]
max_vals = torch.tensor(max_vals_np, dtype=torch.float32, device=device)
min_vals = torch.tensor(min_vals_np, dtype=torch.float32, device=device)

#%% Main processing functions

def process_recording(model, recording, fsamp, b, a, max_vals, min_vals):
    start_time = time.time()
    filtered_data = lib.filter_data(recording, b, a)
    artifact_free_data = lib.remove_artifact_50overlap(filtered_data, fsamp)
    if artifact_free_data.shape[0] < fsamp:
        predictions = np.array([])
        dt = time.time() - start_time
    else:
        features = lib.extract_segment_features_min_max(artifact_free_data.cpu().numpy(), fsamp, max_vals.cpu().numpy(), min_vals.cpu().numpy())
        features = torch.tensor(features, dtype=torch.float32).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        with torch.no_grad():
            predictions = model(features)
            predictions = torch.sigmoid(predictions)
        dt = time.time() - start_time
        predictions = predictions.cpu().numpy()
    return predictions, dt

def main(model, raw_data, lens, fsamp, b, a, max_vals, min_vals):
    with tqdm_joblib(tqdm(total=raw_data.shape[0], desc="Processing Recordings")):
        batch_size = 32
        results = []
        for i in tqdm(range(0, raw_data.shape[0], batch_size), desc="Processing Batches"):
            batch_results = Parallel(n_jobs=4, timeout=3600)(
                delayed(process_recording)(model, raw_data[j, :lens[j]], fsamp, b, a, max_vals, min_vals)
                for j in range(i, min(i + batch_size, raw_data.shape[0])))
            results.extend(batch_results)
    predictions, dts = zip(*results)
    return predictions, dts

predictions, dts = main(model, raw_data, lens, fsamp, b, a, max_vals, min_vals)
df = pd.DataFrame({'prediction': predictions, 'dt': dts})
filename = 'predictions.txt'
df.to_csv(filename, sep='\t', index=False)