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
        meta = pd.read_csv(metapath, sep=';')
        # meta = pd.read_feather(metapath)
        lens = meta['length'].to_numpy()
        clss = meta['class'].to_numpy()

    print(f"Data loaded with shape: {data.shape}")
    return data, clss, lens, meta

#%% Load trained model
from trained_model.MLP_architecture import MLP_STIM

def load_model(model_path):
    model = MLP_STIM(9, 1).to(device)
    mapped_state_dict = torch.load(os.path.join(model_path,'MLP_parameters.pth'), weights_only=True, map_location=device)
    model.load_state_dict(mapped_state_dict)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

#%% Main processing functions

def process_recording(model, recording, fsamp, b, a):
    start_time = time.time()
    filtered_data = lib.filter_data(recording, b, a)
    artifact_free_data, _ = lib.remove_artifact(filtered_data, fsamp)
    if artifact_free_data.shape[0] < fsamp:
        predictions = np.array([])
        dt = time.time() - start_time
    else:
        features = lib.extract_segment_features_min_max(artifact_free_data, fsamp)
        features = torch.tensor(features, dtype=torch.float32).to(device)
        with torch.no_grad():
            predictions = model(features)
            predictions = torch.sigmoid(predictions)
        dt = time.time() - start_time
        predictions = predictions.cpu().numpy()
    return predictions, dt

def main(model, raw_data, lens, fsamp, b, a):
    with tqdm_joblib(tqdm(total=raw_data.shape[0], desc="Processing Recordings")):
        batch_size = 10
        results = []
        for i in tqdm(range(0, raw_data.shape[0], batch_size), desc="Processing Batches"):
            batch_results = Parallel(n_jobs=4, timeout=3600)(
                delayed(process_recording)(model, raw_data[j, :lens[j]], fsamp, b, a)
                for j in range(i, min(i + batch_size, raw_data.shape[0])))
            results.extend(batch_results)
    predictions, dts = zip(*results)
    return predictions, dts

def save_results(predictions, dts, meta, name):
    all_predictions = []
    mean_predictions = []
    all_meta = []
    for i, preds in enumerate(predictions):
        mean_predictions.append(np.mean(preds))
        for pred in preds:
            all_predictions.append(pred[0])
            all_meta.append(meta.iloc[i])
    df = pd.DataFrame(all_meta)
    df['prediction'] = all_predictions
    filename = f'preds_{name}.csv'
    df.to_csv(filename, index=False, sep=';')

    df = pd.DataFrame(meta)
    df['prediction'] = np.array(mean_predictions)
    df['time'] = np.array(dts,dtype=np.float32)
    filename = f'preds_per_rec_{name}.csv'
    df.to_csv(filename, index=False, sep=';')

#%%

model_path = os.path.join(os.getcwd(),'trained_model')
fsamp = 24000
b, a = lib.initialize_filter_coefficients(fsamp)

model = load_model(model_path)

# filepath = "//192.168.164.127/public/projects/2024_UniTO_MERs_Classification/Cieciersky_2023_TESTING/raw_data/raw_all.npz"
# metapath = "//192.168.164.127/public/projects/2024_UniTO_MERs_Classification/Cieciersky_2023_TESTING/raw_data/raw_all.feather"
filepath = 'data.npz'
metapath = 'metadata.csv'
print('Loading data...')
raw_data, clss, lens, meta = load_data(filepath, metapath)
predictions, dts = main(model, raw_data, lens, fsamp, b, a)

save_results(predictions, dts, meta, '')

# main_path = "//192.168.164.127/public/projects/2024_UniTO_MERs_Classification/myMethod_Python_testing/all_subdatasets"
# for n in range(1, 25):
#     filepath = os.path.join(main_path,f"raw_all_{n}.npz")
#     metapath = os.path.join(main_path,f"raw_all_{n}.feather")
#     print('Loading data...')
#     raw_data, clss, lens, meta = load_data(filepath, metapath)
#     predictions, dts = main(model, raw_data, lens, fsamp, b, a)
#     save_results(predictions, dts, meta, f'{n}')