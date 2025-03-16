import matplotlib.pyplot as plt
import numpy as np

def plot_signals(patient, side, data, meta, fsamp):

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
    fig, ax = plt.subplots(figsize=(15, 8))
    for i, signal in enumerate(normalized_signals):
        c = '#212e47'
        if patient_meta.iloc[i]['class'] == 1: c = '#e26d0e'
        ax.plot(time_vector, signal - i, linewidth=0.3, color=c)  # Offset each signal for clarity
    
    ax.set_yticks(-np.arange(len(normalized_signals)))
    ax.set_yticklabels(patient_meta['depth']/1000)  # Convert depth to mm
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Estimated Distance from Target (mm)')
    ax.set_title(f'Signals for {patient} - {side}')
    # Increase font size
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.title.set_size(18)

    plt.show()
    
    return fig