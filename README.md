# ML-STIM
**Machine Learning for SubThalamic nucleus Intraoperative Mapping**

<p align="center">

<img  src="https://github.com/Biolab-PoliTO/ML-STIM/blob/main/docs/ML-STIM_figure.jpg" style="width:100%; height:auto;"/></p>

Deep Brain Stimulation (DBS) of the SubThalamic Nucleus (STN) is an effective electroceutical therapy for treating motor symptoms in patients with Parkinson’s disease. Accurate placement of the stimulating electrode within the STN is essential for achieving optimal therapeutic outcomes. To this end, MicroElectrode Recordings (MERs) are acquired during surgery to provide intraoperative visual and auditory confirmation of the electrode position.

This work introduces ```ML-STIM```, a machine learning-based pipeline for real-time classification of MERs to identify the STN during DBS procedures. ```ML-STIM``` is designed for high classification accuracy and real-time applicability. It incorporates interpretable machine learning techniques to ensure compatibility with clinical practices.

## What ```ML-STIM``` algorithm does:
1.	Load MERs stored as rows of a `numpy` array `.npz`, together with a metafile `.csv`;
2.	Apply `ML-STIM` pipeline to each MER;
3. 	Export results in ```.csv``` format.

## Files description:
The following files are provided within this GitHub repository:
- `main.py`: main script to execute `ML-STIM`
- `lib.py`: is a collection of function called within `main.py` to perform pre-processing (filtering + artifact removal) and feature extraction.
- `trained_model`: the folder includes the trained model:
	- `MLP_architecture.py`: it initialize the MultiLayer Perceptron (MLP) with predefined architecture
	- `MLP_parameters.pth`: trained parameters for the model
	- `normalization_intervals.txt`: it contains extreme values for the extracted features
</p>

## A simple workflow
A simplified workflow for a MER processing and classification looks as follows.

1. data loading:

```r
# Path definition
filepath = "path/to/raw_data.npz"
metapath = "path/to/raw_data.csv"

with np.load(filepath) as npfh:
	raw_data = npfh['data']			# Load data matrix
meta = pd.read_feather(metapath)		# Load metadata

print(f"Data loaded with shape: {raw_data.shape}")
```
2. signal processing:

```r
# Import library
import lib

# process signal
fsamp = 24000		# Sampling frequency (Hz)
b, a = lib.initialize_filter_coefficients(fsamp)

recording = raw_data[0, :meta['length'][0].to_numpy()]
filtered_data = lib.filter_data(recording, b, a)
artifact_free_data = lib.remove_artifact(filtered_data, fsamp)
```

3. feature extraction:
```r
features = lib.extract_segment_features(artifact_free_data, fsamp)
```

4. classification:
```r
import torch
# Import architecture
from trained_model.MLP_architecture import MLP_STIM

# Define model Path
model_path = 'path/to/trained_model'

# Import model
model = MLP_STIM.to(device)	# Initialize empty model
params = torch.load(os.path.join(model_path,'MLP_parameters.pth'), 
                    weights_only=True, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.load_state_dict(params)

# Classify the recording
prediction = model(features)
prediction = torch.sigmoid(prediction)
```

## How to prepare your data:
To use this analysis framework, your data must be in ```.npz``` format.
**Example Data**
...
**Data Structure**
Your metadata file should contain a table with variables (columns):
- `patient`: patient id (e.g. `P7`)
- `side`: hemisphere (`LEFT` or `RIGHT`)
- `depth`: Estimated Distance from Target (EDT) expressed in `μm`
- `length`: signal length (in samples) before zero-padding
- `class`: label (`0` if the record is acquired from outside the STN, `1` if inside the STN)

```
	patient	side	electrode	depth	length	class
0	P1	LEFT	Electrode1	-5000	240000	0
1	P1	LEFT	Electrode1	-1000	240000	1
2	P1	LEFT	Electrode1	6000	227648	0
```
For a representative example of the expected input format, refer to the ```metadata.csv``` and ```sample.npz``` file.

## References
[1] Author, F., Author, S., Author, T. (2025). Title. Journal, chapter(edition), pp-pp. https://doi.org/link/to/doi

##  How to contribute to ```ML-STIM```
Contributions are the heart of the open-source community, making it a fantastic space for learning, inspiration, and innovation. While we've done our best, our code may contain inaccuracies or might not fully meet your needs. If you come across any issues—or have ideas for improvements—we encourage you to contribute! Follow the instructions below to suggest edits or enhancements. Every contribution is **greatly appreciated**!

Bugs are tracked as **GitHub issues**. Whenever you report an issue, please make sure to:
1.	Use a concise and descriptive title
2.	Report your MATLAB version
3.	Report whether the code ran successfully on the test data available within the repository.


## Contacts
**Fabrizio Sciscenti**, Ph.D. Candidate - [BIOLAB@Polito](https://biolab.polito.it/people/fabrizio-sciscenti/)
[@FSciscenti](https://x.com/FSciscenti) - fabrizio.sciscenti@polito.it

**Marco Ghislieri**, Ph.D. - [BIOLAB@Polito](https://biolab.polito.it/people/marco-ghislieri/)
[@MarcoGhislieri](https://twitter.com/MarcoGhislieri) - marco.ghislieri@polito.it