# Arrhythmia Beat Classification

This project is an implementation of an arrhythmia beat classification system using the publicly available MIT-BIH Arrhythmia Database. The aim of this project is to classify different types of heartbeats from ECG data using machine learning techniques. We build upon the approach outlined in the article [Combining Low-dimensional Wavelet Features and Support Vector Machine for Arrhythmia Beat Classification](https://www.nature.com/articles/s41598-017-06596-z).

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Method](#method)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Classification](#classification)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)
- [Contributors](#contributors)
- [Citations](#citations)

## Introduction
The detection of heart arrhythmia from ECG signals is a key step in the diagnosis of cardiac disorders. This project focuses on building an automated arrhythmia detection system using a machine learning pipeline to classify different heartbeat types from ECG recordings.

## Dataset
We use the **[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)**, which has been a standard for arrhythmia research since 1980. The dataset consists of 48 half-hour ECG recordings, with annotations by cardiologists. These recordings include both common and less common arrhythmias, making it suitable for training classification models.

## Method
1. **Beat Segmentation**: The ECG signal is segmented into episodes based on QRS intervals using the R-peak locations provided in the dataset.
2. **Wavelet Decomposition**: Features are extracted using wavelet multi-resolution analysis (WMRA).
3. **Principal Component Analysis (PCA)**: PCA is used to reduce the dimensionality of the wavelet features.
4. **Classification**: Three machine learning algorithms are used to classify the heartbeats: **_K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Neural Networks_**.

## Preprocessing

- **Beat Segmentation**: ECG recordings are segmented into beats using QRS intervals based on R-peak locations provided by MITDB.
- **Feature Extraction**: Wavelet analysis is applied to extract features from each beat. We used **_bior_** wavelet family and specifically the **_bior 6.8_**.
- **Data Normalization**: StandardScaler is used to normalize the data.
  
   Example code:
   ```python
   import qrsintervals
   # Define extraction parameters
   mitdbih_db_path = 'mit-bih-arrhythmia-database-1.0.0'
   annotations = ['A', 'V', 'N', 'L', 'R']
   qrs_interval_length = 320
   qrsintervals.extract_qrs_intervals_and_save(mitdbih_db_path, 'ecg_signals.mat', annotations)
   ...
   # Define the feature extraction parameters
   classes = 'A', 'V', 'N', 'L', 'R'
   signals_per_class = 2546
   wavelet = pywt.Wavelet('bior6.8')
   decomposition_level = 8
   # Extract features
   ext_feat_signals = preprocessing.extract_features(
	   ECG_signals,         # The ECG signals from the ecg_signals.mat
	   classes,             # The classification classes (A, V, N, L, R)
	   signals_per_class,   # How many signals per class we have
	   wavelet,         # The Wavelet
	   decomposition_level  # The decomposition level for the Wavelet decomposition
   )
   ...
	
   ```

## Feature Extraction
Features are extracted from each ECG signal using **wavelet transformations**. These features are then reduced to a lower-dimensional representation using **Principal Component Analysis (PCA)**.

## Classification
We apply three machine learning algorithms for heartbeat classification:
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Neural Networks**

Metrics used for comparison:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

## Results

The table below summarizes the performance of each classifier on the test data:

| Algorithm             | Accuracy | Precision | Recall | F1 Score |
| --------------------- | -------- | --------- | ------ | -------- |
| **Nearest Neighbors** | 88.71%   | 88.77%    | 88.71% | 88.63%   |
| **Linear SVM**        | 92.41%   | 92.32%    | 92.41% | 92.34%   |
| **RBF SVM**           | 83.50%   | 87.50%    | 83.50% | 84.17%   |
| **Neural Net**        | 95.50%   | 95.48%    | 95.50% | 95.46%   |

## Installation and Usage

1. Clone the repository: 

   ```bash
   git clone https://github.com/your-username/arrhythmia-beat-classification.git
   cd arrhythmia-beat-classification
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:

```bash
jupyter notebook Arrhythmia_Beat_Classifier.ipynb
```


## Contributors

- Houssem Khalfi
- Chaima Jeljli

## Citations

Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). (PMID: 11446209)

Goldberger, A., et al. "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220." (2000).

Qin, Q., Li, J., Zhang, L. et al. Combining Low-dimensional Wavelet Features and Support Vector Machine for Arrhythmia Beat Classification. Sci Rep 7, 6067 (2017). https://doi.org/10.1038/s41598-017-06596-z

