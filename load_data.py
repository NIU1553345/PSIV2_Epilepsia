#Read parquet files and return a dataframe

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os

#Read a parquet and return a dataframe
def load_data_parquet(path):
    # Read parquet files
    df = pq
    df = pq.read_table(path).to_pandas()
    return df

# print(load_data_parquet("C:/Users/arnau/Desktop/4t Eng/1r Semestre/PSIV 2/Reptes/Epilepsia/Sample of original  EEG Recording-20241205/input/chb01_seizure_metadata_1.parquet"))


#Read .npz file and return a dataframe
def load_data_npz(path):
    # Read npz files
    df = np.load(path, allow_pickle=True)
    print(list(df.keys()))
    return df

df = load_data_npz("C:/Users/arnau/Desktop/4t Eng/1r Semestre/PSIV 2/Reptes/Epilepsia/Sample of original  EEG Recording-20241205/input/chb01_seizure_EEGwindow_1.npz")['EEG_win']
print(df.shape)


