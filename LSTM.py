from Models.EpilepsyLSTM import EpilepsyLSTM  # Ensure the path is correct
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold
import os
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import pickle
import torch.nn as nn
import random


### DEFINE VARIABLES
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
N_CLASSES = 2  # Number of classes: 2 = {seizure, non-seizure}

# Default hyperparameters
def get_default_hyperparameters():
    # Initialize dictionaries
    inputmodule_params = {}
    net_params = {}
    outmodule_params = {}
    
    # Network input parameters
    inputmodule_params['n_nodes'] = 21  # Number of features
    
    # LSTM unit parameters
    net_params['Lstacks'] = 1  # Stacked layers (num_layers)
    net_params['dropout'] = 0.0
    net_params['hidden_size'] = 256  # Hidden state size
    
    # Network output parameters
    outmodule_params['n_classes'] = 2
    outmodule_params['hd'] = 128  # Hidden state size for the output module
    
    return inputmodule_params, net_params, outmodule_params


#############################################################

# Read parquet files and return a dataframe
def load_data_parquet(path, file):
    num = file[3:5]
    df = pq.read_table(path).to_pandas()
    return df, num

# Read .npz file and return a dataframe
def load_data_npz(path, file):
    num = file[3:5]
    df = np.load(path, allow_pickle=True)
    
    return df['EEG_win'], num
    

def diccionari_intervals(parquet_files,npz_files):
    dic = {}
    
    for idx, row in parquet_files.iterrows():
        global_interval = row['global_interval']
        class_value = row['class']
        signal_value = npz_files[idx]
        
        
        if global_interval not in dic:
            dic[global_interval] = [[], []]
        
        dic[global_interval][0].append(class_value)  
        dic[global_interval][1].append(signal_value)  
    
    return dic

def processar_dades(path_directori):
    parquet_files = {}
    npz_files = {}
    
    dic_final = {}
    
    for file in os.listdir(path_directori):
        file_path = os.path.join(path_directori, file)
        if file_path.endswith(".parquet"):
            df_metadates, num = load_data_parquet(file_path, file)
            parquet_files[num] = df_metadates
           
        else:
            df, num = load_data_npz(file_path, file)
            #tensor_patient = torch.from_numpy(df.astype(np.float32))
            npz_files[num] = df
    
    
    for pacient in parquet_files:
        
        
          dic_final[pacient] = diccionari_intervals(parquet_files[pacient], npz_files[pacient])
        

    return dic_final
    
path_directori = "/export/fhome/maed/EpilepsyDataSet/"
#path_directori = "/export/fhome/maed06/VirtualEnv/repte4/dataset/"

dic = processar_dades(path_directori)


y_train=[]
y_test=[]
X_train = []
X_test = []
llargades_intervals_train = []
llargades_intervals_test = []
for pacient,d in dic.items():
    intervals=list(d.keys())
   
    random.shuffle(intervals)
    
    iterval_test=intervals.pop()
    X_test.extend(d[iterval_test][1])
    y_test.extend(d[iterval_test][0])
    llargades_intervals_test.append(len(d[iterval_test][0]))
    for interval in intervals:
        y_train.extend(d[interval][0])
        X_train.extend(d[interval][1])
        llargades_intervals_train.append(len(d[interval][1]))


with open("X_test.pkl", "wb") as file:  
    pickle.dump(data, file)
with open("y_test.pkl", "wb") as file:  
    pickle.dump(data, file)

def generar_finestres(llargades_intervals, finestra_size):
    finestres = []
    start_index = 0  

    for llargada in llargades_intervals:
        n_finestra = llargada // finestra_size  
        residue = llargada % finestra_size  

        
        for i in range(n_finestra):
            start = start_index + i * finestra_size
            end = start + finestra_size
            finestres.append((start, end))

        
        if residue > 0:
        
            start = start_index + n_finestra * finestra_size
            end = start + residue
           
            if finestres:
                finestres[-1] = (finestres[-1][0], end)  
            else:
                finestres.append((start, end))

        
        start_index += llargada

    return finestres

window_size = 5
finestres_train = generar_finestres(llargades_intervals_train,window_size)
finestres_test = generar_finestres(llargades_intervals_test,window_size)

with open("finestres_test.pkl", "wb") as file:  
    pickle.dump(data, file)

inputmodule_params, net_params, outmodule_params = get_default_hyperparameters()
model = EpilepsyLSTM(inputmodule_params, net_params, outmodule_params)
model.init_weights()
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Convert the data to PyTorch tensors
X_train = np.array(X_train, dtype=np.float32)
X_tensor = torch.from_numpy(X_train).float().to(DEVICE)
y_train = np.array(y_train, dtype=np.int64)
y_tensor = torch.from_numpy(y_train).long().to(DEVICE)


# Training loop
n_epochs = 20


for epoch in range(n_epochs):
    model.train()
    for start,end in finestres_train:
        batch_X = X_tensor[start:end]
        batch_y = y_tensor[start:end]
        print(start,end)
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')


torch.save(model.state_dict(), "./modelLSTM-windows20epoch5windowsize.pth")



