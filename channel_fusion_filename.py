
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
import random
import torch.optim as optim

# Read parquet files and return a dataframe
def load_data_parquet(path):
    df = pq.read_table(path).to_pandas()
    return df

# Read .npz file and return a dataframe
def load_data_npz(path):
    df = np.load(path, allow_pickle=True)
    print(list(df.keys()))
    return df

# Load data paths
path_parquet = r'C:\Users\Usuario\OneDrive\Escriptori\UAB\4t\psiv\4-epilepsia\dataset\chb01_seizure_metadata_1.parquet'
path_npz = r'C:\Users\Usuario\OneDrive\Escriptori\UAB\4t\psiv\4-epilepsia\dataset\chb01_seizure_EEGwindow_1.npz'

patient = load_data_npz(path_npz)['EEG_win']
metadata = load_data_parquet(path_parquet)

# Convert the dataframes into tensors
tensor_patient = torch.from_numpy(patient.astype(np.float32))
print(tensor_patient.shape)
tensor_class = torch.tensor(metadata['class'])
print(tensor_class)

#############################################################
# Fusio tensors
#############################################################

# Fusio usant concatenation
patient_concatenation = tensor_patient.view(tensor_patient.shape[0], tensor_patient.shape[1]*tensor_patient.shape[2]).unsqueeze(1)  # [Nbatch, 1, Nchannels*L]
print(patient_concatenation.shape)

# Fusio usant average
avg_pool = nn.AvgPool2d((21, 1))
patient_average = avg_pool(tensor_patient)  # [Nbatch, 1, L]
print(patient_average.shape)


#Fusio usant Weighted
#FER MÉS ENDAVANT

#############################################################
# Convolutional Unit
#############################################################

class CNN1D(nn.Module):
    def __init__(self, input_length, num_classes=2):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=8, stride=2, padding=3),   # 16 filtros @ T/4
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)       # T/8
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=8, stride=2, padding=3),  # 32 filtros @ T/16
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)       # T/32
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=1),  # 64 filtros @ T/64
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)       # T/128
        )

        #Per aconseguir el output length sha de fer aquest calcul per cada capa de convolucio
        #otuput_length = (input_length + 2*padding - kernel_size)/ stride + 1

        # Flatten intermedio para el FC de 128 neuronas
        self.fc1 = nn.Linear(64*3, 128)         #Average o Weighted
        # self.fc1 = nn.Linear(64*83, 128)      #Flatten
        
        # Segundo Flatten para clasificación final
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Primer flatten para salida al FC intermedio
        x_flat1 = x.view(x.size(0), -1)  # (batch_size, features)
        x_128 = self.fc1(x_flat1)
        
        # Segundo flatten desde la salida anterior
        output = self.fc2(x_128)
        return x_128, output


# Group filenames per number
f = list(metadata['filename'])
grouped = {}

for item in f:
    parts = item.split('_')
    if len(parts) > 1:
        number = parts[1].split('.')[0]  # Obtenir el número entre "_" i ".edf"
        if number not in grouped:
            grouped[number] = []
        grouped[number].append(item)

# Divideix els grups en entrenament i test
group_keys = list(grouped.keys())
random.shuffle(group_keys)  # Mescla aleatòriament els grups

train_keys = group_keys[:int(0.8 * len(group_keys))]  # 80% per entrenament
test_keys = group_keys[int(0.8 * len(group_keys)):]  # 20% per test

# Obtenir els índexs corresponents per cada partició
train_filenames = [filename for key in train_keys for filename in grouped[key]]
test_filenames = [filename for key in test_keys for filename in grouped[key]]

train_indices = [i for i, filename in enumerate(metadata['filename']) if filename in train_filenames]
test_indices = [i for i, filename in enumerate(metadata['filename']) if filename in test_filenames]

# Crea datasets de train i test
dataset = TensorDataset(patient_average, tensor_class)  # tensor_class ha de tenir valors 0 o 1
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Comprova els resultats
print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Batch size ajustat
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Inicializa el modelo con longitud de entrada
model = CNN1D(input_length=128*21, num_classes=2)  # El modelo implementado previamente

# Definimos la pérdida y el optimizador
criterion = nn.CrossEntropyLoss()  # Cambiamos BCELoss a CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        # Paso hacia adelante: obtenemos la salida final y la intermedia
        _, outputs = model(inputs)
        
        # Calcula la pérdida (CrossEntropyLoss espera labels de tipo Long)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluación del modelo
model.eval()
correct = 0
total = 0
predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        _, outputs = model(inputs)
        
        predicted = torch.argmax(outputs, dim=1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predictions.extend(predicted.cpu().numpy())

accuracy = correct / total
print(f"Accuracy en el test: {accuracy * 100:.2f}%")

print("Primeras 50 prediccions:", predictions[:50])