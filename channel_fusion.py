from load_data import load_data_parquet, load_data_npz
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

#############################################################
# Load data
#############################################################

patient = load_data_npz("C:/Users/arnau/Desktop/4t Eng/1r Semestre/PSIV 2/Reptes/Epilepsia/Sample of original  EEG Recording-20241205/input/chb02_seizure_EEGwindow_1.npz")['EEG_win']
metadata = load_data_parquet("C:/Users/arnau/Desktop/4t Eng/1r Semestre/PSIV 2/Reptes/Epilepsia/Sample of original  EEG Recording-20241205/input/chb02_seizure_metadata_1.parquet")

#Transoforma els dataframe en tensors
tensor_patient = torch.from_numpy(patient.astype(np.float32))
print(tensor_patient.shape)
tensor_class = torch.tensor(metadata['class'])
print(tensor_class)

#############################################################
# Fusio tensors
#############################################################

#Fusio usant concatenation
patient_concatenation = tensor_patient.view(tensor_patient.shape[0], tensor_patient.shape[1]*tensor_patient.shape[2]).unsqueeze(1)# [Nbatch, 1, Nchannels*L]
print(patient_concatenation.shape)

#Fusio usant average
avg_pool = nn.AvgPool2d((21,1))
patient_average = avg_pool(tensor_patient) #[Nbatch, 1, L]
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

# Creación del dataset
dataset = TensorDataset(patient_average, tensor_class)  # tensor_class debe tener valores 0 o 1
train_size = int(0.8 * len(dataset)) 
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Batch size ajustado
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
