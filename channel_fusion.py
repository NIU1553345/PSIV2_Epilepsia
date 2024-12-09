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

patient = load_data_npz("C:/Users/arnau/Desktop/4t Eng/1r Semestre/PSIV 2/Reptes/Epilepsia/Sample of original  EEG Recording-20241205/input/chb01_seizure_EEGwindow_1.npz")['EEG_win']
metadata = load_data_parquet("C:/Users/arnau/Desktop/4t Eng/1r Semestre/PSIV 2/Reptes/Epilepsia/Sample of original  EEG Recording-20241205/input/chb01_seizure_metadata_1.parquet")

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
    def __init__(self):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1)  
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)


        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1)  
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1)  
        self.relu3 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))  

        print(f"Forma de x despres de la 1ra capa: {x.shape}")

        x = self.pool1(self.relu2(self.conv2(x)))  

        print(f"Forma de x despres de la 2na capa: {x.shape}")

        x = self.pool1(self.relu3(self.conv3(x)))  

        print(f"Forma de x despres de la 3ra capa: {x.shape}")

        x = x.view(x.size(0), -1)  
        
        print(f"Forma de x antes de la capa fully connected: {x.shape}")

        x = self.fc(x)  
        x = torch.sigmoid(x)

        return x
    
dataset = TensorDataset(patient_average, tensor_class)
train_size = int(0.8 * len(dataset)) 
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = CNN1D()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)  
        loss = criterion(outputs.squeeze(), labels.squeeze())  
        loss.backward()  
        optimizer.step()  
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

model.eval() 
correct = 0
total = 0
predictions = []

with torch.no_grad(): 
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs.squeeze() > 0.5).float() 
        total += labels.size(0)
        correct += (predicted.squeeze() == labels.squeeze()).sum().item()
        predictions.extend(predicted.squeeze().cpu().numpy())

accuracy = correct / total
print(f"Accuracy en el conjunto de prueba: {accuracy * 100:.2f}%")

# Mostrar las primeras 10 predicciones
print("Primeras 10 predicciones:", predictions[:10])

                                                        










