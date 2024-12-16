
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
import random
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import KFold

#############################################################
# Load data functions
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
    print(list(df.keys()))
    return df['EEG_win'], num

def processar_dades(path_directori):
    parquet_files = {}
    npz_files = {}
    
    for file in os.listdir(path_directori):
        file_path = os.path.join(path_directori, file)
        if file_path.endswith(".parquet"):
            df_metadates, num = load_data_parquet(file_path, file)
            parquet_files[num] = df_metadates
        else:
            df, num = load_data_npz(file_path, file)
            tensor_patient = torch.from_numpy(df.astype(np.float32))
            npz_files[num] = tensor_patient

    return parquet_files, npz_files


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


#############################################################
# Main
#############################################################

path_directori = "C:/Users/arnau/Desktop/4t Eng/1r Semestre/PSIV 2/Reptes/Epilepsia/Sample of original  EEG Recording-20241205/Dataset_petit"

parquet_pacients, npz_pacients = processar_dades(path_directori)

tensors_train = []
tensors_test = []

avg_pool = nn.AvgPool2d((21, 1))
# patient_concatenation = tensor_patient.view(tensor_patient.shape[0], tensor_patient.shape[1]*tensor_patient.shape[2]).unsqueeze(1)  # [Nbatch, 1, Nchannels*L]


for num_p, metadata in parquet_pacients.items():
    grouped = metadata.groupby(['filename'])
    ultim = grouped.size().index[-1]
    pacient_average = avg_pool(npz_pacients[num_p])

    positives_test = []
    positives_train = []
    negatives_test = []
    negatives_train = []
    for i, filename in enumerate(metadata['filename']):
        if filename == ultim:
            if metadata['class'][i] == 1:
                positives_test.append(i)
            else:
                negatives_test.append(i)
        else:
            if metadata['class'][i] == 1:
                positives_train.append(i)
            else:
                negatives_train.append(i)

    length_train = min(len(positives_train), len(negatives_train))
    length_test = min(len(positives_test), len(negatives_test))
    
    train_indices = positives_train[:length_train] + negatives_train[:length_train]
    test_indices = positives_test[:length_test] + negatives_test[:length_test]

    #Crear 2 tensors per pacient, train con grouped menos el último filename y el test solo con los que tengan el último filename
    tensor_class = torch.tensor(metadata['class'])
    
    dataset = TensorDataset(pacient_average , tensor_class)  # tensor_class ha de tenir valors 0 o 1
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f"Pacient {num_p}: Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    tensors_train.append(train_dataset)
    tensors_test.append(test_dataset)

#Concatenar tots els tensors
train_dataset = torch.utils.data.ConcatDataset(tensors_train)
test_dataset = torch.utils.data.ConcatDataset(tensors_test)

print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Batch size ajustat
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Inicializa el modelo con longitud de entrada
model = CNN1D(input_length=128, num_classes=2)  # El modelo implementado previamente

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


model.eval()


total = 0
correct = 0
predictions = []
true_labels = []


with torch.no_grad():
    for inputs, labels in test_loader:
        _, outputs = model(inputs)
        
        # Prediccions
        predicted = torch.argmax(outputs, dim=1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Calculant accuracy
accuracy = correct / total
print(f"Accuracy en el test: {accuracy * 100:.2f}%")

# Càlcul de la matriu de confusió
conf_matrix = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Càlcul de les mètriques de recall

recall_per_class = precision_recall_fscore_support(true_labels, predictions, average=None)[1]

recall_pos = recall_per_class[1]  # Per la classe 1
recall_neg = recall_per_class[0]  # Per la classe 0

print(f"Recall Positiu (classe 1): {recall_pos:.2f}")
print(f"Recall Negatiu (classe 0): {recall_neg:.2f}")

print("Primeras 50 prediccions:", predictions[:50])

