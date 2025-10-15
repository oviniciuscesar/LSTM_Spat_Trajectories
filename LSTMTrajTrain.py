import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os

# --- 1 CONFIGURAÇÃO E HIPERPARÂMETROS ---

SEED = 42  # Escolha qualquer valor inteiro
np.random.seed(SEED)
torch.manual_seed(SEED)


# diretórios
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
model_dir = os.path.join(script_dir, 'models')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# Features de entrada para cada passo da sequência
INPUT_SIZE_PER_STEP = 8  # t, traj_type1, traj_type2, traj_type3, r, sin, cos, dur
LSTM_HIDDEN_SIZE = 64  # Tamanho do estado oculto da LSTM
LSTM_NUM_LAYERS = 2  # Número de camadas da LSTM
OUTPUT_SIZE = 4  # saída da LSTM: r_norm, sin, cos, dur_norm

# Parâmetros da sequência e treinamento
SEQUENCE_LENGTH = 10  # usar os últimos 10 pontos para prever o próximo
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
NUM_EPOCHS = 1500

# arquivo de dados e modelo
DATA_FILE_PATH = os.path.join(data_dir, '_SpatTrajData.txt')
MODEL_SAVE_PATH = os.path.join(model_dir, 'lstm_model.pth')


# --- 2 CLASSE DE DATASET PARA LSTM ---
class TrajectorySequenceDataset(Dataset):
    """
        Carrega os dados a partir do arquivo. txt e cria sequências deslizantes:
        Para cada ponto i, a sequência de entrada é:
        [i, i+1, ..., i+sequence_length-1]
        E o label é o ponto seguinte:
        [i+sequence_length]
    """
    def __init__(self, txt_file, sequence_length):
        self.sequence_length = sequence_length
        
        with open(txt_file) as fin:
            lines = [line.replace(';', '') for line in fin]
        data = np.loadtxt(lines, delimiter=' ')
        
        # features: t, type1, type2, type3, r, sin, cos, dur (todas as 8 colunas)
        # labels: r, sin, cos, dur (as últimas 4 colunas)
        all_features = data
        labels_only = data[:, 4:]
    
        self.sequences = []
        self.labels = []
        
        # cria sequências com janela deslizante
        for i in range(len(all_features) - sequence_length):
            self.sequences.append(all_features[i : i + sequence_length])
            self.labels.append(labels_only[i + sequence_length])

        self.sequences = torch.from_numpy(np.array(self.sequences)).float()
        self.labels = torch.from_numpy(np.array(self.labels)).float()

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]

    def __len__(self):
        return len(self.sequences)


# --- 3 DEFINIÇÃO DO MODELO LSTM ---
"""
    LSTM: prevê as coordenadas de um ponto na trajetória a partir das coordenadas dos pontos anteriores.
    Entrada: sequência de pontos (batch_size, sequence_length, input_size)
    Saída: próximo ponto (batch_size, output_size)  
    Arquitetura:
    - LSTM com nº de camadas = num_layers e unidades ocultas = hidden_size
    - Camada Linear com tamanho = output_size para mapear o estado oculto para a saída desejada (r, sin, cos, dur)
"""
class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    # forward: LSTM -> Linear -> Tanh
    def forward(self, x, temperature: float = 1.0):
        # Inicializa o estado oculto com zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass pela LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Pega a saída do último passo de tempo da sequência
        out = out[:, -1, :]
        
        # Passa pela camada linear final
        out = self.fc(out)

        # aplica a temperatura: divide a saída da camada linear pelo valor da temperatura
        # garante que a temperatura não seja zero para evitar divisão por zero
        if temperature <= 0.0:
            temperature = 1.0
        out_temp = out / temperature

        # Passa pela função de ativação final
        out_final = self.tanh(out_temp)
        return out_final


# --- 4 TREINAMENTO ---
"""
    Treina o modelo LSTM usando os dados do arquivo txt.
    Usa Early Stopping para evitar overfitting.
    Otimização com Adam e perda MSE.
"""
if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Usando o dispositivo: {device}')
    
    # Carregar e preparar os dados
    full_dataset = TrajectorySequenceDataset(DATA_FILE_PATH, SEQUENCE_LENGTH)
    train_size = int(0.8 * len(full_dataset)) # 80% para treino
    val_size = len(full_dataset) - train_size # 20% para validação
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # criar dataloaders para batch training
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Instanciar modelo, perda e otimizador
    model = LSTM_Model(INPUT_SIZE_PER_STEP, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, OUTPUT_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nArquitetura do Modelo:")
    print(model)

    # Lógica de Early Stopping
    best_val_loss = float('inf')
    patience = 200
    patience_counter = 0

    print("\nIniciando o treinamento do modelo LSTM...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, labels in val_loader:
                outputs = model(sequences.to(device))
                loss = criterion(outputs, labels.to(device))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f'Época [{epoch+1}/{NUM_EPOCHS}], Perda de Validação: {avg_val_loss:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Parada antecipada na época {epoch+1}. Melhor perda de validação: {best_val_loss:.6f}")
            break
            
    print("Treinamento concluído.")
    print(f"O melhor modelo LSTM foi salvo em: {MODEL_SAVE_PATH}")