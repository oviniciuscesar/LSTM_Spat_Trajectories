import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# Importa a definição do modelo do script de treino
from LSTMTrajTrain import LSTM_Model, INPUT_SIZE_PER_STEP, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, OUTPUT_SIZE, SEQUENCE_LENGTH, MODEL_SAVE_PATH

plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)


# Parâmetros de normalização e teste
MIN_DUR_NORM = 100.0
MAX_DUR_NORM = 1000.0
MIN_RADIUS_NORM = 0.0
MAX_RADIUS_NORM = 1.0
NUM_PONTOS_GERACAO = 500 # Número de pontos a serem gerados


# --- 2 FUNÇÕES AUXILIARES  ---
def desnormalizar_raio(raio_norm):
    """Converte o raio normalizado [-1, 1] de volta para a escala original."""
    return MIN_RADIUS_NORM + ((raio_norm + 1) / 2) * (MAX_RADIUS_NORM - MIN_RADIUS_NORM)

def reconstruir_azimute(az_sin, az_cos, t, total_revolutions=1):
    """Reconstrói o ângulo contínuo usando o 't' como guia."""
    angulo_base = math.atan2(az_sin, az_cos)
    angulo_esperado = t * total_revolutions * 2 * math.pi
    num_voltas = round((angulo_esperado - angulo_base) / (2 * math.pi))
    angulo_reconstruido = angulo_base + num_voltas * 2 * math.pi
    return angulo_reconstruido

def gerar_trajetoria_original(tipo, num_passos):
    """Gera os pontos da trajetória original para comparação."""
    trajetoria = []
    if tipo == 'circular':
        for i in range(num_passos + 1):
            t = i / num_passos
            trajetoria.append({'azimuth': t * 2 * math.pi, 'radius': 1.0})
    elif tipo == 'espiral':
        radii_per_turn = [1.0, 0.79, 0.59, 0.39, 0.18, 0.0]
        for i in range(num_passos + 1):
            t = i / num_passos
            turn_index = min(math.floor(t * len(radii_per_turn)), len(radii_per_turn) - 1)
            trajetoria.append({'azimuth': t * 6 * 2 * math.pi, 'radius': radii_per_turn[turn_index]})
    elif tipo == 'lissajous': # Usando a Lissajous como terceira trajetória de referência
         for i in range(num_passos + 1):
            t = i / num_passos
            x = math.sin(t * 1 * 2 * math.pi + (math.pi/2))
            y = math.sin(t * 2 * 2 * math.pi)
            trajetoria.append({'azimuth': math.atan2(y, x), 'radius': math.sqrt(x**2 + y**2)})
    return trajetoria


def gerar_previsao_trajetoria_lstm(model, seed_sequence, traj_type, total_revolutions, num_passos):
    """Gera uma trajetória autorregressiva usando o modelo LSTM treinado."""
    model.eval()
    pontos_previstos = []
    
    # Transforma a semente em um tensor do PyTorch
    current_sequence = torch.tensor(seed_sequence, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        for i in range(num_passos):
            # Faz a previsão do próximo ponto
            next_point_norm = model(current_sequence).squeeze(0).numpy()
            
            # Decodifica o ponto previsto
            raio_norm, az_sin, az_cos, dur_norm = next_point_norm
            t = (len(seed_sequence) + i) / (len(seed_sequence) + num_passos -1) # Estima t
            
            raio_real = desnormalizar_raio(raio_norm)
            azimute_real = reconstruir_azimute(az_sin, az_cos, t, total_revolutions)
            
            pontos_previstos.append({'azimuth': azimute_real, 'radius': raio_real})
            
            # Cria o próximo passo da sequência de entrada
            next_input_step = np.array([t] + traj_type + list(next_point_norm))
            
            # Adiciona o novo passo e remove o mais antigo para a próxima iteração
            next_sequence_np = np.vstack([current_sequence.numpy().squeeze(0)[1:], next_input_step])
            current_sequence = torch.from_numpy(next_sequence_np).unsqueeze(0).float()
            
    return pontos_previstos


# --- 3  TESTE  ---
"""    Testa o modelo LSTM treinado, gera trajetórias e plota os resultados.
"""
if __name__ == "__main__":
    # Carrega o modelo treinado
    device = torch.device('cpu')
    model = LSTM_Model(INPUT_SIZE_PER_STEP, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, OUTPUT_SIZE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    print(f"Modelo '{MODEL_SAVE_PATH}' carregado com sucesso.")


    # Carrega o dataset original para pegar as sementes
    full_data = np.loadtxt('data/SpatTrajData.txt', delimiter=' ', converters={-1: lambda s: float(s.strip().replace(';',''))})
    
    #reorganiza os dados para facilitar a extração das sementes
    t_and_type = full_data[:, :4]
    coords_and_dur = full_data[:, 4:]
    full_features = np.hstack((t_and_type, coords_and_dur))

    print("Procurando sequências de semente no dataset...")
    # Procura a primeira sequência que começa com o tipo Lissajous [1,0,0]
    idx_lissajous = np.where((full_features[:, 1] == 1.0) & (full_features[:, 2] == 0.0) & (full_features[:, 3] == 0.0))[0][0]
    seed_lissajous = full_features[idx_lissajous : idx_lissajous + SEQUENCE_LENGTH]
    tipo_lissajous = [1.0, 0.0, 0.0]

    # Procura a primeira sequência que começa com o tipo Espiral [0,1,0]
    idx_espiral = np.where((full_features[:, 1] == 0.0) & (full_features[:, 2] == 1.0) & (full_features[:, 3] == 0.0))[0][0]
    seed_espiral = full_features[idx_espiral : idx_espiral + SEQUENCE_LENGTH]
    tipo_espiral = [0.0, 1.0, 0.0]
    
    # Procura a primeira sequência que começa com o tipo Círculo [0,0,1]
    idx_circulo = np.where((full_features[:, 1] == 0.0) & (full_features[:, 2] == 0.0) & (full_features[:, 3] == 1.0))[0][0]
    seed_circulo = full_features[idx_circulo : idx_circulo + SEQUENCE_LENGTH]
    tipo_circulo = [0.0, 0.0, 1.0]

    # Gerar previsões
    print("Gerando previsões com o modelo LSTM...")
    previsoes_lissajous = gerar_previsao_trajetoria_lstm(model, seed_lissajous, tipo_lissajous, total_revolutions=1, num_passos=NUM_PONTOS_GERACAO)
    previsoes_espiral = gerar_previsao_trajetoria_lstm(model, seed_espiral, tipo_espiral, total_revolutions=6, num_passos=NUM_PONTOS_GERACAO)
    previsoes_circulo = gerar_previsao_trajetoria_lstm(model, seed_circulo, tipo_circulo, total_revolutions=1, num_passos=NUM_PONTOS_GERACAO)

    # Gerar dados originais para comparação
    originais_lissajous = gerar_trajetoria_original('lissajous', NUM_PONTOS_GERACAO)
    originais_espiral = gerar_trajetoria_original('espiral', NUM_PONTOS_GERACAO)
    originais_circulo = gerar_trajetoria_original('circular', NUM_PONTOS_GERACAO)

    # Plotar e salvar cada trajetória
    fig, axs = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(18, 6))

    trajetorias = [
        ("Lissajous", originais_lissajous, previsoes_lissajous),
        ("Espiral", originais_espiral, previsoes_espiral),
        ("Circular", originais_circulo, previsoes_circulo)
    ]

    for ax, (nome, orig, prev) in zip(axs, trajetorias):
        ax.plot([p['azimuth'] for p in prev], [p['radius'] for p in prev], label='LSTM', color='red', linestyle='--')
        ax.set_title(nome)
        ax.legend(loc='upper right')

    plt.suptitle('Reconstrução das Trajetórias LSTM')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "trajetorias_originais_LSTM.png"), dpi=300)
    plt.close()
    print("Figura das três trajetórias lado a lado salva em 'plots/trajetorias_LSTM.png'.")