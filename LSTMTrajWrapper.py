import torch
import torch.nn as nn
import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import math

# Importa a definição do modelo
from LSTMTrajTrain import LSTM_Model, INPUT_SIZE_PER_STEP, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, OUTPUT_SIZE, SEQUENCE_LENGTH, MODEL_SAVE_PATH

# diretório para salvar plots de teste
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# Parâmetros de normalização para o teste
MIN_DUR_NORM = 100.0
MAX_DUR_NORM = 1000.0
MIN_RADIUS_NORM = 0.0
MAX_RADIUS_NORM = 1.0


# --- 1 WRAPPER STATELESS COMPATÍVEL COM TORCH.TS ---
"""
    Wrapper para o modelo LSTM treinado compatível com torch.ts
    Permite processar um passo de tempo por vez, mantendo o estado interno.
    Exporta o modelo como um módulo TorchScript para fácil integração.
"""
class LSTM_SpatWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        self.INPUT_SIZE_PER_STEP = INPUT_SIZE_PER_STEP

        # REGISTRO DE MÉTODOS E ATRIBUTOS (compatível com TorchScript)
        self._methods = ["forward"]
        self._attributes = ["forward_input_shape", "forward_output_shape"]

        # SHAPES DE ENTRADA E SAÍDA
        self.forward_input_shape = [self.SEQUENCE_LENGTH*self.INPUT_SIZE_PER_STEP]  # [t, traj_type_1, traj_type_2, traj_type_3, radius_norm, az_sin, az_cos, dur_norm]
        self.forward_output_shape = [4]  # retorna [radius_norm, az_sin, az_cos, dur_norm]


    @torch.jit.export
    def get_methods(self) -> List[str]:
        """Retorna lista de métodos disponíveis"""
        return self._methods

    @torch.jit.export
    def get_attributes(self) -> List[str]:
        """Retorna lista de atributos disponíveis"""
        return self._attributes

    @torch.jit.export
    def forward(self, input_step: torch.Tensor) -> torch.Tensor:
        """
        Processa uma sequência de passos [1, seqlength, num_features] e retorna r, sin, cos, dur [4]
        seqência de passos - t, traj_type1, traj_type2, traj_type3, r, sin, cos, dur [seqlength, num_features] -> Saída: r, sin, cos, dur [num_outputs]
        Estados internos são inicializados com zeros a cada chamada do forward.
        """
        input_step = input_step.reshape(1, self.SEQUENCE_LENGTH, self.INPUT_SIZE_PER_STEP)  # [1, seqlength, num_features]
        return self.model(input_step).squeeze(0).squeeze(0)  # remove as dimensões extras do batch e sequência
    

# --- Funções auxiliares  ---
def desnormalizar_raio(raio_norm):
    return MIN_RADIUS_NORM + ((raio_norm + 1) / 2) * (MAX_RADIUS_NORM - MIN_RADIUS_NORM)

def reconstruir_azimute(az_sin, az_cos, t, total_revolutions=1):
    angulo_base = math.atan2(az_sin, az_cos)
    angulo_esperado = t * total_revolutions * 2 * math.pi
    num_voltas = round((angulo_esperado - angulo_base) / (2 * math.pi))
    return angulo_base + num_voltas * 2 * math.pi

def gerar_trajetoria_original(tipo, num_passos):
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
    elif tipo == 'lissajous':
         for i in range(num_passos + 1):
            t = i / num_passos
            x = math.sin(t * 1 * 2 * math.pi + (math.pi/2))
            y = math.sin(t * 2 * 2 * math.pi)
            trajetoria.append({'azimuth': math.atan2(y, x), 'radius': math.sqrt(x**2 + y**2)})
    return trajetoria


# teste do wrapper exportado
def gerar_previsao_trajetoria_wrapper(model, seed_sequence, traj_type, total_revolutions, num_passos):
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

if __name__ == "__main__":
    
    # --- EXPORTAÇÃO ---
    wrapper_dir = os.path.dirname(os.path.abspath(__file__))
    ts_dir = os.path.join(wrapper_dir, "torchscript")
    os.makedirs(ts_dir, exist_ok=True)
    
    trained_model = LSTM_Model(INPUT_SIZE_PER_STEP, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, OUTPUT_SIZE)
    trained_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu'))
    trained_model.eval()

    wrapper = LSTM_SpatWrapper(trained_model)
    scripted_wrapper = torch.jit.script(wrapper)
    output_path = os.path.join(ts_dir, "LSTMSpatTraj.ts")
    scripted_wrapper.save(output_path)
    print(f"Wrapper LSTM exportado com sucesso para: {output_path}")

    # --- Teste do wrapper exportado  ---
    print("\nTestando o wrapper exportado de forma autorregressiva para as 3 trajetórias...")

    loaded_model = torch.jit.load(output_path)
    loaded_model.eval()

    # Carrega o dataset original para pegar as sementes
    full_data = np.loadtxt('data/SpatTrajData.txt', delimiter=' ', converters={-1: lambda s: float(s.strip().replace(';',''))})
    
    # Reorganiza os dados
    t_and_type = full_data[:, :4]
    coords_and_dur = full_data[:, 4:]
    full_features = np.hstack((t_and_type, coords_and_dur))
    
    # Encontra as sementes
    idx_lissajous = np.where((full_features[:, 1] == 1.0) & (full_features[:, 2] == 0.0) & (full_features[:, 3] == 0.0))[0][0]
    seed_lissajous = full_features[idx_lissajous : idx_lissajous + SEQUENCE_LENGTH]
    tipo_lissajous = [1.0, 0.0, 0.0]

    idx_espiral = np.where((full_features[:, 1] == 0.0) & (full_features[:, 2] == 1.0) & (full_features[:, 3] == 0.0))[0][0]
    seed_espiral = full_features[idx_espiral : idx_espiral + SEQUENCE_LENGTH]
    tipo_espiral = [0.0, 1.0, 0.0]

    idx_circulo = np.where((full_features[:, 1] == 0.0) & (full_features[:, 2] == 0.0) & (full_features[:, 3] == 1.0))[0][0]
    seed_circulo = full_features[idx_circulo : idx_circulo + SEQUENCE_LENGTH]
    tipo_circulo = [0.0, 0.0, 1.0]

    # busca a primeira sequência híbrida [0.5, 0, 0.5]
    idx_hibrida = np.where((full_features[:, 1] == 0.5) & (full_features[:, 2] == 0.5) & (full_features[:, 3] == 0.0))[0][0]
    seed_hibrida = full_features[idx_hibrida : idx_hibrida + SEQUENCE_LENGTH]
    tipo_hibrida = [0.5, 0.5, 0.0]

    # Gera previsões 
    NUM_PONTOS_GERACAO = 500
    previsoes_lissajous = gerar_previsao_trajetoria_wrapper(loaded_model, seed_lissajous, tipo_lissajous, total_revolutions=1, num_passos=NUM_PONTOS_GERACAO)
    previsoes_espiral = gerar_previsao_trajetoria_wrapper(loaded_model, seed_espiral, tipo_espiral, total_revolutions=6, num_passos=NUM_PONTOS_GERACAO)
    previsoes_circulo = gerar_previsao_trajetoria_wrapper(loaded_model, seed_circulo, tipo_circulo, total_revolutions=1, num_passos=NUM_PONTOS_GERACAO)
    previsoes_hibrida = gerar_previsao_trajetoria_wrapper(loaded_model, seed_hibrida, tipo_hibrida, total_revolutions=6, num_passos=NUM_PONTOS_GERACAO)

    # Gera dados originais para comparação
    originais_lissajous = gerar_trajetoria_original('lissajous', NUM_PONTOS_GERACAO)
    originais_espiral = gerar_trajetoria_original('espiral', NUM_PONTOS_GERACAO)
    originais_circulo = gerar_trajetoria_original('circular', NUM_PONTOS_GERACAO)

    # Plota e salva cada trajetória
    fig, axs = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(18, 6))

    trajetorias = [
        ("Lissajous", originais_lissajous, previsoes_lissajous),
        ("Espiral", originais_espiral, previsoes_espiral),
        ("Circular", originais_circulo, previsoes_circulo)
    ]

    for ax, (nome, orig, prev) in zip(axs, trajetorias):
        ax.plot([p['azimuth'] for p in prev], [p['radius'] for p in prev], label='LSTM (Wrapper)', color='red', linestyle='--')
        ax.set_title(nome)
        ax.legend(loc='upper right')

    plt.suptitle('Reconstrução das Trajetórias (LSTM Wrapper)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "trajetorias_originais_LSTM_wrapper.png"), dpi=300)
    plt.close()
    print("Figura corrigida das três trajetórias salva em 'plots/trajetorias_originais_LSTM_wrapper.png'.")

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    ax.plot([p['azimuth'] for p in previsoes_hibrida], [p['radius'] for p in previsoes_hibrida], label='LSTM (Wrapper)', color='red', linestyle='--')
    ax.set_title('Reconstrução da Trajetória Híbrida')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "trajetoria_hibrida_LSTM_wrapper.png"), dpi=300)
    plt.close()
    print("Figura da trajetória híbrida salva em 'plots/trajetoria_hibrida_lstm_wrapper.png'.")
