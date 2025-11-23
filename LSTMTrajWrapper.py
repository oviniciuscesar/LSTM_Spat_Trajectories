import torch
import torch.nn as nn
import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import math

# Importa a definição do modelo
from LSTMTrajTrain import LSTM_Model, INPUT_SIZE_PER_STEP, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, OUTPUT_SIZE, SEQUENCE_LENGTH, MODEL_SAVE_PATH

SEED = 42  # Escolha qualquer valor inteiro
np.random.seed(SEED)
torch.manual_seed(SEED)


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
        self._methods = ["forward", "mix", "forward_mix", "get_mix"]
        self._attributes = ["forward_input_shape", "forward_output_shape", "forward_mix_input_shape", "forward_mix_output_shape", "mix"]
        
        # SHAPES DE ENTRADA E SAÍDA
        self.forward_input_shape = [self.SEQUENCE_LENGTH*self.INPUT_SIZE_PER_STEP]  # [t, traj_type_1, traj_type_2, traj_type_3, radius_norm, az_sin, az_cos, dur_norm]
        self.forward_output_shape = [4]  # retorna [radius_norm, az_sin, az_cos, dur_norm]

        self.forward_mix_input_shape = [self.SEQUENCE_LENGTH*self.INPUT_SIZE_PER_STEP]
        self.forward_mix_output_shape = [4]  # retorna [radius_norm, az_sin, az_cos, dur_norm]
        self.mix = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)


    @torch.jit.export
    def get_methods(self) -> List[str]:
        """Retorna lista de métodos disponíveis"""
        return self._methods

    @torch.jit.export
    def get_attributes(self) -> List[str]:
        """Retorna lista de atributos disponíveis"""
        return self._attributes
    
    @torch.jit.export
    def set_mix(self, w: torch.Tensor) -> None:
        """Aceita Tensor com 3 elementos do PD."""
        if w.dtype not in (torch.float32, torch.float64):
            w = w.to(torch.float32)
        w = w.view(-1)
        if w.numel() != 3:
            raise RuntimeError("set_mix_tensor: esperado tensor com 3 elementos.")
        w = torch.clamp(w, min=1e-8)
        self.mix = (w / w.sum()).to(torch.float32)

    @torch.jit.export
    def get_mix(self) -> List[float]:
        """Retorna mistura atual (normalizada)."""
        return [float(self.mix[0]), float(self.mix[1]), float(self.mix[2])]

    @torch.jit.export
    def forward(self, input_flat: torch.Tensor) -> torch.Tensor:
        """
        Recebe sequência achatada [SEQUENCE_LENGTH*INPUT_SIZE_PER_STEP]
        Retorna próximo ponto [4]: (radius_norm, az_sin, az_cos, dur_norm)
        Usa interpolação latente via self.mix.
        """
        # faz reshape da entrada achatada
        x = input_flat.reshape(1, self.SEQUENCE_LENGTH, self.INPUT_SIZE_PER_STEP)
        y = self.model.forward(x)  # (1,4)
        return y.squeeze(0)
    

    @torch.jit.export
    def forward_mix(self, input_flat: torch.Tensor) -> torch.Tensor:
        """
        Recebe sequência achatada [SEQUENCE_LENGTH*INPUT_SIZE_PER_STEP]
        Retorna próximo ponto [4]: (radius_norm, az_sin, az_cos, dur_norm)
        Usa interpolação latente via mix.
        mix: (3,) com pesos que somam ~1.0 (convexo). Ex.: [alpha, 1-alpha, 0] (interp entre tipo0 e tipo1)
        """
        # faz reshape da entrada achatada
        x = input_flat.reshape(1, self.SEQUENCE_LENGTH, self.INPUT_SIZE_PER_STEP)
        # passa pela forward com mix passado pelo método set_mix
        y = self.model.forward_mix(x, self.mix)  # (B,4)
        return y.squeeze(0)
    

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
def gerar_previsao_trajetoria_wrapper(model, seed_sequence, mix_weights, total_revolutions, num_passos):
    """Gera uma trajetória autorregressiva usando o modelo LSTM treinado."""
    model.eval()
    pontos_previstos = []
    
    # Transforma a semente em um tensor do PyTorch
    current_sequence = torch.tensor(seed_sequence, dtype=torch.float32).unsqueeze(0)


    mix = torch.tensor(mix_weights, dtype=torch.float32)
    mix = torch.clamp(mix, min=1e-8)
    mix = mix / mix.sum()

    # define a mistura no modelo
    model.set_mix(mix)
    
    with torch.no_grad():
        for i in range(num_passos):
            # achata a sequência atual
            input_flat = current_sequence.view(-1)
            # Faz a previsão do próximo ponto
            next_point_norm = model(input_flat).numpy()
            
            # Decodifica o ponto previsto
            raio_norm, az_sin, az_cos, dur_norm = next_point_norm
            t = (len(seed_sequence) + i) / (len(seed_sequence) + num_passos -1) # Estima t
            
            raio_real = desnormalizar_raio(raio_norm)
            azimute_real = reconstruir_azimute(az_sin, az_cos, t, total_revolutions)
            
            pontos_previstos.append({'azimuth': azimute_real, 'radius': raio_real})

            # Monta próximo passo: [t, mix0, mix1, mix2, r_norm, sin, cos, dur_norm]
            next_input_step = np.array([t, mix[0].item(), mix[1].item(), mix[2].item(),
                                        raio_norm, az_sin, az_cos, dur_norm], dtype=np.float32)
             
            # Adiciona o novo passo e remove o mais antigo para a próxima iteração
            next_sequence_np = np.vstack([current_sequence.numpy().squeeze(0)[1:], next_input_step])
            current_sequence = torch.from_numpy(next_sequence_np).unsqueeze(0).float()
            
    return pontos_previstos


def testar_interpolacao(loaded_model, seed_sequence, w_from, w_to, total_revolutions, num_passos, n_alphas=5, save_path="plots/interpolacao.png", titulo="Interpolação Latente"):
    """
    Faz inferência autorregressiva variando alpha entre w_from e w_to e plota as curvas em um gráfico polar.
    w_from, w_to: listas de 3 pesos (ex.: [1,0,0] e [0,0,1]).
    """
    alphas = np.linspace(0.0, 1.0, n_alphas)
    cmap = plt.get_cmap("viridis")
    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, projection='polar')
    for i, a in enumerate(alphas):
        mix = (1.0 - a) * np.array(w_from, dtype=np.float32) + a * np.array(w_to, dtype=np.float32)
        previsoes = gerar_previsao_trajetoria_wrapper(loaded_model, seed_sequence, mix, total_revolutions=total_revolutions, num_passos=num_passos)
        color = cmap(i / max(1, n_alphas - 1))
        ax.plot([p['azimuth'] for p in previsoes], [p['radius'] for p in previsoes],
                label=f"α={a:.2f} | mix={mix.round(2)}", color=color, linewidth=1.6)
    ax.set_title(titulo)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figura de interpolação salva em '{save_path}'.")



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
    tipo_circulo = [0.460652, 0.201567, 0.694444] # 0.0160652 0.201567 0.694444

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

    # --- Teste de interpolação: Lissajous (1,0,0) -> Circular (0,0,1) usando a semente Lissajous ---
    testar_interpolacao(
        loaded_model,
        seed_sequence=seed_lissajous,
        w_from=[1.0, 0.0, 0.0],
        w_to=[0.0, 0.0, 1.0],
        total_revolutions=1,
        num_passos=NUM_PONTOS_GERACAO,
        n_alphas=6,
        save_path=os.path.join(plots_dir, "interpolacao_lissajous_to_circular.png"),
        titulo="Interpolação Latente: Lissajous → Circular"
    )

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
        ax.plot([p['azimuth'] for p in prev], [p['radius'] for p in prev], label='LSTM', color='red', linestyle='--')
        ax.set_title(nome)
        ax.legend(loc='upper right')

    fig.subplots_adjust(top=0.88, wspace=0.25)
    plt.suptitle('Reconstrução das Trajetórias (LSTM)', fontsize=16)
    # plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "recon_orig_traj_wrapper.png"), dpi=300)
    plt.close()
    print("Figura corrigida das três trajetórias salva em 'plots/recon_orig_traj_wrapper.png'.")

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    ax.plot([p['azimuth'] for p in previsoes_hibrida], [p['radius'] for p in previsoes_hibrida], label='LSTM', color='red', linestyle='--')
    ax.set_title('Reconstrução da Trajetória Híbrida')
    ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(plots_dir, "recon_hibrid_traj_wrapper.png"), dpi=300)
    plt.close()
    print("Figura da trajetória híbrida salva em 'plots/recon_hibrid_traj_wrapper.png'.")