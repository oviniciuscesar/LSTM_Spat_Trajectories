import numpy as np
import math
import torch
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt

# --- 1 CONFIGURAÇÃO ---

SEED = 42  # Escolha qualquer valor inteiro
np.random.seed(SEED)
torch.manual_seed(SEED)

# diretório atual
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
plot_dir = os.path.join(script_dir, 'plots')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Parâmetros do Dataset
NUM_PASSOS = 500  # Aumenta a densidade
ADICIONAR_RUIDO = False # Adiciona ruído pequeno aos labels
ARQUIVO_SAIDA = os.path.join(data_dir, 'SpatTrajData.txt')

# Parâmetros de Normalização
MIN_RADIUS_NORM = 0.0
MAX_RADIUS_NORM = 1.0
MIN_DUR_NORM = 100.0
MAX_DUR_NORM = 1000.0


# --- 2 FUNÇÕES DE GERAÇÃO DAS TRAJETÓRIAS "PURAS" ---
"""
    Funções para converter entre coordenadas polares e cartesianas,
    gerar trajetórias puras (circular, espiral, lissajous) e criar híbridos.
"""

# Funções de Conversão
def polar_to_cartesian(radius, azimuth):
    x = radius * np.cos(azimuth)
    y = radius * np.sin(azimuth)
    return x, y

def cartesian_to_polar(x, y):
    radius = np.sqrt(x**2 + y**2)
    azimuth = np.arctan2(y, x)
    return radius, azimuth

# Geração de trajetória circular
def gerar_circular(num_passos):
    pontos = []
    for i in range(num_passos + 1):
        t = i / num_passos
        ponto = {
            't': t, 'radius': 1.0, 'azimuth': t * 2 * np.pi,
            'duration': 100.0 * (100.0 / 100.0)**t
        }
        pontos.append(ponto)
    return pontos

# Geração de trajetória espiral
def gerar_espiral_etapas(num_passos):
    pontos = []
    radii = [1.0, 0.79, 0.59, 0.39, 0.18, 0.0]
    durs = [100.0, 935.4, 583.4, 363.8, 226.9, 200.0]
    num_turns = len(radii)
    for i in range(num_passos + 1):
        t = i / num_passos
        turn_index = min(math.floor(t * num_turns), num_turns - 1)
        ponto = {
            't': t, 'radius': radii[turn_index], 'azimuth': t * 6 * 2 * np.pi,
            'duration': durs[turn_index]
        }
        pontos.append(ponto)
    return pontos

#Geração de trajetória Lissajous
def gerar_lissajous(num_passos):
    pontos = []
    a, b = 1, 2
    delta = np.pi / 2  # Para começar em azimuth=1.57
    t0 = 0
    t1 = 2 * np.pi  # ciclo completo
    for i in range(num_passos + 1):
        t = t0 + (t1 - t0) * (i / num_passos)
        x = np.sin(a * t + delta)
        y = np.sin(b * t)
        radius, azimuth = cartesian_to_polar(x, y)
        r_norm = (radius - MIN_RADIUS_NORM) / (MAX_RADIUS_NORM - MIN_RADIUS_NORM)
        duration = MIN_DUR_NORM * (MAX_DUR_NORM / MIN_DUR_NORM)**r_norm
        ponto = {'t': (t - t0) / (t1 - t0), 'radius': radius, 'azimuth': azimuth, 'duration': duration}
        pontos.append(ponto)
    return pontos


# --- 3 FUNÇÃO PARA CRIAR TRAJETÓRIAS HÍBRIDAS ---
"""
    Cria uma trajetória híbrida a partir de duas trajetórias existentes.
"""

def criar_hibrido(traj1_pontos, traj2_pontos, peso1=0.5, peso2=0.5):
    pontos_hibridos = []
    for p1, p2 in zip(traj1_pontos, traj2_pontos):
        x1, y1 = polar_to_cartesian(p1['radius'], p1['azimuth'])
        x2, y2 = polar_to_cartesian(p2['radius'], p2['azimuth'])

        # Interpolação linear em coordenadas cartesianas
        x_h = x1 * peso1 + x2 * peso2
        y_h = y1 * peso1 + y2 * peso2
        
        radius_h, azimuth_h = cartesian_to_polar(x_h, y_h)
        
        # Interpolação linear da duração
        duration_h = p1['duration'] * peso1 + p2['duration'] * peso2
        
        ponto = {'t': p1['t'], 'radius': radius_h, 'azimuth': azimuth_h, 'duration': duration_h}
        pontos_hibridos.append(ponto)
    return pontos_hibridos


# --- 4 FUNÇÃO PRINCIPAL ---
def main():
    print("Gerando trajetórias puras...")
    traj_circulo = gerar_circular(NUM_PASSOS)
    traj_espiral = gerar_espiral_etapas(NUM_PASSOS)
    traj_lissajous = gerar_lissajous(NUM_PASSOS)
    
    print("Gerando trajetórias híbridas...")
    traj_circ_esp = criar_hibrido(traj_circulo, traj_espiral)
    traj_circ_liss = criar_hibrido(traj_circulo, traj_lissajous)
    traj_esp_liss = criar_hibrido(traj_espiral, traj_lissajous)

    trajetorias = {
        'circulo': (traj_circulo, [0.0, 0.0, 1.0]),
        'lissajous': (traj_lissajous, [1.0, 0.0, 0.0]),
        'espiral': (traj_espiral, [0.0, 1.0, 0.0]),
        'circ_esp': (traj_circ_esp, [0.0, 0.5, 0.5]),
        'circ_liss': (traj_circ_liss, [0.5, 0.0, 0.5]),
        'esp_liss': (traj_esp_liss, [0.5, 0.5, 0.0])
    }
    
    dataset_final = []
    
    print("Montando e normalizando o dataset final...")
    for nome, (pontos, traj_type) in trajetorias.items():
        for p in pontos:
            # Adiciona ruído opcional aos labels
            radius = p['radius'] + (np.random.uniform(-0.01, 0.01) if ADICIONAR_RUIDO else 0)
            azimuth = p['azimuth'] + (np.random.uniform(-0.02, 0.02) if ADICIONAR_RUIDO else 0)
            duration = p['duration']
            
            # Normalização e codificação
            t = p['t']
            radius_norm = -1.0 + 2.0 * (np.clip(radius, MIN_RADIUS_NORM, MAX_RADIUS_NORM) - MIN_RADIUS_NORM) / (MAX_RADIUS_NORM - MIN_RADIUS_NORM)
            azimuth_sin = np.sin(azimuth)
            azimuth_cos = np.cos(azimuth)
            duration_norm = -1.0 + 2.0 * (np.clip(duration, MIN_DUR_NORM, MAX_DUR_NORM) - MIN_DUR_NORM) / (MAX_DUR_NORM - MIN_DUR_NORM)
            
            # Formata a linha como [t, type1, type2, type3, r_norm, sin, cos, dur_norm]
            linha = [t] + traj_type + [radius_norm, azimuth_sin, azimuth_cos, duration_norm]
            dataset_final.append(linha)
            
    # Salva o arquivo final
    np.savetxt(ARQUIVO_SAIDA, np.array(dataset_final), fmt='%.6f', delimiter=' ', newline=';\n')
    print(f"\nDataset melhorado com {len(dataset_final)} exemplos foi salvo em '{ARQUIVO_SAIDA}'")

    print("Plotando trajetórias originais...")
    fig1, axs1 = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(18, 6))
    originais = ['circulo', 'lissajous', 'espiral']
    for ax, nome in zip(axs1, originais):
        pontos, traj_type = trajetorias[nome]
        ax.plot([p['azimuth'] for p in pontos], [p['radius'] for p in pontos], label=nome)
        ax.set_title(f'Trajetória: {nome}')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "trajetorias_originais.png"), dpi=300)
    plt.close()
    print(f"Figura das trajetórias originais salva em '{plot_dir}/trajetorias_originais.png'")

    # Plota trajetórias híbridas (circ_esp, circ_liss, esp_liss)
    print("Plotando trajetórias híbridas...")
    fig2, axs2 = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(18, 6))
    hibridas = ['circ_esp', 'circ_liss', 'esp_liss']
    for ax, nome in zip(axs2, hibridas):
        pontos, traj_type = trajetorias[nome]
        ax.plot([p['azimuth'] for p in pontos], [p['radius'] for p in pontos], label=nome)
        ax.set_title(f'Trajetória: {nome}')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "trajetorias_hibridas.png"), dpi=300)
    plt.close()
    print(f"Figura das trajetórias híbridas salva em '{plot_dir}/trajetorias_hibridas.png'")

if __name__ == '__main__':
    main()
