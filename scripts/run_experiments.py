import numpy as np
import time
from logistic_regression_iterative import treinar_modelo as treino_iago
from logistic_regression_vectorized import treinar_modelo_vectorizado as treino_thiago
import os

# Carregar dados preparados pelo Raimundo
path = 'data/'
X_train = np.load(os.path.join(path, 'X_train.npy'))
y_train = np.load(os.path.join(path, 'y_train.npy'))

def executar_testes(repeticoes=5):
    tempos_iago = []
    tempos_thiago = []
    
    print(f"--- Iniciando {repeticoes} repetições para comparação ---")
    
    for i in range(repeticoes):
        # Teste Iago (Não-Vetorizado)
        _, _, t_i = treino_iago(X_train, y_train, 0.01, 1000)
        tempos_iago.append(t_i)
        
        # Teste Thiago (Vetorizado)
        _, _, t_t = treino_thiago(X_train, y_train, 0.01, 1000)
        tempos_thiago.append(t_t)
        
    media_iago = np.mean(tempos_iago)
    media_thiago = np.mean(tempos_thiago)
    speedup = media_iago / media_thiago
    desvio_iago = np.std(tempos_iago)
    desvio_thiago = np.std(tempos_thiago)

    print("\n" + "="*30)
    print(f"RESULTADOS FINAIS (Média de {repeticoes} execuções):")
    print(f"Tempo Médio Iago (Não-Vetorizado): {media_iago:.4f}s")
    print(f"Tempo Médio Thiago (Vetorizado): {media_thiago:.4f}s")
    print(f"Desvio Padrão Iago: {desvio_iago:.4f}")
    print(f"Desvio Padrão Thiago: {desvio_thiago:.4f}")
    print(f"SPEEDUP: {speedup:.2f}x mais rápido")
    print("="*30)

if __name__ == "__main__":
    executar_testes()