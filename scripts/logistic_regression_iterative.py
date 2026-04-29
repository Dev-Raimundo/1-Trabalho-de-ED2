import numpy as np
import os
import time

def carregar_dados():
    path = 'data/'
    X_train = np.load(os.path.join(path, 'X_train.npy'))
    y_train = np.load(os.path.join(path, 'y_train.npy'))
    return X_train, y_train

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def treinar_modelo(X, y, lr, epochs):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    
    print(f"Iniciando treino: {epochs} iterações...")
    inicio = time.perf_counter()

    for epoch in range(epochs):
        custo_total = 0
        dw = np.zeros(n)
        db = 0.0

        for i in range(m):
            z_i = 0.0
            for j in range(n):
                z_i += w[j] * X[i][j]
            z_i += b
            
            a_i = sigmoid(z_i)
            erro = a_i - y[i]
            
            custo_total += -(y[i] * np.log(a_i + 1e-15) + (1 - y[i]) * np.log(1 - a_i + 1e-15))
            
            for j in range(n):
                dw[j] += erro * X[i][j]
            db += erro

        for j in range(n):
            w[j] -= lr * (dw[j] / m)
        b -= lr * (db / m)

        if epoch % 100 == 0:
            print(f"Época {epoch} | Custo: {custo_total/m:.6f}")

    fim = time.perf_counter()
    return w, b, fim - inicio

if __name__ == "__main__":
    X, y = carregar_dados()
    pesos, bias, duracao = treinar_modelo(X, y, 0.01, 1000)
    
    print(f"\n✅ Treinamento concluído em {duracao:.4f}s")
    print(f"Pesos finais: {pesos}")
    print(f"Bias final: {bias}")