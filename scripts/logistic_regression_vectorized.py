import numpy as np
import os
import time

def carregar_dados():
    path = 'data/'
    # Criar arquivos dummy para demonstração se não existirem
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(os.path.join(path, 'X_train.npy')) or not os.path.exists(os.path.join(path, 'y_train.npy')):
        print("Criando dados de exemplo para demonstração...")
        np.random.seed(42)
        X_train_dummy = np.random.rand(100, 5) * 10 # 100 amostras, 5 features
        y_train_dummy = (np.random.rand(100) > 0.5).astype(int) # Binário
        np.save(os.path.join(path, 'X_train.npy'), X_train_dummy)
        np.save(os.path.join(path, 'y_train.npy'), y_train_dummy)
        print("Dados de exemplo criados.")

    X_train = np.load(os.path.join(path, 'X_train.npy'))
    y_train = np.load(os.path.join(path, 'y_train.npy'))
    return X_train, y_train

def sigmoid(z):
    # np.clip para evitar overflow/underflow com np.exp em valores muito grandes/pequenos
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def treinar_modelo_vectorizado(X, y, lr, epochs):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    print(f"Iniciando treino vetorizado: {epochs} iterações...")
    inicio = time.perf_counter()

    for epoch in range(epochs):
        # 1. Calcular z para todas as amostras de uma vez
        z = np.dot(X, w) + b

        # 2. Aplicar a função sigmoide para obter as previsões (a)
        a = sigmoid(z)

        # 3. Calcular o erro para todas as amostras
        erro = a - y

        # 4. Calcular o gradiente de w (dw) e b (db) de forma vetorizada
        dw = np.dot(X.T, erro) / m
        db = np.sum(erro) / m

        # 5. Atualizar os pesos (w) e o bias (b)
        w -= lr * dw
        b -= lr * db

        # Calcular o custo para monitoramento (vetorizado)
        # Adicionar um pequeno valor (1e-15) para evitar np.log(0)
        custo = -np.mean(y * np.log(a + 1e-15) + (1 - y) * np.log(1 - a + 1e-15))

        if epoch % 100 == 0:
            print(f"Época {epoch} | Custo: {custo:.6f}")

    fim = time.perf_counter()
    return w, b, fim - inicio

# Demonstração do uso da função vetorizada
if __name__ == '__main__':
    X, y = carregar_dados()

    # Chamar a versão vetorizada
    pesos_vec, bias_vec, duracao_vec = treinar_modelo_vectorizado(X, y, 0.01, 1000)

    print(f"\n Treinamento Vetorizado concluído em {duracao_vec:.4f}s")
    print(f"Pesos finais (vetorizado): {pesos_vec}")
    print(f"Bias final (vetorizado): {bias_vec}")

""" Principais mudanças na versão vetorizada:

1.  **Cálculo de `z`**: Em vez de um loop para cada amostra e suas características, `np.dot(X, w) + b` calcula `z` para todas as amostras de uma vez.
2.  **Cálculo de `a` (previsões)**: `sigmoid(z)` é aplicado a todo o vetor `z` simultaneamente.
3.  **Cálculo do `erro`**: `a - y` subtrai os vetores de previsões e valores reais.
4.  **Gradientes (`dw`, `db`)**:
    *   `dw = np.dot(X.T, erro) / m` calcula o gradiente de `w` usando a transposta de `X` e o vetor de erros.
    *   `db = np.sum(erro) / m` calcula o gradiente de `b` somando todos os erros.
5.  **Atualização de `w` e `b`**: As atualizações são feitas em vetores inteiros (`w -= lr * dw`, `b -= lr * db`).
6.  **Cálculo do Custo**: Utiliza `np.mean` para calcular a média do custo sobre todas as amostras de forma vetorizada. """
