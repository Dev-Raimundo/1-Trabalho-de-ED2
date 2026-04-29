import time
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def test_performance(m, nx=10):
    X = np.random.randn(nx, m)
    Y = np.random.randint(0, 2, (1, m))
    w = np.zeros((nx, 1))
    b = 0
    
    # Versão Iterativa
    start = time.time()
    Z = np.zeros((1, m))
    for i in range(m):
        Z[0, i] = np.dot(w.T, X[:, i]).item() + b
    A = sigmoid(Z)
    t_iter = time.time() - start
    
    # Versão Vetorizada
    start = time.time()
    A_vec = sigmoid(np.dot(w.T, X) + b)
    t_vec = time.time() - start
    
    return t_iter, t_vec

amostras = [100, 1000, 10000, 100000, 500000]
speedups = []

for m in amostras:
    ti, tv = test_performance(m)
    speedups.append(ti / tv)

plt.figure(figsize=(8, 5))
plt.plot(amostras, speedups, marker='o', color='g', linestyle='--')
plt.xscale('log') # Escala logarítmica para ver melhor o crescimento
plt.title("Evolução do Speedup vs. Volume de Dados (m)")
plt.xlabel("Número de Amostras (m)")
plt.ylabel("Fator de Aceleração (Speedup)")
plt.grid(True)
plt.savefig("evolucao_speedup.png")
plt.show()