import matplotlib.pyplot as plt
import numpy as np

# 1. Definição dos dados coletados no seu ambiente (Média de 5 execuções)
labels = ['Não-Vetorizada (Iago)', 'Vetorizada (Thiago)']
tempos_medios = [11.6656, 0.1300]
desvios = [0.1759, 0.0164] 

# Configuração estética dos gráficos
plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 2. Gráfico de Barras: Comparação de Tempo de Execução
bars = ax1.bar(labels, tempos_medios, yerr=desvios, color=['#e74c3c', '#2ecc71'], capsize=10)
ax1.set_ylabel('Tempo Médio de Treinamento (segundos)')
ax1.set_title('Comparativo de Tempo: Iterativo vs Vetorizado')

# Adicionando rótulos de dados nas barras
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height:.4f}s',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), 
                textcoords="offset points",
                ha='center', va='bottom', fontweight='bold')

# 3. Gráfico de Speedup (Representação do Ganho)
speedup = tempos_medios[0] / tempos_medios[1]
ax2.bar(['Speedup Alcançado'], [speedup], color='#3498db')
ax2.set_title('Fator de Aceleração (Speedup)')
ax2.set_ylabel('Vezes mais rápido')
ax2.annotate(f'{speedup:.2f}x', 
             xy=(0, speedup), 
             xytext=(0, 3),
             textcoords="offset points",
             ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('graficos_resultados.png') # Salva a imagem para o relatório
plt.show()

print(f"Análise Final: A versão vetorizada é {speedup:.2f} vezes mais rápida.")