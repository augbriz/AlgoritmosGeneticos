import yfinance as yf
import nolds
import matplotlib.pyplot as plt
import numpy as np
import datetime
today = datetime.date.today()
print("Fecha actual:", today)
ticker = "YPF"
data = yf.download(ticker, start="2008-01-01", end=today)

if data.empty or 'Close' not in data.columns:
    raise ValueError("No se pudieron obtener datos de cierre para el ticker especificado.")

serie = data['Close'].dropna().values[-1000:].astype(float).flatten()
Lyapunov = nolds.lyap_r(serie, emb_dim=10, trajectory_len=20)
print("Exponente de Lyapunov:", Lyapunov)

#gráfico
plt.figure(figsize=(12, 6))
plt.plot(serie, linewidth=0.8, color='blue')
plt.title(f'Serie Temporal de Precios de Cierre - {ticker}\n(Últimos 1000 datos)', fontsize=14)
plt.xlabel('Tiempo (días)', fontsize=12)
plt.ylabel('Precio de Cierre (USD)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Agregar información del exponente de Lyapunov en el gráfico
plt.text(0.02, 0.98, f'Exponente de Lyapunov: {Lyapunov:.6f}', 
         transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.show()

