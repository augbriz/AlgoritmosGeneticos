import yfinance as yf
import nolds
ticker = "YPF"
data = yf.download(ticker, start="2020-01-01", end="2025-07-23")

if data.empty or 'Close' not in data.columns:
    raise ValueError("No se pudieron obtener datos de cierre para el ticker especificado.")


serie = data['Close'].dropna().values[-1000:].astype(float).flatten()
Lyapunov = nolds.lyap_r(serie, emb_dim=10, trajectory_len=20)
print("Exponente de Lyapunov:", Lyapunov)
