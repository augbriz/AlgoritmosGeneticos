import numpy as np                
import pandas as pd               
import matplotlib.pyplot as plt   
import warnings
warnings.filterwarnings('ignore') 
# importación de los datos 
import yfinance as yf            
# Framework deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Preprocesamiento de los datos
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Configuración de TensorFlow ---
print("TensorFlow version:", tf.__version__)
print("GPU disponible:", tf.config.list_physical_devices('GPU'))

# Configurar reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

print("✅ Importaciones completadas exitosamente")
print("🎯 Preparado para implementar modelo LSTM")

# =============================================================================
# CONSTANTES DEL PROYECTO
# =============================================================================

# Configuración de datos
TICKER = "YPF"
START_DATE = "2008-01-01"
END_DATE = "2025-07-23"
NUM_DATOS = 1000  # Últimos 1000 datos (como en análisis caótico)

# Configuración del modelo (valores base - se optimizarán con AG)
LOOKBACK_WINDOW = 20      # Ventana temporal: últimos 20 días para predecir
TRAIN_SPLIT = 0.7         # 70% para entrenamiento
VAL_SPLIT = 0.15          # 15% para validación
TEST_SPLIT = 0.15         # 15% para prueba

print(f"📊 Configuración del proyecto:")
print(f"   • Ticker: {TICKER}")
print(f"   • Datos: {NUM_DATOS} puntos históricos")
print(f"   • Ventana temporal: {LOOKBACK_WINDOW} días")
print(f"   • División: {TRAIN_SPLIT:.0%} train, {VAL_SPLIT:.0%} val, {TEST_SPLIT:.0%} test")
