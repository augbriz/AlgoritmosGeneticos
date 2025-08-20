import numpy as np                
import pandas as pd               
import matplotlib.pyplot as plt   
import warnings
warnings.filterwarnings('ignore')
import datetime
today = datetime.date.today()
print("Fecha actual:", today) 
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

print("Importaciones completadas exitosamente")
print("Preparado para implementar modelo LSTM")

# =============================================================================
# CONSTANTES DEL PROYECTO


# Configuración de datos
TICKER = "YPF"
START_DATE = "2008-01-01"
END_DATE = today
NUM_DATOS = 1000  # Últimos 1000 datos (como en análisis caótico)

# Configuración del modelo (valores base - se optimizarán con AG)
LOOKBACK_WINDOW = 20      # Ventana temporal: últimos 20 días para predecir
TRAIN_SPLIT = 0.7         # 70% para entrenamiento
VAL_SPLIT = 0.15          # 15% para validación
TEST_SPLIT = 0.15         # 15% para prueba

print(f"Configuracion del proyecto:")
print(f"   • Ticker: {TICKER}")
print(f"   • Datos: {NUM_DATOS} puntos historicos")
print(f"   • Ventana temporal: {LOOKBACK_WINDOW} dias")
print(f"   • Division: {TRAIN_SPLIT:.0%} train, {VAL_SPLIT:.0%} val, {TEST_SPLIT:.0%} test")


# CARGA Y VISUALIZACIÓN DE DATOS


def cargar_datos_ypf():    
    # Descargar datos 
    data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    
    # Verificar que los datos se descargaron correctamente
    if data.empty or 'Close' not in data.columns:
        raise ValueError(f"No se pudieron obtener datos de cierre para {TICKER}")
    
    # Tomar los últimos NUM_DATOS puntos (1000, como en análisis caótico)
    precios_cierre = data['Close'].dropna().tail(NUM_DATOS)
    
    print(f"Datos cargados exitosamente:")
    print(f"   • Total de datos: {len(precios_cierre)}")
    print(f"   • Rango de fechas: {precios_cierre.index[0].date()} a {precios_cierre.index[-1].date()}")
    print(f"   • Precio minimo: ${float(precios_cierre.min()):.2f}")
    print(f"   • Precio maximo: ${float(precios_cierre.max()):.2f}")
    print(f"   • Precio promedio: ${float(precios_cierre.mean()):.2f}")
    print(f"   • Volatilidad (std): ${float(precios_cierre.std()):.2f}")
    
    return precios_cierre

def visualizar_datos(precios):
    print("\nCreando visualizacion de la serie temporal...")
    
    plt.figure(figsize=(14, 8))
    

    plt.subplot(2, 1, 1)
    plt.plot(precios.index, precios.values, linewidth=0.8, color='blue', alpha=0.8)
    plt.title(f'Serie Temporal - {TICKER} (Últimos {len(precios)} días)', fontsize=16, fontweight='bold')
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Precio de Cierre (USD)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    
    stats_text = f'Datos: {len(precios)} puntos\nÚltimo precio: ${float(precios.iloc[-1]):.2f}\nPromedio: ${float(precios.mean()):.2f}'
    plt.text(0.02, 0.98, stats_text, 
              transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # distribución de precios
    plt.subplot(2, 1, 2)
    plt.hist(precios.values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribución de Precios', fontsize=14)
    plt.xlabel('Precio (USD)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
                                                                            
# Cargar datos
datos_ypf = cargar_datos_ypf()

# Visualizar datos
visualizar_datos(datos_ypf)

# Análisis básico
# analisis_basico = analizar_datos_basico(datos_ypf)

# =============================================================================
# PASO 3: PREPROCESAMIENTO PARA LSTM
# =============================================================================

def normalizar_datos(serie):
    """
    Normaliza los datos a escala 0-1 usando MinMaxScaler
    
    ¿Por qué normalizar?
    - Las LSTM funcionan mejor con datos en escala pequeña (0-1)
    - Evita que valores grandes dominen el entrenamiento
    - Acelera la convergencia del modelo
    
    Args:
        serie (pd.Series): Serie temporal de precios
    
    Returns:
        tuple: (datos_normalizados, scaler_objeto)
    """    
    # Crear el escalador MinMax (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Convertir serie a array numpy y reshape para sklearn
    datos_array = serie.values.reshape(-1, 1)
    
    # Ajustar el scaler y transformar los datos
    datos_normalizados = scaler.fit_transform(datos_array)
    
    print(f"Normalizacion completada:")
    print(f"   • Valor original minimo: ${float(serie.min()):.2f}")
    print(f"   • Valor original maximo: ${float(serie.max()):.2f}")
    print(f"   • Valor normalizado minimo: {datos_normalizados.min():.4f}")
    print(f"   • Valor normalizado maximo: {datos_normalizados.max():.4f}")
    
    return datos_normalizados.flatten(), scaler

def crear_secuencias_temporales(datos, lookback_window, prediction_horizon=1):
    """
    Crea secuencias temporales para entrenar la LSTM
    
    ¿Qué hace?
    - Convierte la serie temporal en ventanas deslizantes
    - Cada ventana tiene 'lookback_window' días de historia
    - El objetivo es predecir el siguiente día (o días)
    
    Ejemplo: Si lookback_window=3
    [1,2,3,4,5,6,7] se convierte en:
    X=[[1,2,3], [2,3,4], [3,4,5], [4,5,6]]
    y=[4, 5, 6, 7]
    
    Args:
        datos (array): Datos normalizados
        lookback_window (int): Ventana temporal (días pasados)
        prediction_horizon (int): Días a predecir (normalmente 1)
    
    Returns:
        tuple: (X, y) arrays para entrenamiento
    """    
    X, y = [], []
    
    # Crear ventanas deslizantes
    for i in range(lookback_window, len(datos) - prediction_horizon + 1):
        # Ventana de entrada (últimos 'lookback_window' días)
        X.append(datos[i-lookback_window:i])
        # Objetivo (siguiente día o días)
        if prediction_horizon == 1:
            y.append(datos[i])  # Solo el siguiente día
        else:
            y.append(datos[i:i+prediction_horizon])  # Múltiples días
    
    X, y = np.array(X), np.array(y)
    
    print(f"Secuencias creadas:")
    print(f"   • Total de secuencias: {len(X)}")
    print(f"   • Forma de X (entrada): {X.shape}")
    print(f"   • Forma de y (objetivo): {y.shape}")
    
    # Reshape X para LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    print(f"   • X reshape para LSTM: {X.shape}")
    
    return X, y

def dividir_datos(X, y, train_split=0.7, val_split=0.15, test_split=0.15):
    """
    Divide los datos en entrenamiento, validación y prueba
    
    Para series temporales NO usamos división aleatoria
    Mantenemos el orden temporal:
    - Train: datos más antiguos
    - Validation: datos intermedios  
    - Test: datos más recientes
    
    Args:
        X, y: Arrays de secuencias
        train_split, val_split, test_split: Proporciones de división
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """

    
    # Validar que las proporciones sumen 1
    if abs(train_split + val_split + test_split - 1.0) > 0.001:
        raise ValueError("Las proporciones deben sumar 1.0")
    
    n_samples = len(X)
    
    # Calcular índices de corte (manteniendo orden temporal)
    train_end = int(n_samples * train_split)
    val_end = int(n_samples * (train_split + val_split))
    
    # Dividir manteniendo orden temporal
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]
    
    print(f"Division completada:")
    print(f"   • Entrenamiento: {len(X_train)} secuencias ({len(X_train)/n_samples:.1%})")
    print(f"   • Validacion: {len(X_val)} secuencias ({len(X_val)/n_samples:.1%})")
    print(f"   • Prueba: {len(X_test)} secuencias ({len(X_test)/n_samples:.1%})")
    
    
    plt.figure(figsize=(14,6))
    plt.plot(range(len(y)), y, label='Serie completa', color='lightgray')

    plt.plot(range(0, train_end), y[:train_end], label='Train', color='blue')
    plt.plot(range(train_end, val_end), y[train_end:val_end], label='Validation', color='orange')
    plt.plot(range(val_end, n_samples), y[val_end:], label='Test', color='green')

    plt.title("División Train / Validation / Test")
    plt.xlabel("Índice temporal")
    plt.ylabel("Valor normalizado")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    
    return X_train, X_val, X_test, y_train, y_val, y_test

def visualizar_normalizacion(datos_originales, datos_normalizados):
    """
    Visualiza el efecto de la normalización
    """
    print("\nCreando visualizacion de la normalizacion...")
    
    plt.figure(figsize=(15, 6))
    
    # Datos originales
    plt.subplot(1, 2, 1)
    plt.plot(datos_originales.values, color='blue', alpha=0.8)
    plt.title('Datos Originales', fontsize=14, fontweight='bold')
    plt.ylabel('Precio (USD)', fontsize=12)
    plt.xlabel('Tiempo (días)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # # Datos normalizados
    plt.subplot(1, 2, 2)
    plt.plot(datos_normalizados, color='red', alpha=0.8)
    plt.title('Datos Normalizados (0-1)', fontsize=14, fontweight='bold')
    plt.ylabel('Valor Normalizado', fontsize=12)
    plt.xlabel('Tiempo (días)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Ejecutar preprocesamiento

# Calcular cortes sobre la SERIE CRUDA (sin normalizar)
n_raw = len(datos_ypf)
train_end_raw = int(n_raw * TRAIN_SPLIT)
val_end_raw   = int(n_raw * (TRAIN_SPLIT + VAL_SPLIT))

# Ajustar scaler SOLO con el tramo de entrenamiento
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(datos_ypf.iloc[:train_end_raw].values.reshape(-1, 1))

# Transformar toda la serie usando ese scaler
datos_normalizados = scaler.transform(datos_ypf.values.reshape(-1, 1)).flatten()

# Crear secuencias sobre la serie ya normalizada
X, y = crear_secuencias_temporales(datos_normalizados, LOOKBACK_WINDOW)

# Dividir secuencias cronológicamente 
X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(
    X, y, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
)
# =============================================================================
# CONSTRUCCIÓN DEL MODELO LSTM
# =============================================================================

def crear_modelo_lstm(input_shape, 
                     lstm_units=50, 
                     num_layers=2, 
                     dropout_rate=0.2,
                     learning_rate=0.001):
    """
    Crea el modelo LSTM con arquitectura optimizada para series financieras
    
    Args:
        input_shape: Forma de entrada (timesteps, features)
        lstm_units: Neuronas por capa LSTM
        num_layers: Número de capas LSTM
        dropout_rate: Tasa de dropout para regularización
        learning_rate: Tasa de aprendizaje del optimizador
    
    Returns:
        Modelo LSTM compilado y listo para entrenar
    """
    print(f"\n Hiperparametros:")
    print(f"   • Input shape: {input_shape}")
    print(f"   • LSTM units: {lstm_units}")
    print(f"   • Num layers: {num_layers}")
    print(f"   • Dropout rate: {dropout_rate}")
    print(f"   • Learning rate: {learning_rate}")
    
    model = Sequential()
    
    # Primera capa LSTM
    model.add(LSTM(
        units=lstm_units,
        return_sequences=True if num_layers > 1 else False,
        input_shape=input_shape,
        name='LSTM_1'
    ))
    model.add(Dropout(dropout_rate, name='Dropout_1'))
    
    # Capas LSTM adicionales
    for i in range(2, num_layers + 1):
        return_seq = True if i < num_layers else False
        model.add(LSTM(
            units=lstm_units,
            return_sequences=return_seq,
            name=f'LSTM_{i}'
        ))
        model.add(Dropout(dropout_rate, name=f'Dropout_{i}'))
    
    # Capa de salida
    model.add(Dense(1, activation='linear', name='Output'))
    
    # Compilar modelo
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    
    print(f"Modelo construido exitosamente")
    return model

def configurar_callbacks():
    """
    Configura callbacks para controlar el entrenamiento
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks

def mostrar_parametros_modelo(modelo):
    """
    Muestra información detallada sobre los parámetros del modelo
    """
    # print("\nAnalisis de parametros del modelo:")
    # print("="*50)
    
    # total_params = modelo.count_params()
    # trainable_params = sum([tf.keras.backend.count_params(w) for w in modelo.trainable_weights])
    
    # print(f"Total de parametros: {total_params:,}")
    # print(f"Parametros entrenables: {trainable_params:,}")
    # print(f"Parametros no entrenables: {total_params - trainable_params:,}")
    
    # # Desglose por capa
    # print(f"\nDesglose por capas:")
    # for i, layer in enumerate(modelo.layers):
    #     layer_type = type(layer).__name__
    #     params = layer.count_params()
    #     trainable = sum([tf.keras.backend.count_params(w) for w in layer.trainable_weights])
    #     non_trainable = params - trainable
        
    #     print(f"   • Capa {i+1} ({layer_type}): {params:,} parametros")
    #     print(f"     - Entrenables: {trainable:,}")
    #     print(f"     - No entrenables: {non_trainable:,}")
    
    # print("\nAnalisis de parametros completado")
    pass


# =============================================================================
# ENTRENAMIENTO DEL MODELO LSTM
# =============================================================================

def entrenar_modelo(modelo, X_train, y_train, X_val, y_val, 
                   epochs=100, batch_size=32, callbacks=None):
    """
    Entrena el modelo LSTM con los datos proporcionados
    
    Args:
        modelo: Modelo LSTM compilado
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        epochs: Número de épocas para entrenar
        batch_size: Tamaño del batch
        callbacks: Lista de callbacks para el entrenamiento
    
    Returns:
        Historial del entrenamiento (historial de pérdidas y métricas)
    """
    print("\nIniciando entrenamiento del modelo LSTM...")
    
    # Entrenar modelo
    historial = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=False, 
        verbose=2
    )
    
    print("Entrenamiento completado")
    return historial

# Crear modelo
modelo_lstm = crear_modelo_lstm(input_shape=(LOOKBACK_WINDOW, 1))

# Configurar callbacks
callbacks = configurar_callbacks()

# Entrenar modelo
historial_modelo = entrenar_modelo(
    modelo_lstm, X_train, y_train, X_val, y_val, 
    epochs=100, batch_size=32, callbacks=callbacks
)

# =============================================================================
# FASE 5: EVALUACIÓN Y PRUEBA DEL MODELO LSTM
# =============================================================================

def evaluar_modelo(modelo, X_test, y_test):
    """
    Evalúa el modelo LSTM con los datos de prueba
    
    Args:
        modelo: Modelo LSTM entrenado
        X_test, y_test: Datos de prueba
    
    Returns:
        dict: Pérdida y métricas del modelo en los datos de prueba
    """    
    # Evaluar modelo
    resultados = modelo.evaluate(X_test, y_test, verbose=0)
    
    # Crear diccionario de resultados
    metrica_resultados = dict(zip(modelo.metrics_names, resultados))
    
    print(f"Evaluacion completada")
    for nombre, valor in metrica_resultados.items():
        print(f"   • {nombre}: {valor:.4f}")
    
    return metrica_resultados

def graficar_predicciones(modelo, X, y_real, scaler, titulo='Predicciones del Modelo'):
    """
    Grafica las predicciones del modelo LSTM contra los valores reales
    
    Args:
        modelo: Modelo LSTM entrenado
        X: Datos de entrada (X_test o similar)
        y_real: Valores reales (y_test o similar)
        scaler: Objeto scaler para deshacer la normalización
        titulo: Título del gráfico
    """
    print(f"\nGraficando {titulo}...")
    
    
    y_pred = modelo.predict(X)
    
    # Invertir la normalización
    y_real_invertido = scaler.inverse_transform(y_real.reshape(-1, 1))
    y_pred_invertido = scaler.inverse_transform(y_pred)
    
    def evaluar_rmse_en_usd(modelo, X_test, y_test, scaler):
        y_pred = modelo.predict(X_test, verbose=0).ravel()
        y_pred_usd = scaler.inverse_transform(y_pred.reshape(-1,1)).ravel()
        y_test_usd = scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
        return np.sqrt(np.mean((y_test_usd - y_pred_usd)**2))

    rmse_model_usd = evaluar_rmse_en_usd(modelo_lstm, X_test, y_test, scaler)
    print(f"Modelo LSTM - RMSE en USD: {rmse_model_usd:.4f}")

    
    # Graficar
    plt.figure(figsize=(14, 8))
    plt.plot(y_real_invertido, label='Real', linewidth=2, color='blue')
    plt.plot(y_pred_invertido, label='Predicción', linewidth=2, color='red')
    plt.title(titulo, fontsize=16, fontweight='bold')
    plt.xlabel('Días')
    plt.ylabel('Precio (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()

# Evaluar modelo
resultados_evaluacion = evaluar_modelo(modelo_lstm, X_test, y_test)

# Graficar predicciones
graficar_predicciones(modelo_lstm, X_test, y_test, scaler, titulo='Predicciones en Datos de Prueba')




