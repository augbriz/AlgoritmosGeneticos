import numpy as np                
import pandas as pd               
import matplotlib.pyplot as plt   
import warnings
warnings.filterwarnings('ignore')
import datetime
today = datetime.date.today()
print("Fecha actual:", today) 

import yfinance as yf            

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

PREDICTION_HORIZON = 5

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

def crear_secuencias_temporales(datos, lookback_window, prediction_horizon=PREDICTION_HORIZON):
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
    
    
    
    y_plot = y[:, 0] if y.ndim == 2 else y  # t+1
    plt.figure(figsize=(14,6))
    plt.plot(range(len(y_plot)), y_plot, label='Serie', color='lightgray')

    plt.plot(range(0, train_end), y_plot[:train_end], label='Train', color='blue')
    plt.plot(range(train_end, val_end), y_plot[train_end:val_end], label='Validation', color='orange')
    plt.plot(range(val_end, n_samples), y_plot[val_end:], label='Test', color='green')

    plt.title("División Train / Validation / Test ")
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
    model.add(Dense(PREDICTION_HORIZON, activation='linear', name='Output'))
    
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

def rmse(y_true, y_pred):
    """Calcula el Root Mean Square Error"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def graficar_metricas_entrenamiento(historial):
    """
    Grafica la evolución de las métricas durante el entrenamiento
    
    Args:
        historial: Historial devuelto por model.fit()
    """
    # Crear figura con subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Gráfico 1: Loss
    axes[0].plot(historial.history['loss'], label='Training Loss', linewidth=2, color='blue')
    axes[0].plot(historial.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
    axes[0].set_title('Evolución del Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico 2: RMSE
    axes[1].plot(historial.history['rmse'], label='Training RMSE', linewidth=2, color='blue')
    axes[1].plot(historial.history['val_rmse'], label='Validation RMSE', linewidth=2, color='red')
    axes[1].set_title('Evolución del RMSE', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('RMSE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Gráfico 3: Learning Rate (si está disponible)
    if 'learning_rate' in historial.history:
        axes[2].plot(historial.history['learning_rate'], linewidth=2, color='green')
        axes[2].set_title('Evolución del Learning Rate', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')  # Escala logarítmica para mejor visualización
        axes[2].grid(True, alpha=0.3)
    else:
        # Si no hay learning rate, mostrar un gráfico de comparación loss vs val_loss
        epochs = range(1, len(historial.history['loss']) + 1)
        axes[2].plot(epochs, historial.history['loss'], 'b-', label='Training Loss')
        axes[2].plot(epochs, historial.history['val_loss'], 'r-', label='Validation Loss')
        axes[2].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir resumen de métricas finales
    print("\n" + "="*60)
    print("RESUMEN DE MÉTRICAS FINALES")
    print("="*60)
    print(f"Loss final (training): {historial.history['loss'][-1]:.6f}")
    print(f"Loss final (validation): {historial.history['val_loss'][-1]:.6f}")
    print(f"RMSE final (training): {historial.history['rmse'][-1]:.6f}")
    print(f"RMSE final (validation): {historial.history['val_rmse'][-1]:.6f}")
    
    # Encontrar la mejor época (menor val_loss)
    mejor_epoch = np.argmin(historial.history['val_loss']) + 1
    mejor_val_loss = min(historial.history['val_loss'])
    mejor_val_rmse = historial.history['val_rmse'][mejor_epoch - 1]
    
    print(f"\nMejor época: {mejor_epoch}")
    print(f"Mejor val_loss: {mejor_val_loss:.6f}")
    print(f"Mejor val_rmse: {mejor_val_rmse:.6f}")
    print("="*60)

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

def graficar_predicciones_multi(modelo, X, y_real, scaler, h=0, titulo=None):
    if titulo is None:
        titulo = f'Predicciones (h = t+{h+1})'

    y_pred = modelo.predict(X, verbose=0)           # (N, H)
    
    # Validar que el horizonte solicitado esté disponible
    max_horizonte_disponible = y_pred.shape[1]
    if h >= max_horizonte_disponible:
        print(f"  Horizonte h={h} no disponible. Máximo disponible: {max_horizonte_disponible-1}")
        print(f"Usando horizonte h={max_horizonte_disponible-1} en su lugar.")
        h = max_horizonte_disponible - 1
    
    # seleccionar horizonte h
    y_pred_h_norm = y_pred[:, h].reshape(-1,1)
    y_real_h_norm = y_real[:, h].reshape(-1,1)

    y_pred_h = scaler.inverse_transform(y_pred_h_norm).ravel()
    y_real_h = scaler.inverse_transform(y_real_h_norm).ravel()

    print(f"RMSE (h=t+{h+1}) en USD: {rmse(y_real_h, y_pred_h):.4f}")

    # Graficar
    plt.figure(figsize=(14, 8))
    plt.plot(y_real_h, label='Real', linewidth=2, color='blue')
    plt.plot(y_pred_h, label='Predicción', linewidth=2, color='red')
    plt.title(titulo, fontsize=16, fontweight='bold')
    plt.xlabel('Días')
    plt.ylabel('Precio (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.show()

def evaluar_predicciones_futuras(modelo, X_test, y_test, scaler, datos_historicos, fechas_historicas=None, max_horizonte=PREDICTION_HORIZON):
    """
    Visualización simple y efectiva de predicciones futuras con análisis de error
    
    Args:
        modelo: Modelo LSTM entrenado
        X_test, y_test: Datos de prueba
        scaler: Scaler para desnormalizar
        datos_historicos: Precios históricos para el contexto
        fechas_historicas: Fechas correspondientes a los datos históricos
        max_horizonte: Número de días a predecir (default: PREDICTION_HORIZON)
    """
    
    # 1. Obtener predicciones y calcular RMSE por horizonte
    y_pred_all = modelo.predict(X_test, verbose=0)
    
    rmse_por_horizonte = []
    predicciones_ejemplo = []
    
    # Calcular RMSE para cada horizonte usando datos de test
    for h in range(max_horizonte):
        y_pred_h_norm = y_pred_all[:, h].reshape(-1, 1)
        y_real_h_norm = y_test[:, h].reshape(-1, 1)
        
        y_pred_h = scaler.inverse_transform(y_pred_h_norm).ravel()
        y_real_h = scaler.inverse_transform(y_real_h_norm).ravel()
        
        rmse_h = rmse(y_real_h, y_pred_h)
        rmse_por_horizonte.append(rmse_h)
        
        # Tomar la última predicción como ejemplo para el futuro
        predicciones_ejemplo.append(y_pred_h[-1])
    
    # Calcular RMSE promedio
    rmse_promedio = np.mean(rmse_por_horizonte)
    
    # 2. GRÁFICO 1: Serie histórica + predicciones futuras con banda de error
    plt.figure(figsize=(15, 8))
    
    # Datos históricos (últimos 100 días para contexto)
    n_contexto = min(100, len(datos_historicos))
    hist_reciente = datos_historicos[-n_contexto:]
    
    # Crear índices para el gráfico
    x_historico = range(len(hist_reciente))
    x_futuro = range(len(hist_reciente), len(hist_reciente) + max_horizonte)
    
    # Plotear serie histórica
    plt.plot(x_historico, hist_reciente, 'b-', linewidth=2, label='Histórico', alpha=0.8)
    
    # Plotear predicciones futuras
    plt.plot(x_futuro, predicciones_ejemplo, 'r-', linewidth=3, label='Predicciones', marker='o', markersize=6)
    
    # Agregar bandas de error (área de confianza)
    upper_band = [pred + error for pred, error in zip(predicciones_ejemplo, rmse_por_horizonte)]
    lower_band = [pred - error for pred, error in zip(predicciones_ejemplo, rmse_por_horizonte)]
    
    plt.fill_between(x_futuro, lower_band, upper_band, alpha=0.2, color='red', label='±RMSE (zona de confianza)')
    
    plt.title('Predicciones Futuras con Banda de Error', fontsize=16, fontweight='bold')
    plt.xlabel('Días')
    plt.ylabel('Precio (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 3. GRÁFICO 2: Tabla de predicciones
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Preparar datos para la tabla
    tabla_data = []
    for i in range(max_horizonte):
        tabla_data.append([
            f't+{i+1}',
            f'${predicciones_ejemplo[i]:.2f}',
            f'±${rmse_por_horizonte[i]:.2f}'
        ])
    
    # Agregar fila del promedio
    tabla_data.append(['', '', ''])  # Fila vacía
    tabla_data.append(['PROMEDIO', '', f'±${rmse_promedio:.2f}'])
    
    # Crear tabla
    tabla = ax.table(cellText=tabla_data,
                    colLabels=['Horizonte', 'Predicción (USD)', 'RMSE histórico (USD)'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.35, 0.4])
    
    # Formatear tabla
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(12)
    tabla.scale(1.2, 2)
    
    # Resaltar header
    for i in range(3):
        tabla[(0, i)].set_facecolor('#4CAF50')
        tabla[(0, i)].set_text_props(weight='bold', color='white')
    
    # Resaltar fila del promedio
    for i in range(3):
        tabla[(max_horizonte + 2, i)].set_facecolor('#FFC107')
        tabla[(max_horizonte + 2, i)].set_text_props(weight='bold')
    
    plt.title('Tabla de Predicciones con Error Esperado', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    # 4. GRÁFICO 3: Curva de error vs horizonte
    plt.figure(figsize=(10, 6))
    
    horizontes = list(range(1, max_horizonte + 1))
    plt.plot(horizontes, rmse_por_horizonte, 'o-', linewidth=3, markersize=10, color='red')
    
    # Agregar línea del promedio
    plt.axhline(y=rmse_promedio, color='blue', linestyle='--', linewidth=2, 
               label=f'RMSE Promedio: ${rmse_promedio:.2f}')
    
    # Anotar valores en cada punto
    for i, (h, error) in enumerate(zip(horizontes, rmse_por_horizonte)):
        plt.annotate(f'${error:.2f}', (h, error), textcoords="offset points", 
                    xytext=(0,15), ha='center', fontweight='bold', fontsize=11)
    
    plt.title('Evolución del Error por Horizonte de Predicción', fontsize=16, fontweight='bold')
    plt.xlabel('Horizonte (días adelante)', fontsize=12)
    plt.ylabel('RMSE (USD)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 5. Resumen en consola
    print("\n" + "="*60)
    print(" RESUMEN DE PREDICCIONES FUTURAS")
    print("="*60)
    for i, (pred, error) in enumerate(zip(predicciones_ejemplo, rmse_por_horizonte)):
        print(f"t+{i+1}: ${pred:.2f} (el modelo suele fallar ~${error:.2f} USD)")
    print(f"\n RMSE PROMEDIO: ${rmse_promedio:.2f} USD")
    print("="*60)
    
    return {
        'predicciones': predicciones_ejemplo,
        'rmse_por_horizonte': rmse_por_horizonte,
        'rmse_promedio': rmse_promedio
    }
    
    # Resumen interpretativo
    print("\n" + "="*70)
    print("RESUMEN INTERPRETATIVO")
    print("="*70)
    
    mejor_horizonte = horizontes[np.argmin(rmse_valores)]
    peor_horizonte = horizontes[np.argmax(rmse_valores)]
    
    print(f"• RMSE PROMEDIO (todos los horizontes): ${rmse_promedio:.2f} USD")
    print(f"• Mejor rendimiento: t+{mejor_horizonte} (RMSE: ${min(rmse_valores):.2f} USD)")
    print(f"• Peor rendimiento: t+{peor_horizonte} (RMSE: ${max(rmse_valores):.2f} USD)")
    print(f"• Incremento de error promedio por horizonte: ${np.mean(np.diff(rmse_valores)):.2f} USD")
    
    degradacion = ((rmse_valores[-1] - rmse_valores[0]) / rmse_valores[0]) * 100
    print(f"• Degradación total (t+1 vs t+{max_horizonte}): {degradacion:.1f}%")
    
    print(f"\n Interpretación del rendimiento:")
    for i, (h, rmse_val) in enumerate(zip(horizontes, rmse_valores)):
        if i == 0:
            print(f"  - El modelo erra ~${rmse_val:.1f} USD en t+{h}")
        else:
            diff = rmse_val - rmse_valores[0]
            print(f"  - El modelo erra ~${rmse_val:.1f} USD en t+{h} (+${diff:.1f} USD vs t+1)")
    
    print(f"  - PROMEDIO: El modelo tiene un error promedio de ~${rmse_promedio:.1f} USD")
    print("="*70)
    
    return {
        'horizontes': horizontes,
        'rmse': rmse_valores,
        'rmse_promedio': rmse_promedio,
        'mejor_horizonte': mejor_horizonte,
        'peor_horizonte': peor_horizonte
    }
    
def _rmse_promedio_normalizado(modelo, X_val, y_val):
    """RMSE promedio (normalizado) sobre todos los horizontes."""
    y_pred = modelo.predict(X_val, verbose=0)  # (N, H)
    H = y_val.shape[1] if y_val.ndim == 2 else 1
    rmse_h = []
    for h in range(H):
        y_true_h = y_val[:, h] if H > 1 else y_val
        y_pred_h = y_pred[:, h] if H > 1 else y_pred.ravel()
        rmse_h.append(np.sqrt(np.mean((y_true_h - y_pred_h) ** 2)))
    return float(np.mean(rmse_h)), [float(v) for v in rmse_h]

def _rmse_promedio_usd(modelo, X_val, y_val, scaler):
    """RMSE promedio (en USD) sobre todos los horizontes."""
    y_pred = modelo.predict(X_val, verbose=0)  # (N, H)
    H = y_val.shape[1] if y_val.ndim == 2 else 1
    rmse_h_usd = []
    for h in range(H):
        y_true_h = y_val[:, h].reshape(-1, 1) if H > 1 else y_val.reshape(-1, 1)
        y_pred_h = y_pred[:, h].reshape(-1, 1) if H > 1 else y_pred.reshape(-1, 1)
        y_true_h_usd = scaler.inverse_transform(y_true_h).ravel()
        y_pred_h_usd = scaler.inverse_transform(y_pred_h).ravel()
        rmse_h_usd.append(float(np.sqrt(np.mean((y_true_h_usd - y_pred_h_usd) ** 2))))
    return float(np.mean(rmse_h_usd)), rmse_h_usd

def ga_evaluate(hp: dict, do_plots: bool = False):
    """
    Entrena y evalúa un conjunto de hiperparámetros en VALIDACIÓN.
    hp: dict con claves opcionales:
        - lstm_units (int)      - num_layers (int)
        - dropout_rate (float)  - learning_rate (float)
        - batch_size (int)      - epochs (int)
    Devuelve: dict con 'fitness' (RMSE prom. normalizado) y métricas extra.
    """
    # --- 1) Leer hiperparámetros con defaults ---
    lstm_units    = int(hp.get("lstm_units", 50))
    num_layers    = int(hp.get("num_layers", 2))
    dropout_rate  = float(hp.get("dropout_rate", 0.2))
    learning_rate = float(hp.get("learning_rate", 1e-3))
    batch_size    = int(hp.get("batch_size", 32))
    epochs        = int(hp.get("epochs", 100))

    # --- 2) Construir modelo con esos HP ---
    modelo = crear_modelo_lstm(
        input_shape=(LOOKBACK_WINDOW, 1),
        lstm_units=lstm_units,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    cbs = configurar_callbacks()

    # --- 3) Entrenar (train/val) ---
    hist = entrenar_modelo(
        modelo, X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size, callbacks=cbs
    )

    # --- 4) Fitness: RMSE promedio normalizado en VALIDACIÓN ---
    rmse_prom_norm, rmse_h_norm = _rmse_promedio_normalizado(modelo, X_val, y_val)

    # --- 5) Métrica humana: RMSE promedio USD (VALIDACIÓN) ---
    rmse_prom_usd, rmse_h_usd = _rmse_promedio_usd(modelo, X_val, y_val, scaler)

    # --- 6) (Opcional) gráficos rápidos ---
    if do_plots:
        try:
            graficar_metricas_entrenamiento(hist)
            h_last = (y_val.shape[1] - 1) if y_val.ndim == 2 else 0
            graficar_predicciones_multi(
                modelo, X_val, y_val, scaler, h=h_last,
                titulo=f'Validación (t+{h_last+1})'
            )
        except Exception as e:
            print(f"[GA] plot skip: {e}")

    # --- 7) Salida para el AG ---
    return {
        "fitness": rmse_prom_norm,                  # <- usar para seleccionar (minimizar)
        "rmse_prom_norm": rmse_prom_norm,
        "rmse_h_norm": rmse_h_norm,                # lista H (t+1 .. t+H)
        "rmse_prom_usd": rmse_prom_usd,
        "rmse_h_usd": rmse_h_usd,                  # lista H (en USD)
        "best_epoch": int(np.argmin(hist.history['val_loss']) + 1),
        "val_loss_min": float(np.min(hist.history['val_loss'])),
        "params": {
            "lstm_units": lstm_units,
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "lookback_window": LOOKBACK_WINDOW,
            "prediction_horizon": PREDICTION_HORIZON
        }
    }
    

#defino la función para llamarla desde ag_opt.py
def main():
# Graficar evolución de métricas durante entrenamiento
    graficar_metricas_entrenamiento(historial_modelo)

# Evaluar modelo
    resultados_evaluacion = evaluar_modelo(modelo_lstm, X_test, y_test)

# Graficar predicciones del último horizonte disponible
    ultimo_horizonte = PREDICTION_HORIZON - 1  # Índice del último horizonte (0-based)
    graficar_predicciones_multi(modelo_lstm, X_test, y_test, scaler, h=ultimo_horizonte, 
                           titulo=f'Predicciones en Test (t+{ultimo_horizonte + 1})')

# Evaluación completa de todos los horizontes de predicción
    print("\n" + " ANÁLISIS COMPLETO DE HORIZONTES DE PREDICCIÓN ")

# Necesitamos los datos históricos desnormalizados para el contexto
    datos_historicos_desnorm = scaler.inverse_transform(datos_normalizados.reshape(-1, 1)).ravel()

    resultados_horizontes = evaluar_predicciones_futuras(
        modelo_lstm, X_test, y_test, scaler, 
        datos_historicos=datos_historicos_desnorm, 
        fechas_historicas=None,  
        max_horizonte=PREDICTION_HORIZON  
    )

if __name__ == "__main__":
    main()