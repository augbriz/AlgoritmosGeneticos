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

print("✅ Importaciones completadas exitosamente")
print("🎯 Preparado para implementar modelo LSTM")

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

print(f"📊 Configuración del proyecto:")
print(f"   • Ticker: {TICKER}")
print(f"   • Datos: {NUM_DATOS} puntos históricos")
print(f"   • Ventana temporal: {LOOKBACK_WINDOW} días")
print(f"   • División: {TRAIN_SPLIT:.0%} train, {VAL_SPLIT:.0%} val, {TEST_SPLIT:.0%} test")


# CARGA Y VISUALIZACIÓN DE DATOS


def cargar_datos_ypf():
    """
    Carga los datos históricos de YPF usando la misma configuración
    que en el análisis caótico para mantener consistencia
    
    Returns:
        pd.Series: Serie temporal con los precios de cierre de YPF
    """
    print("\n📥 Descargando datos de YPF...")
    
    # Descargar datos (igual que en YPF.py para mantener consistencia)
    data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    
    # Verificar que los datos se descargaron correctamente
    if data.empty or 'Close' not in data.columns:
        raise ValueError(f"No se pudieron obtener datos de cierre para {TICKER}")
    
    # Tomar los últimos NUM_DATOS puntos (1000, como en análisis caótico)
    precios_cierre = data['Close'].dropna().tail(NUM_DATOS)
    
    print(f"✅ Datos cargados exitosamente:")
    print(f"   • Total de datos: {len(precios_cierre)}")
    print(f"   • Rango de fechas: {precios_cierre.index[0].date()} a {precios_cierre.index[-1].date()}")
    print(f"   • Precio mínimo: ${float(precios_cierre.min()):.2f}")
    print(f"   • Precio máximo: ${float(precios_cierre.max()):.2f}")
    print(f"   • Precio promedio: ${float(precios_cierre.mean()):.2f}")
    print(f"   • Volatilidad (std): ${float(precios_cierre.std()):.2f}")
    
    return precios_cierre

def visualizar_datos(precios):
    """
    Visualiza los datos cargados para verificar que todo esté correcto
    
    Args:
        precios (pd.Series): Serie temporal con los precios de cierre
    """
    print("\n📊 Creando visualización de la serie temporal...")
    
    plt.figure(figsize=(14, 8))
    
    # Gráfico principal
    plt.subplot(2, 1, 1)
    plt.plot(precios.index, precios.values, linewidth=0.8, color='blue', alpha=0.8)
    plt.title(f'Serie Temporal - {TICKER} (Últimos {len(precios)} días)', fontsize=16, fontweight='bold')
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Precio de Cierre (USD)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Agregar información estadística en el gráfico
    stats_text = f'Datos: {len(precios)} puntos\nÚltimo precio: ${float(precios.iloc[-1]):.2f}\nPromedio: ${float(precios.mean()):.2f}'
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Gráfico de distribución de precios
    plt.subplot(2, 1, 2)
    plt.hist(precios.values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribución de Precios', fontsize=14)
    plt.xlabel('Precio (USD)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    


def analizar_datos_basico(precios):
    """
    Análisis básico de los datos para entender mejor la serie temporal
    
    Args:
        precios (pd.Series): Serie temporal con los precios de cierre
    """
    print("\n🔍 Análisis básico de la serie temporal:")
    
    # Estadísticas descriptivas
    print(f"\n📈 Estadísticas descriptivas:")
    print(f"   • Cuenta: {len(precios)}")
    print(f"   • Media: ${float(precios.mean()):.2f}")
    print(f"   • Mediana: ${float(precios.median()):.2f}")
    print(f"   • Desviación estándar: ${float(precios.std()):.2f}")
    print(f"   • Mínimo: ${float(precios.min()):.2f}")
    print(f"   • Máximo: ${float(precios.max()):.2f}")
    
    # Análisis de tendencia básico
    precio_inicial = float(precios.iloc[0])
    precio_final = float(precios.iloc[-1])
    variacion_total = ((precio_final - precio_inicial) / precio_inicial) * 100
    
    print(f"\n📊 Análisis de tendencia:")
    print(f"   • Precio inicial: ${precio_inicial:.2f}")
    print(f"   • Precio final: ${precio_final:.2f}")
    print(f"   • Variación total: {variacion_total:+.2f}%")
    
    # Análisis de volatilidad básico
    returns = precios.pct_change().dropna()
    volatilidad_diaria = float(returns.std())
    volatilidad_anualizada = volatilidad_diaria * np.sqrt(252)  # 252 días de trading por año
    
    print(f"\n⚡ Análisis de volatilidad:")
    print(f"   • Volatilidad diaria: {volatilidad_diaria:.4f}")
    print(f"   • Volatilidad anualizada: {volatilidad_anualizada:.4f} ({volatilidad_anualizada*100:.2f}%)")
    
    return {
        'estadisticas': precios.describe(),
        'variacion_total': variacion_total,
        'volatilidad_anualizada': volatilidad_anualizada,
        'returns': returns
    }

# Ejecutar la carga y análisis de datos
print("\n" + "="*70)
print("🚀 FASE 1: CARGA Y EXPLORACIÓN DE DATOS")
print("="*70)

# Cargar datos
datos_ypf = cargar_datos_ypf()

# Visualizar datos
visualizar_datos(datos_ypf)

# Análisis básico
analisis_basico = analizar_datos_basico(datos_ypf)

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
    print("\n🔄 Normalizando datos a escala 0-1...")
    
    # Crear el escalador MinMax (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Convertir serie a array numpy y reshape para sklearn
    datos_array = serie.values.reshape(-1, 1)
    
    # Ajustar el scaler y transformar los datos
    datos_normalizados = scaler.fit_transform(datos_array)
    
    print(f"✅ Normalización completada:")
    print(f"   • Valor original mínimo: ${float(serie.min()):.2f}")
    print(f"   • Valor original máximo: ${float(serie.max()):.2f}")
    print(f"   • Valor normalizado mínimo: {datos_normalizados.min():.4f}")
    print(f"   • Valor normalizado máximo: {datos_normalizados.max():.4f}")
    
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
    print(f"\n🏗️ Creando secuencias temporales...")
    print(f"   • Ventana temporal: {lookback_window} días")
    print(f"   • Horizonte de predicción: {prediction_horizon} día(s)")
    
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
    
    print(f"✅ Secuencias creadas:")
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
    
    ¡IMPORTANTE! Para series temporales NO usamos división aleatoria
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
    print(f"\n✂️ Dividiendo datos temporalmente...")
    
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
    
    print(f"✅ División completada:")
    print(f"   • Entrenamiento: {len(X_train)} secuencias ({len(X_train)/n_samples:.1%})")
    print(f"   • Validación: {len(X_val)} secuencias ({len(X_val)/n_samples:.1%})")
    print(f"   • Prueba: {len(X_test)} secuencias ({len(X_test)/n_samples:.1%})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def visualizar_normalizacion(datos_originales, datos_normalizados):
    """
    Visualiza el efecto de la normalización
    """
    print("\n📊 Creando visualización de la normalización...")
    
    plt.figure(figsize=(15, 6))
    
    # Datos originales
    plt.subplot(1, 2, 1)
    plt.plot(datos_originales.values, color='blue', alpha=0.8)
    plt.title('Datos Originales', fontsize=14, fontweight='bold')
    plt.ylabel('Precio (USD)', fontsize=12)
    plt.xlabel('Tiempo (días)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Datos normalizados
    plt.subplot(1, 2, 2)
    plt.plot(datos_normalizados, color='red', alpha=0.8)
    plt.title('Datos Normalizados (0-1)', fontsize=14, fontweight='bold')
    plt.ylabel('Valor Normalizado', fontsize=12)
    plt.xlabel('Tiempo (días)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualización de normalización completada")

# Ejecutar preprocesamiento
print("\n" + "="*70)
print("🚀 FASE 2: PREPROCESAMIENTO PARA LSTM")
print("="*70)

# Paso 1: Normalizar datos
datos_normalizados, scaler = normalizar_datos(datos_ypf)

# Paso 2: Visualizar normalización
visualizar_normalizacion(datos_ypf, datos_normalizados)

# Paso 3: Crear secuencias temporales
X, y = crear_secuencias_temporales(datos_normalizados, LOOKBACK_WINDOW)

# Paso 4: Dividir datos
X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(
    X, y, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
)

print(f"\n🎯 Preprocesamiento completado exitosamente!")
print(f"✅ Datos listos para entrenar la LSTM")

# =============================================================================
# FASE 3: CONSTRUCCIÓN DEL MODELO LSTM
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
    print(f"\n🧠 Construyendo modelo LSTM...")
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
        metrics=['mae']
    )
    
    print(f"✅ Modelo construido exitosamente")
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

def visualizar_arquitectura_modelo(modelo):
    """
    Crea visualizaciones intuitivas de la arquitectura LSTM
    
    Args:
        modelo: Modelo LSTM compilado
    """
    print("\n🎨 Creando visualizaciones de la arquitectura...")
    
    # 1. Resumen textual del modelo
    print("\n📋 Resumen de la arquitectura:")
    modelo.summary()
    
    # 2. Visualización gráfica con plot_model
    try:
        from tensorflow.keras.utils import plot_model
        import os
        
        print("\n🖼️ Intentando crear diagrama de arquitectura...")
        
        # Crear el gráfico de arquitectura
        plot_model(
            modelo, 
            to_file='lstm_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',  # Top to Bottom
            expand_nested=True,
            dpi=150
        )
        
        if os.path.exists('lstm_architecture.png'):
            print("✅ Diagrama de arquitectura guardado como 'lstm_architecture.png'")
            print("🔍 Puedes abrir el archivo para ver la arquitectura gráfica")
        else:
            print("❌ No se pudo crear el archivo de diagrama")
        
    except ImportError as e:
        print("⚠️ Para visualización gráfica automática, instala:")
        print("   pip install pydot")
        print("   Y descarga graphviz desde: https://graphviz.gitlab.io/download/")
        print(f"   Error: {str(e)}")
    except Exception as e:
        print(f"❌ Error al crear diagrama: {str(e)}")
        print("🔄 Continuando con visualización manual...")
    
    # 3. Visualización manual de la arquitectura
    visualizar_arquitectura_manual(modelo)

def visualizar_arquitectura_manual(modelo):
    """
    Crea una visualización manual e intuitiva de la arquitectura LSTM
    """
    print("\n🧠 Arquitectura LSTM visualizada:")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Configurar el gráfico
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Título
    ax.text(5, 7.5, 'Arquitectura LSTM para Predicción de YPF', 
            fontsize=16, fontweight='bold', ha='center')
    
    # 1. Datos de entrada
    entrada = plt.Rectangle((0.5, 6), 2, 0.8, fill=True, 
                           facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(entrada)
    ax.text(1.5, 6.4, 'ENTRADA\n(20 días, 1 feature)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 2. Primera capa LSTM
    lstm1 = plt.Rectangle((0.5, 4.5), 2, 1, fill=True, 
                         facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(lstm1)
    ax.text(1.5, 5, 'LSTM Layer 1\n(50 unidades)\nMemoria temporal', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 3. Dropout 1
    dropout1 = plt.Rectangle((3, 4.5), 1.5, 1, fill=True, 
                            facecolor='orange', edgecolor='black', linewidth=2)
    ax.add_patch(dropout1)
    ax.text(3.75, 5, 'Dropout\n(20%)\nRegularización', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 4. Segunda capa LSTM
    lstm2 = plt.Rectangle((0.5, 3), 2, 1, fill=True, 
                         facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(lstm2)
    ax.text(1.5, 3.5, 'LSTM Layer 2\n(50 unidades)\nPatrones complejos', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 5. Dropout 2
    dropout2 = plt.Rectangle((3, 3), 1.5, 1, fill=True, 
                            facecolor='orange', edgecolor='black', linewidth=2)
    ax.add_patch(dropout2)
    ax.text(3.75, 3.5, 'Dropout\n(20%)\nRegularización', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 6. Capa densa
    dense = plt.Rectangle((0.5, 1.5), 2, 1, fill=True, 
                         facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(dense)
    ax.text(1.5, 2, 'Dense Layer\n(1 neurona)\nPredicción final', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 7. Salida
    salida = plt.Rectangle((0.5, 0.2), 2, 0.8, fill=True, 
                          facecolor='yellow', edgecolor='black', linewidth=2)
    ax.add_patch(salida)
    ax.text(1.5, 0.6, 'SALIDA\nPrecio predicho\n(normalizado)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Flechas de flujo
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    ax.annotate('', xy=(1.5, 4.5), xytext=(1.5, 5.8), arrowprops=arrow_props)
    ax.annotate('', xy=(1.5, 3), xytext=(1.5, 4.5), arrowprops=arrow_props)
    ax.annotate('', xy=(1.5, 1.5), xytext=(1.5, 3), arrowprops=arrow_props)
    ax.annotate('', xy=(1.5, 0.2), xytext=(1.5, 1.5), arrowprops=arrow_props)
    
    # Explicaciones laterales
    ax.text(6, 6, '📊 FLUJO DE DATOS', fontsize=12, fontweight='bold')
    ax.text(6, 5.5, '1. Entrada: 20 días de precios normalizados', fontsize=10)
    ax.text(6, 5.2, '2. LSTM 1: Aprende patrones temporales básicos', fontsize=10)
    ax.text(6, 4.9, '3. Dropout: Previene overfitting', fontsize=10)
    ax.text(6, 4.6, '4. LSTM 2: Aprende patrones más complejos', fontsize=10)
    ax.text(6, 4.3, '5. Dropout: Más regularización', fontsize=10)
    ax.text(6, 4.0, '6. Dense: Convierte a predicción numérica', fontsize=10)
    ax.text(6, 3.7, '7. Salida: Precio del día siguiente', fontsize=10)
    
    ax.text(6, 3, '🧠 CONCEPTOS CLAVE', fontsize=12, fontweight='bold')
    ax.text(6, 2.5, '• LSTM: Memoria de largo y corto plazo', fontsize=10)
    ax.text(6, 2.2, '• Dropout: "Apaga" neuronas aleatoriamente', fontsize=10)
    ax.text(6, 1.9, '• Secuencial: Información fluye hacia adelante', fontsize=10)
    ax.text(6, 1.6, '• 50 unidades: Balance capacidad/eficiencia', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def mostrar_parametros_modelo(modelo):
    """
    Muestra información detallada sobre los parámetros del modelo
    """
    print("\n📊 Análisis de parámetros del modelo:")
    print("="*50)
    
    total_params = modelo.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in modelo.trainable_weights])
    
    print(f"🔢 Total de parámetros: {total_params:,}")
    print(f"🎯 Parámetros entrenables: {trainable_params:,}")
    print(f"🔒 Parámetros no entrenables: {total_params - trainable_params:,}")
    
    # Desglose por capa
    print(f"\n📋 Desglose por capas:")
    for i, layer in enumerate(modelo.layers):
        layer_type = type(layer).__name__
        params = layer.count_params()
        trainable = sum([tf.keras.backend.count_params(w) for w in layer.trainable_weights])
        non_trainable = params - trainable
        
        print(f"   • Capa {i+1} ({layer_type}): {params:,} parámetros")
        print(f"     - Entrenables: {trainable:,}")
        print(f"     - No entrenables: {non_trainable:,}")
    
    print("\n✅ Análisis de parámetros completado")


# =============================================================================
# FASE 4: ENTRENAMIENTO DEL MODELO LSTM
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
    print("\n🚀 Iniciando entrenamiento del modelo LSTM...")
    
    # Entrenar modelo
    historial = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )
    
    print("✅ Entrenamiento completado")
    return historial

def graficar_historial(historial):
    """
    Grafica el historial de entrenamiento (pérdida y métrica)
    
    Args:
        historial: Historial del entrenamiento (output de model.fit)
    """
    print("\n📈 Graficando historial de entrenamiento...")
    
    try:
        # Verificar que el historial existe y tiene datos
        if not historial or not hasattr(historial, 'history'):
            print("❌ Error: Historial vacío o inválido")
            return
        
        # Verificar que las claves existen
        required_keys = ['loss', 'val_loss', 'mae', 'val_mae']
        missing_keys = [key for key in required_keys if key not in historial.history]
        
        if missing_keys:
            print(f"❌ Error: Faltan claves en historial: {missing_keys}")
            print(f"   Claves disponibles: {list(historial.history.keys())}")
            return
        
        print("   ✓ Historial válido, creando gráficos...")
        
        # Configurar matplotlib para evitar problemas en Windows
        plt.ioff()  # Desactivar modo interactivo
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pérdida
        epochs = range(1, len(historial.history['loss']) + 1)
        ax1.plot(epochs, historial.history['loss'], label='Pérdida (train)', linewidth=2, color='blue')
        ax1.plot(epochs, historial.history['val_loss'], label='Pérdida (val)', linewidth=2, color='red')
        ax1.set_title('Historial de Pérdida', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Métrica (MAE)
        ax2.plot(epochs, historial.history['mae'], label='MAE (train)', linewidth=2, color='blue')
        ax2.plot(epochs, historial.history['val_mae'], label='MAE (val)', linewidth=2, color='red')
        ax2.set_title('Historial de Métrica (MAE)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)  # Cerrar la figura explícitamente
        
        print("   ✅ Gráficas de historial completadas")
        
        # Mostrar resumen del entrenamiento
        final_train_loss = float(historial.history['loss'][-1])
        final_val_loss = float(historial.history['val_loss'][-1])
        final_train_mae = float(historial.history['mae'][-1])
        final_val_mae = float(historial.history['val_mae'][-1])
        
        print(f"\n📊 Resumen del entrenamiento:")
        print(f"   • Épocas completadas: {len(historial.history['loss'])}")
        print(f"   • Loss final (train): {final_train_loss:.4f}")
        print(f"   • Loss final (val): {final_val_loss:.4f}")
        print(f"   • MAE final (train): {final_train_mae:.4f}")
        print(f"   • MAE final (val): {final_val_mae:.4f}")
        
        # Verificar overfitting
        if final_val_loss > final_train_loss * 1.5:
            print("   ⚠️ Posible overfitting detectado (val_loss >> train_loss)")
        else:
            print("   ✅ Modelo parece estar bien generalizado")
            
    except Exception as e:
        print(f"❌ Error al crear gráficos: {str(e)}")
        print("   🔄 Continuando con el siguiente paso...")
        # Mostrar datos básicos sin gráfico
        try:
            if historial and hasattr(historial, 'history'):
                epochs = len(historial.history.get('loss', []))
                print(f"   📊 Entrenamiento completado: {epochs} épocas")
        except:
            pass

# Ejecutar entrenamiento
print("\n" + "="*70)
print("🚀 FASE 3: CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO LSTM")
print("="*70)

# Crear modelo
modelo_lstm = crear_modelo_lstm(input_shape=(LOOKBACK_WINDOW, 1))

# Configurar callbacks
callbacks = configurar_callbacks()

# Entrenar modelo
historial_modelo = entrenar_modelo(
    modelo_lstm, X_train, y_train, X_val, y_val, 
    epochs=100, batch_size=32, callbacks=callbacks
)

# Graficar historial de entrenamiento
print("\n🔍 Verificando historial de entrenamiento...")
try:
    if 'historial_modelo' in locals() and historial_modelo is not None:
        print("   ✅ Historial encontrado, procediendo a graficar...")
        graficar_historial(historial_modelo)
    else:
        print("   ❌ No se encontró historial de entrenamiento")
        print("   🔄 Continuando con el siguiente paso...")
except Exception as e:
    print(f"   ❌ Error en visualización de historial: {str(e)}")
    print("   🔄 Continuando con el siguiente paso...")

# Visualizar arquitectura del modelo
visualizar_arquitectura_modelo(modelo_lstm)

# Mostrar parámetros del modelo
mostrar_parametros_modelo(modelo_lstm)

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
    print("\n🔍 Evaluando modelo con datos de prueba...")
    
    # Evaluar modelo
    resultados = modelo.evaluate(X_test, y_test, verbose=0)
    
    # Crear diccionario de resultados
    metrica_resultados = dict(zip(modelo.metrics_names, resultados))
    
    print(f"✅ Evaluación completada")
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
    print(f"\n📊 Graficando {titulo}...")
    
    # Hacer predicciones
    y_pred = modelo.predict(X)
    
    # Invertir la normalización
    y_real_invertido = scaler.inverse_transform(y_real.reshape(-1, 1))
    y_pred_invertido = scaler.inverse_transform(y_pred)
    
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
    
    print("✅ Gráfica de predicciones completada")

# Ejecutar evaluación y prueba
print("\n" + "="*70)
print("🚀 FASE 4: EVALUACIÓN Y PRUEBA DEL MODELO LSTM")
print("="*70)

# Evaluar modelo
resultados_evaluacion = evaluar_modelo(modelo_lstm, X_test, y_test)

# Graficar predicciones
graficar_predicciones(modelo_lstm, X_test, y_test, scaler, titulo='Predicciones en Datos de Prueba')

# =============================================================================
# FASE 6: AJUSTE FINO Y OPTIMIZACIÓN DEL MODELO LSTM
# =============================================================================

def ajustar_modelo(modelo, X_train, y_train, X_val, y_val, 
                  epochs=50, batch_size=16, callbacks=None):
    """
    Ajusta el modelo LSTM con nuevos parámetros de entrenamiento
    
    Args:
        modelo: Modelo LSTM entrenado
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        epochs: Número de épocas para entrenar
        batch_size: Tamaño del batch
        callbacks: Lista de callbacks para el entrenamiento
    
    Returns:
        Historial del ajuste (historial de pérdidas y métricas)
    """
    print("\n🔧 Ajustando modelo LSTM...")
    
    # Ajustar modelo
    historial_ajuste = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )
    
    print("✅ Ajuste completado")
    return historial_ajuste

# Ejecutar ajuste fino (opcional)
# historial_ajuste = ajustar_modelo(
#     modelo_lstm, X_train, y_train, X_val, y_val, 
#     epochs=50, batch_size=16, callbacks=callbacks
# )

print("\n🎉 Proceso completado exitosamente!")
print("🔑 Modelo LSTM listo para hacer predicciones")

# =============================================================================
# VISUALIZACIONES ADICIONALES Y ANÁLISIS DETALLADO
# =============================================================================

print("\n" + "="*70)
print("🎨 VISUALIZACIONES ADICIONALES Y ANÁLISIS DETALLADO")
print("="*70)

def crear_visualizacion_entrenamiento():
    """
    Prepara función para visualizar el proceso de entrenamiento
    """
    def plot_training_history(history):
        """
        Visualiza las métricas durante el entrenamiento
        """
        plt.figure(figsize=(15, 5))
        
        # Loss
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Train Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
        plt.title('Pérdida durante entrenamiento')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # MAE
        plt.subplot(1, 3, 2)
        plt.plot(history.history['mae'], label='Train MAE', color='blue')
        plt.plot(history.history['val_mae'], label='Validation MAE', color='red')
        plt.title('Error Absoluto Medio')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning Rate (si está disponible)
        plt.subplot(1, 3, 3)
        if 'lr' in history.history:
            plt.plot(history.history['lr'], label='Learning Rate', color='green')
            plt.title('Tasa de Aprendizaje')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Learning Rate\nno disponible', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.show()
    
    return plot_training_history

# Crear función de visualización de entrenamiento
plot_training_history = crear_visualizacion_entrenamiento()

# Visualizar historial de entrenamiento (si existe)
if 'historial_modelo' in locals():
    print("\n📈 Visualizando historial de entrenamiento...")
    plot_training_history(historial_modelo)

# Visualizaciones detalladas de predicciones
def visualizar_predicciones_detalladas(modelo, X_test, y_test, scaler, n_dias=100):
    """
    Crea visualizaciones detalladas de las predicciones del modelo
    
    Args:
        modelo: Modelo LSTM entrenado
        X_test, y_test: Datos de prueba
        scaler: Objeto MinMaxScaler para desnormalizar
        n_dias: Número de días a mostrar en el gráfico detallado
    """
    print(f"\n🎨 Creando visualizaciones detalladas de predicciones...")
    
    # Hacer predicciones
    y_pred = modelo.predict(X_test, verbose=0)
    
    # Desnormalizar
    y_real_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = scaler.inverse_transform(y_pred).flatten()
    
    # Calcular métricas
    mae = mean_absolute_error(y_real_original, y_pred_original)
    mse = mean_squared_error(y_real_original, y_pred_original)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_real_original - y_pred_original) / y_real_original)) * 100
    
    # Crear figura con múltiples subgráficos
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Gráfico general de predicciones
    plt.subplot(3, 2, 1)
    plt.plot(y_real_original, label='Valores Reales', color='blue', alpha=0.8)
    plt.plot(y_pred_original, label='Predicciones', color='red', alpha=0.8)
    plt.title('Predicciones vs Valores Reales - Vista General', fontsize=14, fontweight='bold')
    plt.xlabel('Días')
    plt.ylabel('Precio (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Agregar métricas en el gráfico
    metrics_text = f'MAE: ${mae:.2f}\nRMSE: ${rmse:.2f}\nMAPE: {mape:.2f}%'
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # 2. Vista detallada (últimos n_dias)
    plt.subplot(3, 2, 2)
    dias_detalle = min(n_dias, len(y_real_original))
    plt.plot(y_real_original[-dias_detalle:], label='Valores Reales', 
             color='blue', marker='o', markersize=3, alpha=0.8)
    plt.plot(y_pred_original[-dias_detalle:], label='Predicciones', 
             color='red', marker='s', markersize=3, alpha=0.8)
    plt.title(f'Vista Detallada - Últimos {dias_detalle} días', fontsize=14, fontweight='bold')
    plt.xlabel('Días')
    plt.ylabel('Precio (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Gráfico de dispersión (Real vs Predicho)
    plt.subplot(3, 2, 3)
    plt.scatter(y_real_original, y_pred_original, alpha=0.6, color='purple')
    
    # Línea de referencia perfecta (y=x)
    min_val = min(y_real_original.min(), y_pred_original.min())
    max_val = max(y_real_original.max(), y_pred_original.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Predicción Perfecta')
    
    plt.title('Correlación: Real vs Predicho', fontsize=14, fontweight='bold')
    plt.xlabel('Precio Real (USD)')
    plt.ylabel('Precio Predicho (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calcular R²
    correlation = np.corrcoef(y_real_original, y_pred_original)[0,1]
    r_squared = correlation ** 2
    plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 4. Histograma de errores
    plt.subplot(3, 2, 4)
    errores = y_real_original - y_pred_original
    plt.hist(errores, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Distribución de Errores', fontsize=14, fontweight='bold')
    plt.xlabel('Error (Real - Predicho)')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)
    
    # Línea vertical en error = 0
    plt.axvline(x=0, color='red', linestyle='--', label='Error = 0')
    plt.legend()
    
    # 5. Errores absolutos a lo largo del tiempo
    plt.subplot(3, 2, 5)
    errores_abs = np.abs(errores)
    plt.plot(errores_abs, color='orange', alpha=0.8)
    plt.title('Errores Absolutos a lo Largo del Tiempo', fontsize=14, fontweight='bold')
    plt.xlabel('Días')
    plt.ylabel('Error Absoluto (USD)')
    plt.grid(True, alpha=0.3)
    
    # Media móvil de errores
    window = 10
    if len(errores_abs) >= window:
        media_movil = np.convolve(errores_abs, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(errores_abs)), media_movil, 
                color='red', linewidth=2, label=f'Media móvil ({window} días)')
        plt.legend()
    
    # 6. Resumen de métricas
    plt.subplot(3, 2, 6)
    plt.axis('off')
    
    # Crear tabla de métricas
    metricas_tabla = [
        ['Métrica', 'Valor'],
        ['MAE (Error Absoluto Medio)', f'${mae:.2f}'],
        ['MSE (Error Cuadrático Medio)', f'${mse:.2f}'],
        ['RMSE (Raíz del MSE)', f'${rmse:.2f}'],
        ['MAPE (Error Porcentual)', f'{mape:.2f}%'],
        ['Correlación', f'{correlation:.4f}'],
        ['R² (Coef. Determinación)', f'{r_squared:.4f}'],
        ['Datos de Prueba', f'{len(y_test)} puntos']
    ]
    
    # Mostrar tabla
    tabla = plt.table(cellText=metricas_tabla[1:], colLabels=metricas_tabla[0],
                     cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 2)
    
    # Colorear encabezados
    for i in range(len(metricas_tabla[0])):
        tabla[(0, i)].set_facecolor('#4CAF50')
        tabla[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Resumen de Métricas de Evaluación', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualizaciones detalladas completadas")
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'correlation': correlation,
        'r_squared': r_squared
    }

# Visualizar ejemplos de secuencias
def visualizar_secuencias_ejemplo(X, y, scaler, n_ejemplos=3):
    """
    Visualiza ejemplos de secuencias de entrada y sus objetivos
    
    Args:
        X: Secuencias de entrada (shape: samples, timesteps, features)
        y: Objetivos correspondientes
        scaler: Scaler para desnormalizar
        n_ejemplos: Número de ejemplos a mostrar
    """
    print(f"\n🔍 Visualizando {n_ejemplos} ejemplos de secuencias temporales...")
    
    plt.figure(figsize=(15, 5 * n_ejemplos))
    
    for i in range(min(n_ejemplos, len(X))):
        plt.subplot(n_ejemplos, 1, i+1)
        
        # Obtener secuencia y objetivo
        secuencia = X[i].flatten()  # (20 días)
        objetivo = y[i]  # (1 día siguiente)
        
        # Desnormalizar
        secuencia_original = scaler.inverse_transform(secuencia.reshape(-1, 1)).flatten()
        objetivo_original = scaler.inverse_transform([[objetivo]])[0][0]
        
        # Graficar secuencia
        dias_secuencia = range(len(secuencia_original))
        plt.plot(dias_secuencia, secuencia_original, 
                'b-o', label='Secuencia de entrada (20 días)', markersize=4)
        
        # Graficar objetivo
        plt.plot([len(secuencia_original)], [objetivo_original], 
                'ro', markersize=8, label='Objetivo (día siguiente)')
        
        plt.title(f'Ejemplo {i+1}: Secuencia → Predicción', fontsize=12, fontweight='bold')
        plt.xlabel('Días')
        plt.ylabel('Precio (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Conectar último punto de secuencia con objetivo
        plt.plot([len(secuencia_original)-1, len(secuencia_original)], 
                [secuencia_original[-1], objetivo_original], 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualización de secuencias completada")

# Ejecutar visualizaciones adicionales
print("\n" + "="*70)
print("🎯 EJECUTANDO VISUALIZACIONES ADICIONALES")
print("="*70)

# Visualizar historial de entrenamiento detallado
if 'historial_modelo' in locals():
    print("\n📈 Visualizando historial de entrenamiento detallado...")
    plot_training_history(historial_modelo)

# Visualizaciones detalladas de predicciones
print("\n🔍 Generando análisis detallado de predicciones...")
metricas_detalladas = visualizar_predicciones_detalladas(
    modelo_lstm, X_test, y_test, scaler, n_dias=50
)

# Visualizar ejemplos de secuencias
print("\n👁️ Mostrando ejemplos de secuencias temporales...")
visualizar_secuencias_ejemplo(X_test, y_test, scaler, n_ejemplos=3)

print(f"\n🎯 Análisis completo finalizado!")
print(f"📊 Métricas principales:")
for metrica, valor in metricas_detalladas.items():
    if metrica in ['mae', 'rmse']:
        print(f"   • {metrica.upper()}: ${valor:.2f}")
    elif metrica == 'mape':
        print(f"   • {metrica.upper()}: {valor:.2f}%")
    else:
        print(f"   • {metrica.upper()}: {valor:.4f}")

print(f"\n✅ Modelo LSTM completamente implementado y evaluado!")
print(f"🔄 Próximo paso: Optimización con Algoritmos Genéticos")

# =============================================================================
# PREDICCIÓN FINAL: ¿QUÉ PREDICE EL MODELO PARA YPF?
# =============================================================================

def hacer_prediccion_final(modelo, datos_recientes, scaler, lookback_window=20):
    """
    Hace una predicción específica para el próximo día de la acción YPF
    usando los datos más recientes disponibles
    
    Args:
        modelo: Modelo LSTM entrenado
        datos_recientes: Últimos datos de la serie temporal
        scaler: Objeto MinMaxScaler para desnormalizar
        lookback_window: Ventana temporal del modelo
    
    Returns:
        dict: Predicción detallada con contexto
    """
    print("\n" + "="*70)
    print("🎯 PREDICCIÓN FINAL: ¿QUÉ PREDICE EL MODELO PARA YPF?")
    print("="*70)
    
    # Obtener los últimos datos disponibles
    ultimos_datos = datos_recientes.tail(lookback_window)
    precio_actual = float(ultimos_datos.iloc[-1])
    fecha_actual = ultimos_datos.index[-1].date()
    
    print(f"\n📊 CONTEXTO ACTUAL:")
    print(f"   • Fecha del último precio: {fecha_actual}")
    print(f"   • Precio actual de YPF: ${precio_actual:.2f}")
    print(f"   • Ventana de análisis: {lookback_window} días anteriores")
    
    # Normalizar los últimos datos
    ultimos_normalizados = scaler.transform(ultimos_datos.values.reshape(-1, 1))
    
    # Preparar entrada para el modelo
    X_prediccion = ultimos_normalizados.reshape(1, lookback_window, 1)
    
    # Hacer predicción
    prediccion_normalizada = modelo.predict(X_prediccion, verbose=0)
    
    # Desnormalizar predicción
    precio_predicho = float(scaler.inverse_transform(prediccion_normalizada)[0][0])
    
    # Calcular variaciones
    variacion_absoluta = precio_predicho - precio_actual
    variacion_porcentual = (variacion_absoluta / precio_actual) * 100
    
    # Determinar tendencia
    if variacion_porcentual > 2:
        tendencia = "📈 FUERTE ALZA"
        interpretacion = "El modelo predice un aumento significativo"
    elif variacion_porcentual > 0.5:
        tendencia = "↗️ ALZA MODERADA"
        interpretacion = "El modelo predice un aumento moderado"
    elif variacion_porcentual > -0.5:
        tendencia = "➡️ ESTABLE"
        interpretacion = "El modelo predice estabilidad de precios"
    elif variacion_porcentual > -2:
        tendencia = "↘️ BAJA MODERADA"
        interpretacion = "El modelo predice una disminución moderada"
    else:
        tendencia = "📉 FUERTE BAJA"
        interpretacion = "El modelo predice una disminución significativa"
    
    print(f"\n🔮 PREDICCIÓN PARA EL PRÓXIMO DÍA DE TRADING:")
    print(f"   • Precio predicho: ${precio_predicho:.2f}")
    print(f"   • Variación absoluta: ${variacion_absoluta:+.2f}")
    print(f"   • Variación porcentual: {variacion_porcentual:+.2f}%")
    print(f"   • Tendencia: {tendencia}")
    print(f"   • Interpretación: {interpretacion}")
    
    # Análisis de confianza basado en el rendimiento del modelo
    if 'metricas_detalladas' in globals():
        mape = metricas_detalladas.get('mape', 0)
        r_squared = metricas_detalladas.get('r_squared', 0)
        
        if mape < 5 and r_squared > 0.7:
            confianza = "ALTA"
            confianza_emoji = "🟢"
        elif mape < 10 and r_squared > 0.5:
            confianza = "MEDIA"
            confianza_emoji = "🟡"
        else:
            confianza = "BAJA"
            confianza_emoji = "🔴"
        
        print(f"\n🎯 NIVEL DE CONFIANZA: {confianza_emoji} {confianza}")
        print(f"   • MAPE del modelo: {mape:.2f}%")
        print(f"   • R² del modelo: {r_squared:.4f}")
    
    # Mostrar contexto histórico reciente
    print(f"\n📈 CONTEXTO DE LOS ÚLTIMOS 5 DÍAS:")
    ultimos_5_dias = ultimos_datos.tail(5)
    for i in range(len(ultimos_5_dias)):
        fecha = ultimos_5_dias.index[i]
        precio = ultimos_5_dias.iloc[i]
        es_ultimo = i == len(ultimos_5_dias) - 1
        emoji = "👉" if es_ultimo else "  "
        # Manejar diferentes tipos de fecha
        if hasattr(fecha, 'date'):
            fecha_str = fecha.date()
        else:
            fecha_str = str(fecha)
        print(f"   {emoji} {fecha_str}: ${float(precio):.2f}")
    
    return {
        'precio_actual': precio_actual,
        'precio_predicho': precio_predicho,
        'variacion_absoluta': variacion_absoluta,
        'variacion_porcentual': variacion_porcentual,
        'tendencia': tendencia,
        'interpretacion': interpretacion,
        'fecha_prediccion': fecha_actual
    }

def mostrar_escenarios_prediccion(prediccion_base, precio_actual):
    """
    Muestra diferentes escenarios basados en la predicción
    """
    print(f"\n📋 ESCENARIOS DE INVERSIÓN:")
    
    precio_predicho = prediccion_base['precio_predicho']
    variacion_pct = prediccion_base['variacion_porcentual']
    
    # Escenarios de inversión
    inversion_ejemplo = 10000  # $10,000 como ejemplo
    acciones_posibles = inversion_ejemplo / precio_actual
    
    print(f"\n💰 Ejemplo con inversión de ${inversion_ejemplo:,.0f}:")
    print(f"   • Acciones que se pueden comprar: {acciones_posibles:.0f}")
    
    if variacion_pct > 0:
        ganancia_potencial = acciones_posibles * prediccion_base['variacion_absoluta']
        print(f"   • Ganancia potencial: ${ganancia_potencial:+.2f}")
        print(f"   • ROI esperado: {variacion_pct:+.2f}%")
        print(f"   • 📈 Estrategia sugerida: COMPRAR")
    else:
        perdida_potencial = acciones_posibles * prediccion_base['variacion_absoluta']
        print(f"   • Pérdida potencial: ${perdida_potencial:+.2f}")
        print(f"   • ROI esperado: {variacion_pct:+.2f}%")
        print(f"   • 📉 Estrategia sugerida: MANTENER o VENDER")
    
    print(f"\n⚠️ DISCLAIMER:")
    print(f"   • Esta es una predicción basada en datos históricos")
    print(f"   • Los mercados financieros son impredecibles")
    print(f"   • Siempre consulte con un asesor financiero")
    print(f"   • No invierta más de lo que puede permitirse perder")

def visualizar_prediccion_final(datos_historicos, prediccion, scaler):
    """
    Crea una visualización específica de la predicción final
    """
    print(f"\n📊 Creando visualización de la predicción final...")
    
    # Preparar datos para visualización
    ultimos_30_dias = datos_historicos.tail(30)
    fechas = ultimos_30_dias.index
    precios = ultimos_30_dias.values
    
    # Crear fecha para la predicción (próximo día de trading)
    from datetime import timedelta
    fecha_prediccion = fechas[-1] + timedelta(days=1)
    precio_predicho = prediccion['precio_predicho']
    precio_actual = float(precios[-1])  # Convertir a escalar
    
    plt.figure(figsize=(14, 8))
    
    # Gráfico principal
    plt.plot(fechas, precios, 'b-', linewidth=2, label='Precios Históricos', marker='o', markersize=4)
    
    # Punto actual
    plt.plot(fechas[-1], precio_actual, 'go', markersize=10, label=f'Precio Actual: ${precio_actual:.2f}')
    
    # Predicción
    plt.plot(fecha_prediccion, precio_predicho, 'rs', markersize=12, 
             label=f'Predicción: ${precio_predicho:.2f}')
    
    # Línea conectora
    plt.plot([fechas[-1], fecha_prediccion], [precio_actual, precio_predicho], 
             'r--', alpha=0.7, linewidth=2)
    
    # Formateo
    plt.title('YPF - Predicción para el Próximo Día de Trading', fontsize=16, fontweight='bold')
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Precio (USD)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Añadir información de la predicción
    variacion = prediccion['variacion_porcentual']
    color_texto = 'green' if variacion > 0 else 'red'
    plt.text(0.02, 0.98, 
             f"Variación Predicha: {variacion:+.2f}%\n{prediccion['tendencia']}", 
             transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', color=color_texto,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualización de predicción final completada")

# EJECUTAR PREDICCIÓN FINAL
print("\n" + "="*70)
print("🚀 EJECUTANDO PREDICCIÓN FINAL PARA YPF")
print("="*70)

# Hacer predicción para el próximo día
prediccion_final = hacer_prediccion_final(modelo_lstm, datos_ypf, scaler, LOOKBACK_WINDOW)

# Mostrar escenarios de inversión
mostrar_escenarios_prediccion(prediccion_final, prediccion_final['precio_actual'])

# Visualizar predicción
visualizar_prediccion_final(datos_ypf, prediccion_final, scaler)

# RESUMEN EJECUTIVO FINAL
print("\n" + "="*70)
print("📋 RESUMEN EJECUTIVO - PREDICCIÓN LSTM PARA YPF")
print("="*70)

print(f"""
🎯 PREDICCIÓN PRINCIPAL:
   El modelo LSTM predice que YPF cotizará a ${prediccion_final['precio_predicho']:.2f} 
   en el próximo día de trading, representando una variación de {prediccion_final['variacion_porcentual']:+.2f}%

📊 FUNDAMENTO TÉCNICO:
   • Modelo entrenado con {NUM_DATOS} puntos históricos
   • Arquitectura: 2 capas LSTM con 50 unidades cada una
   • Ventana temporal: {LOOKBACK_WINDOW} días de historia para predicción
   • Accuracy del modelo: R² = {metricas_detalladas.get('r_squared', 0):.4f}

💡 INTERPRETACIÓN:
   {prediccion_final['interpretacion']}

⚠️ RIESGOS Y LIMITACIONES:
   • Modelo basado en patrones históricos
   • No considera eventos fundamentales o noticias
   • Mercados pueden ser impredecibles en el corto plazo
   • Error promedio del modelo: ±{metricas_detalladas.get('mape', 0):.1f}%

🔮 PRÓXIMOS PASOS:
   ✅ Modelo LSTM base implementado
   🔄 Pendiente: Optimización con Algoritmos Genéticos
   🎯 Objetivo: Mejorar precisión mediante evolución de hiperparámetros
""")

#print(f"\n🎉 IMPLEMENTACIÓN LSTM COMPLETADA CON PREDICCIÓN CLARA!")
#print(f"🚀 Sistema listo para la fase de optimización con Algoritmos Genéticos")

