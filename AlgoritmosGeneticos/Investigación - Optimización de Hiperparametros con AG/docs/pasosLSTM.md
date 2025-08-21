### Framework de Deep Learning**
**Fecha:** 5 de Agosto, 2025  
**Problema:** ¿TensorFlow o PyTorch para implementar LSTM?

**Decisión:** **TensorFlow con Keras**

**Justificación:**
- ✅ **Facilidad para principiantes:** API más intuitiva con Keras
- ✅ **Documentación superior:** Mejor documentación para LSTM en series temporales
- ✅ **Integración con AG:** Más fácil integración con algoritmos genéticos para optimización de hiperparámetros
- ✅ **Ecosystem completo:** TensorBoard, mejor integración con numpy
- ✅ **Menos código:** Implementación más directa para nuestro caso de uso

**Alternativa descartada:** PyTorch (mejor para investigación avanzada, pero más complejo para nuestro nivel)

---
## **🚀 FASES DE IMPLEMENTACIÓN**

### **Fase 1: Preparación y Configuración** 
- [x] Decisión de framework
- [ ] Importaciones y configuración inicial
- [ ] Carga y exploración de datos
- [ ] Preparación de datos para LSTM

### **Fase 2: Construcción del Modelo**
- [ ] Diseño de arquitectura LSTM
- [ ] Configuración de hiperparámetros base
- [ ] Compilación del modelo
🔄 1. Normalización (0-1):
¿Por qué? Las LSTM funcionan mejor con datos pequeños
MinMaxScaler: Convierte precios ($10-$50) → valores (0.0-1.0)
Guardamos el scaler: Para "desnormalizar" las predicciones después
🏗️ 2. Secuencias Temporales:
Ventanas deslizantes: Últimos 20 días → predice día 21
Formato LSTM: Reshape a (samples, timesteps, features)
Ejemplo práctico:
✂️ 3. División Temporal (MUY IMPORTANTE):
NO aleatoria: Mantenemos orden temporal
Train: Datos más antiguos (pasado)
Validation: Datos intermedios
Test: Datos más recientes (futuro)
📊 4. Visualización:
Antes vs. Después: Vemos el efecto de la normalización
Control de calidad: Verificamos que todo esté correcto
💡 Conceptos clave explicados:
¿Por qué normalizar 0-1?

LSTM aprende mejor con números pequeños
Evita que precios altos ($50) dominen sobre precios bajos ($10)
Acelera el entrenamiento
¿Por qué ventanas deslizantes?

LSTM necesita "contexto temporal"
20 días de historia para predecir el día 21
Es como decirle: "basándote en estas 20 observaciones, ¿qué pasará mañana?"

🎯 Diferencia entre Validación y Test
📊 División de datos típica:

Total: 1000 datos de YPF├── Train (70%): 700 datos - Entrenar el modelo├── Validation (15%): 150 datos - Ajustar hiperparámetros  └── Test (15%): 150 datos - Evaluación final
🔍 ¿Para qué sirve cada conjunto?
🏋️ Datos de ENTRENAMIENTO (Train):
Propósito: El modelo "aprende" de estos datos
Analogía: Como estudiar con libros de texto
¿Qué hace la LSTM? Ajusta sus pesos internos para minimizar errores
⚖️ Datos de VALIDACIÓN:
Propósito: Ajustar hiperparámetros y evitar overfitting
Analogía: Como hacer exámenes de práctica
¿Qué hacemos? Probar diferentes configuraciones del modelo
🎓 Datos de TEST:
Propósito: Evaluación final e imparcial del modelo
Analogía: Como el examen final real
¿Qué hacemos? Medir el rendimiento real del modelo


### **Fase 3: Entrenamiento y Evaluación**
- [ ] División de datos (train/val/test)
- [ ] Entrenamiento del modelo
- [ ] Evaluación de métricas

¿Por qué cada capa?

LSTM Layer 1: Aprende patrones temporales básicos (tendencias, ciclos)
Dropout: Evita overfitting (memorizar en lugar de aprender)
LSTM Layer 2: Aprende patrones más complejos y abstractos
Dense Layer: Convierte el "conocimiento" LSTM en una predicción numérica

Hiperparámetros principales:

LSTM Units	50	Neuronas por capa LSTM (más = más memoria)

Num Layers	2	Capas LSTM apiladas (más = más complejidad)

Dropout Rate	0.2	% de neuronas que se "apagan" (previene overfitting)

Learning Rate	0.001	Velocidad de aprendizaje

Batch Size	32	Datos procesados juntos

Epochs	100	Veces que ve todos los datos

3. Compilación del Modelo
Decisiones importantes:
Loss Function (Función de Pérdida):

MSE (Mean Squared Error): Para predicción de precios
Penaliza más los errores grandes
Optimizer (Optimizador):

Adam: Adapta la velocidad de aprendizaje automáticamente
Mejor que SGD para nuestro caso
Métricas:

MAE: Error absoluto promedio (más interpretable)
MSE: Para seguimiento durante entrenamiento

🚨 4. Callbacks (Controles Inteligentes)
Early Stopping:

EarlyStopping(    monitor='val_loss',    # Vigila pérdida en validación    patience=10,           # Espera 10 epochs sin mejora    restore_best_weights=True  # Mantiene los mejores pesos)
¿Por qué? Evita entrenar de más (overfitting)

Reduce Learning Rate:

ReduceLROnPlateau(    monitor='val_loss',    factor=0.5,           # Reduce LR a la mitad    patience=5            # Si no mejora en 5 epochs)
¿Por qué? Ajuste fino cuando se estanca


### **Fase 4: Optimización con AG**
- [ ] Implementación del algoritmo genético
- [ ] Definición del espacio de búsqueda
- [ ] Optimización de hiperparámetros

---