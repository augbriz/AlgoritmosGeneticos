# **Guía de Implementación LSTM para Predicción de Acciones YPF**
## **Proyecto: Modelado del comportamiento caótico de YPF mediante Algoritmos Genéticos y Redes LSTM**

---

## **1. ¿QUÉ SON LAS REDES LSTM Y POR QUÉ LAS NECESITAMOS?**

### **1.1 El Problema con los Precios de Acciones**

Imagina que quieres predecir si el precio de YPF va a subir o bajar mañana. ¿Qué información usarías?

- El precio de hoy
- El precio de ayer  
- El precio de la semana pasada
- ¿Y el precio de hace un mes? ¿Sigue siendo relevante?

Los precios de las acciones tienen **memoria**: lo que pasó en el pasado influye en lo que va a pasar mañana. Pero hay un problema: **¿cuánto del pasado debemos recordar?**

### **1.2 ¿Qué es una Red LSTM?**

**LSTM** significa "**Long Short-Term Memory**" (Memoria de Largo y Corto Plazo). Es un tipo especial de red neuronal que fue diseñada específicamente para **recordar información importante del pasado** y **olvidar información irrelevante**.

Piensa en una LSTM como un **asistente muy inteligente** que:
- **Recuerda** los eventos importantes del pasado que pueden afectar el futuro
- **Olvida** los ruidos y fluctuaciones menores que no son importantes
- **Aprende** qué información del pasado es realmente útil para hacer predicciones

### **1.3 ¿Cómo Funciona una LSTM? (Explicación Simple)**

Una red LSTM tiene tres "**puertas**" que funcionan como filtros inteligentes:

#### **🚪 Puerta del Olvido (Forget Gate)**
- **¿Qué hace?** Decide qué información del pasado ya no es útil y la descarta
- **Ejemplo con YPF:** Si hubo una caída hace 6 meses por una noticia específica que ya no es relevante, la puerta del olvido la elimina de la memoria

#### **🚪 Puerta de Entrada (Input Gate)** 
- **¿Qué hace?** Decide qué nueva información del presente es importante y vale la pena recordar
- **Ejemplo con YPF:** Si hoy hay un anuncio importante sobre nuevas reservas de petróleo, esta puerta decide que es información valiosa para recordar

#### **🚪 Puerta de Salida (Output Gate)**
- **¿Qué hace?** Decide qué información usar para hacer la predicción de mañana
- **Ejemplo con YPF:** Combina la información relevante del pasado con la información nueva para predecir el precio de mañana

### **1.4 ¿Por Qué LSTM es Perfecta para YPF?**

Las acciones de YPF tienen características especiales que hacen que LSTM sea ideal:

**🔄 Patrones que se Repiten**
- YPF sigue ciertos patrones estacionales (ej: mayor demanda en invierno)
- LSTM puede aprender y recordar estos patrones

**📈 Volatilidad Compleja**
- Los precios pueden tener subidas y bajadas bruscas
- LSTM puede distinguir entre movimientos importantes y ruido del mercado

**🌍 Influencias Múltiples**
- Precio del petróleo internacional
- Política argentina
- Resultados trimestrales de la empresa
- LSTM puede recordar cómo estos factores afectaron en el pasado

### **1.5 Datos Necesarios: ¿Son Suficientes 1000 Puntos?**

Para nuestro proyecto con YPF, **1000 datos históricos (aproximadamente 4 años) son SUFICIENTES** porque:

✅ **Captura Ciclos Importantes:** 4 años incluyen diferentes crisis, bonanzas y ciclos políticos
✅ **Evita Sobreajuste:** Con series caóticas como YPF, demasiados datos pueden incluir información obsoleta
✅ **División Adecuada:** Permite dividir los datos en entrenamiento (700), validación (150) y prueba (150)
✅ **Comportamiento Caótico:** Para sistemas caóticos, la información muy antigua pierde relevancia

---

## **2. GUÍA DE IMPLEMENTACIÓN LSTM**

### **2.1 Fase 1: Preparación de los Datos**

**🎯 Objetivo:** Convertir los precios históricos de YPF en un formato que la LSTM pueda entender

**¿Qué haremos?**
1. **Limpiar los datos:** Eliminar días sin trading, datos faltantes, errores
2. **Normalizar:** Convertir todos los precios a una escala de 0 a 1 para facilitar el aprendizaje
3. **Crear secuencias:** En lugar de datos individuales, crear "ventanas" de tiempo
   - Ejemplo: "Los últimos 20 días de precios" → "Precio de mañana"
4. **Dividir el dataset:** Separar en entrenamiento, validación y prueba

**¿Por qué estas ventanas?**
- Si usamos los últimos 20 días para predecir el día 21, la LSTM aprende patrones como:
  - "Cuando hay 3 días consecutivos de subida, suele haber una corrección"
  - "Los lunes después de bajas fuertes del viernes suelen recuperarse"

### **2.2 Fase 2: Construcción del Modelo LSTM**

**🎯 Objetivo:** Crear la arquitectura de la red neuronal

**Componentes principales:**

#### **📊 Capa de Entrada**
- Recibe las secuencias de precios (ej: últimos 20 días)
- Es como mostrarle a la red el "contexto" para hacer la predicción

#### **🧠 Capas LSTM**
- El "cerebro" que procesa la información temporal
- Pueden ser una o varias capas apiladas
- Más capas = mayor capacidad de aprender patrones complejos (pero también más riesgo de sobreajuste)

#### **🎯 Capa de Salida**
- Produce la predicción final: el precio estimado para mañana
- Es un solo número (el precio predicho)

**¿Cómo decide cuántas capas usar?**
Esto lo optimizaremos después con algoritmos genéticos, pero la idea general es:
- **1 capa LSTM:** Para patrones simples
- **2-3 capas LSTM:** Para patrones más complejos (nuestro caso probable)
- **Más de 3 capas:** Raramente necesario, puede causar sobreentrenamiento

### **2.3 Fase 3: Entrenamiento del Modelo**

**🎯 Objetivo:** Enseñar a la LSTM a reconocer patrones en los datos de YPF

**¿Cómo aprende la LSTM?**
1. **Predicción:** La red ve los primeros 20 días y predice el día 21
2. **Comparación:** Compara su predicción con el precio real del día 21
3. **Ajuste:** Si se equivocó, ajusta sus parámetros internos para mejorar
4. **Repetición:** Hace esto miles de veces con diferentes secuencias

**Controles importantes:**
- **Early Stopping:** Si la red deja de mejorar, para el entrenamiento automáticamente
- **Validación:** Usa datos que nunca vio para verificar que no está "memorizando"

---

## **3. OPTIMIZACIÓN CON ALGORITMOS GENÉTICOS**

### **3.1 ¿Qué son los Hiperparámetros?**

Los **hiperparámetros** son las "configuraciones" del modelo LSTM que nosotros tenemos que decidir antes del entrenamiento. Es como elegir las especificaciones de un auto antes de comprarlo.

**🔧 Principales Hiperparámetros para LSTM:**

#### **🏗️ Arquitectura del Modelo**
- **Número de capas LSTM:** ¿1, 2 o 3 capas?
- **Número de neuronas por capa:** ¿32, 64, 128 neuronas?
- **Ventana temporal:** ¿Usar 10, 20 o 30 días pasados para predecir?

#### **🎓 Parámetros de Aprendizaje**
- **Tasa de aprendizaje:** ¿Qué tan rápido aprende? (muy rápido puede ser inestable, muy lento puede ser ineficiente)
- **Tamaño de lote (batch size):** ¿Cuántos ejemplos ve antes de ajustar?
- **Épocas:** ¿Cuántas veces repasa todos los datos?

#### **🛡️ Regularización**
- **Dropout:** ¿Qué porcentaje de neuronas "apagar" para evitar sobreajuste?
- **Regularización L2:** ¿Cuánto penalizar la complejidad del modelo?

### **3.2 El Problema de la Optimización**

Imagina que tienes que encontrar la **mejor combinación** de todos estos parámetros:
- 3 opciones de capas × 4 opciones de neuronas × 5 opciones de ventana temporal × 4 opciones de tasa de aprendizaje = **240 combinaciones diferentes**

Y esto es solo con unos pocos parámetros. En realidad, hay **miles de combinaciones posibles**.

**¿Cómo encontrar la mejor?**
- **Método manual:** Probar uno por uno → Tomaría meses
- **Búsqueda en grilla:** Probar todas las combinaciones → Tomaría semanas
- **Algoritmos Genéticos:** Evolución inteligente → Días

### **3.3 ¿Cómo Funcionan los Algoritmos Genéticos para Optimización?**

Los **Algoritmos Genéticos** imitan la evolución natural para encontrar las mejores configuraciones:

#### **🧬 Paso 1: Población Inicial**
- Crear 20-50 configuraciones LSTM diferentes (aleatorias)
- Cada configuración es un "individuo" con su propio "ADN" (hiperparámetros)

#### **🏃 Paso 2: Evaluación (Fitness)**
- Entrenar cada configuración LSTM con los datos de YPF
- Medir qué tan bien predice (ej: % de aciertos direccionales)
- Asignar una "puntuación de fitness" a cada configuración

#### **❤️ Paso 3: Selección**
- Las configuraciones que predicen mejor tienen más probabilidad de "reproducirse"
- Las configuraciones malas tienen menos probabilidad de continuar

#### **🧬 Paso 4: Cruce (Crossover)**
- Combinar dos configuraciones "padres" exitosas para crear "hijos"
- Ejemplo: Padre A (2 capas, 64 neuronas) + Padre B (3 capas, 32 neuronas) = Hijo (2 capas, 32 neuronas)

#### **⚡ Paso 5: Mutación**
- Cambiar aleatoriamente algunos parámetros de los "hijos"
- Esto introduce variabilidad y puede descubrir mejores configuraciones

#### **🔄 Paso 6: Nueva Generación**
- Repetir el proceso con la nueva población
- Después de 20-50 generaciones, converge hacia la mejor configuración

### **3.4 ¿Por Qué Funcionan Mejor los AG para LSTM?**

**🎯 Exploración Inteligente**
- No prueban todas las combinaciones, sino que se enfocan en las más prometedoras
- Balancean exploración (probar cosas nuevas) vs. explotación (mejorar lo bueno)

**🔄 Adaptabilidad**
- Si una estrategia funciona, la enfatizan más
- Si una estrategia falla, la abandonan rápidamente

**🎪 Diversidad**
- Mantienen múltiples enfoques simultáneamente
- Evitan quedarse atascados en óptimos locales

---

## **4. MÉTRICAS DE EVALUACIÓN**

### **4.1 ¿Cómo Sabemos si Nuestro Modelo es Bueno?**

Con las acciones, no solo importa **qué tan cerca estamos del precio real**, sino también **si acertamos la dirección** (¿sube o baja?).

### **4.2 Métricas Principales**

#### **📊 Métricas de Precisión Numérica**

**🎯 Error Cuadrático Medio (MSE/RMSE)**
- **¿Qué mide?** Qué tan lejos están nuestras predicciones del precio real
- **¿Por qué importa?** Si predecimos $15 y el precio real es $10, el error es grande
- **Interpretación:** Más bajo = mejor

**📏 Error Absoluto Medio (MAE)**
- **¿Qué mide?** El error promedio en términos absolutos
- **¿Por qué importa?** Más fácil de interpretar (en pesos argentinos)
- **Ejemplo:** "En promedio, nos equivocamos por $2 por acción"

#### **🧭 Métricas Direccionales (Las Más Importantes para Trading)**

**📈 Precisión Direccional**
- **¿Qué mide?** ¿En qué porcentaje de casos acertamos si el precio sube o baja?
- **¿Por qué es crucial?** Para trading, importa más la dirección que el valor exacto
- **Interpretación:** "Acertamos la dirección en el 65% de los casos"

**💰 Retorno de Inversión Simulado**
- **¿Qué mide?** Si siguiéramos las predicciones del modelo, ¿ganaríamos dinero?
- **¿Cómo funciona?** 
  - Si el modelo predice subida → Compramos
  - Si el modelo predice bajada → Vendemos o no compramos
- **Interpretación:** "Siguiendo el modelo, habríamos ganado 15% en 6 meses"

### **4.3 ¿Cómo Medimos la Eficiencia de los Algoritmos Genéticos?**

#### **📈 Evolución del Fitness**
- **¿Qué observamos?** Cómo mejora la mejor configuración generación tras generación
- **Indicadores de éxito:**
  - La puntuación del mejor individuo aumenta con el tiempo
  - La puntuación se estabiliza (converge) después de varias generaciones
  - No hay mejoras por muchas generaciones seguidas

#### **🎯 Comparación con Métodos Tradicionales**
- **Búsqueda aleatoria:** ¿Los AG encuentran mejores configuraciones que elegir al azar?
- **Configuración manual:** ¿Los AG superan las configuraciones elegidas por expertos?
- **Tiempo de convergencia:** ¿Cuántas generaciones necesita para encontrar una buena solución?

#### **🔍 Diversidad de la Población**
- **¿Qué medimos?** Qué tan diferentes son las configuraciones en cada generación
- **Indicador saludable:** Mantener diversidad evita convergencia prematura
- **Señal de alerta:** Si todas las configuraciones se vuelven muy similares muy rápido

### **4.4 Validación Robusta**

#### **📅 Validación Temporal**
- **¿Por qué especial?** Con series temporales, NO podemos usar validación cruzada normal
- **¿Cómo lo hacemos?** Entrenamos con datos del pasado, validamos con datos del futuro
- **Ejemplo:** Entrenar con 2020-2022, validar con 2023, probar con 2024

#### **🎲 Múltiples Semillas**
- **¿Por qué?** Los algoritmos genéticos tienen componentes aleatorios
- **¿Cómo?** Correr el mismo experimento 5-10 veces con diferentes semillas aleatorias
- **¿Qué buscamos?** Resultados consistentes entre corridas

---

## **5. PLAN DE TRABAJO PASO A PASO**

### **1: Entender y Preparar los Datos**
- Análisis exploratorio de la serie de YPF
- Implementar la preparación de datos
- Crear las ventanas temporales
- Dividir en entrenamiento/validación/prueba

### **2: Construir LSTM Básica**
- Implementar el modelo LSTM con configuración fija
- Entrenarlo y evaluar resultados base
- Entender cómo se comporta con datos de YPF

### **3: Implementar Algoritmos Genéticos**
- Definir el espacio de hiperparámetros
- Implementar el algoritmo genético
- Correr la optimización y analizar resultados

### **4: Evaluación Final**
- Probar el mejor modelo en datos de test
- Comparar con métodos benchmark
- Analizar resultados y documentar conclusiones


✅ **Qué son las LSTM** y por qué son perfectas para predecir precios de YPF
✅ **Cómo funcionan** sin entrar en complicaciones matemáticas  
✅ **Qué son los hiperparámetros** y por qué necesitamos optimizarlos
✅ **Cómo los algoritmos genéticos** encuentran las mejores configuraciones automáticamente
✅ **Cómo medir el éxito** tanto del modelo como de la optimización




