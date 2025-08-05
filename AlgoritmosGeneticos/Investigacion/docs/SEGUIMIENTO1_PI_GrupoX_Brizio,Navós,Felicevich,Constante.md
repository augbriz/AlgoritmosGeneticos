## **SEGUIMIENTO Nº 1 – PROYECTO DE INVESTIGACIÓN FECHA: 10/06/2025**

 Denominación del futuro proyecto de investigación: 

 “**Modelado del comportamiento caótico de las acciones de YPF mediante algoritmos genéticos y redes neuronales recurrentes”**

Situación problemática   

 Los mercados bursátiles presentan dinámicas altamente no lineales y sensibles a las condiciones iniciales. En el contexto argentino, las acciones de YPF, empresa energética emblemática y de alto volumen de negociación, refleja tanto factores macroeconómicos locales como variaciones globales en el precio de los commodities. Estas particularidades la convierten en un caso de estudio idóneo para aplicar métricas de la Teoría del Caos, y para evaluar arquitecturas de aprendizaje profundo capaces de capturar dichas complejidades. Comprender estos patrones subyacentes resulta esencial para mejorar la toma de decisiones de inversores y analistas en mercados emergentes.

Problema  

  ¿Puede la combinación de herramientas de la Teoría del Caos con modelos de aprendizaje profundo, optimizados mediante algoritmos genéticos, mejorar la capacidad de predecir la dirección o el rango de variación diaria del precio de la acción de YPF?

Objetivos de la futura investigación

   • **Objetivo general**    Evaluar la viabilidad de un enfoque híbrido que integre indicadores caóticos y redes neuronales recurrentes, optimizadas mediante algoritmos genéticos, para predecir el comportamiento diario de la acción de YPF.

   • **Objetivos específicos**       

1\. Caracterizar la naturaleza caótica de la serie temporal de precios de YPF calculando indicadores como el exponente de Lyapunov.     

 2\. Construir y depurar un conjunto de datos histórico de precios y volumen de YPF. Los precios diarios, el volumen y los indicadores técnicos de YPF se obtendrán inicialmente a través de la API **yfinance** (fuente Yahoo Finance). Se contrastará la consistencia con datos oficiales de BYMA o servicios como Alpha Vantage. 

3\. **Diseñar alguna de las arquitecturas de aprendizaje profundo orientadas a series temporales:** una red neuronal recurrente simple (RNN) o una Long Short‑Term Memory (LSTM). Ambas están diseñadas específicamente para procesar secuencias y mantener información contextual a lo largo del tiempo. Esta capacidad las convierte en herramientas clave para modelar dependencias temporales y detectar patrones en series financieras como las de precios de acciones. La elección final entre la arquitectura RNN o LSTM dependerá de la capacidad de cada modelo para capturar dependencias de corto y largo plazo, lo que se reflejará en la fase de validación.

4\. Implementar un algoritmo genético para **optimizar hiperparámetros** clave de los modelos (número de capas, neuronas, tasa de aprendizaje, ventanas temporales, etc.).       

5\. **Evaluar el desempeño** predictivo mediante métricas de error (RMSE, MAE) y métricas de dirección sobre un conjunto de prueba.  

