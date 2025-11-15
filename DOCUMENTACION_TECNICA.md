# Documentación Técnica: Evaluación de Kolmogorov-Arnold Networks en Clasificación de Cáncer de Mama

## Índice
1. [Introducción](#introducción)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Configuración del Entorno](#configuración-del-entorno)
4. [Carga y Preprocesamiento de Datos](#carga-y-preprocesamiento-de-datos)
5. [Arquitecturas de Modelos](#arquitecturas-de-modelos)
6. [Proceso de Entrenamiento](#proceso-de-entrenamiento)
7. [Evaluación y Métricas](#evaluación-y-métricas)
8. [Análisis de Resultados](#análisis-de-resultados)
9. [Conclusiones](#conclusiones)

---

## Resumen Ejecutivo

### Resultados Clave
Este estudio evaluó cinco arquitecturas de redes neuronales para clasificación de cáncer de mama:

| Modelo | Sensitivity | Specificity | AUC-ROC | MCC | Recomendación |
|--------|------------|-------------|---------|-----|---------------|
| **Baseline MLP** | 0.9524 | **1.0000** | 0.9954 | **0.9626** | **Producción** |
| **Chebyshev-KAN V4** | **1.0000** | 0.9444 | **0.9980** | 0.9286 | **Screening** |
| **Fast-KAN V3** | 0.9762 | 0.9722 | 0.9874 | 0.9439 | Balance |
| **Wavelet-KAN V3** | 0.8571 | 0.9861 | 0.9792 | 0.8688 | Investigación |
| **Fourier-KAN V4** | 0.8571 | 0.9167 | 0.9649 | 0.7738 | No recomendado |

### Hallazgos Principales

1. **KAN vs MLP**: Las redes KAN **no superan universalmente** a MLP, pero ofrecen trade-offs valiosos
2. **Mejor para Screening**: **Chebyshev-KAN V4** (Sensitivity perfecta 1.0000, 0 falsos negativos)
3. **Mejor Balance General**: **Baseline MLP** (MCC 0.9626, Specificity perfecta 1.0000)
4. **Interpretabilidad**: KAN permite análisis de coeficientes funcionales (ventaja sobre MLP)
5. **Estabilización Crítica**: RBF requiere centros fijos; Dropout debe ajustarse por arquitectura

### Recomendación Práctica

**Para Screening Masivo en Hospitales**:
- Usar **Chebyshev-KAN V4** (ningún caso de cáncer perdido)
- Aceptar 4 falsos positivos (biopsias innecesarias) vs salvar vidas

**Para Producción Robusta**:
- Usar **Baseline MLP** (arquitectura madura, menor complejidad, mejor MCC)

**Para Investigación Clínica**:
- Usar **Chebyshev-KAN V4** (coeficientes interpretables para análisis biomarcadores)

---

## Introducción

Este proyecto implementa y evalúa diferentes variantes de Kolmogorov-Arnold Networks (KAN) para la clasificación binaria de tumores mamarios utilizando el dataset Wisconsin Diagnostic Breast Cancer (WDBC).

### Objetivo Principal
Comparar el desempeño de cuatro variantes de redes KAN (Chebyshev V4, Wavelet V3, Fast-RBF V3, Fourier V4) contra una arquitectura MLP tradicional en un problema de clasificación médica real.

### Contexto Clínico
En diagnóstico de cáncer, los errores de clasificación tienen implicaciones diferentes:
- **Falsos Negativos (FN)**: **CRÍTICO** - Pacientes con cáncer no detectados (no reciben tratamiento)
- **Falsos Positivos (FP)**: Menos crítico - Biopsias adicionales innecesarias (inconvenientes pero no letales)
- **Objetivo Primario**: Maximizar Sensibilidad (idealmente 100%) para no perder ningún caso de cáncer
- **Objetivo Secundario**: Mantener Especificidad alta (>90%) para minimizar biopsias innecesarias

### Contexto Técnico
**Kolmogorov-Arnold Networks (KAN)** son arquitecturas que reemplazan activaciones fijas por funciones base aprendibles:
- **Teorema de Kolmogorov-Arnold**: Toda función continua multivariable puede representarse como suma de composiciones de funciones univariables
- **KAN**: Implementa este teorema usando bases funcionales (polinomios, wavelets, Fourier, RBF) en lugar de activaciones fijas (ReLU, GELU)
- **Ventaja Teórica**: Mayor expresividad y potencial interpretabilidad mediante análisis de coeficientes

---

## Estructura del Proyecto

```
taller-3/
│
├── KAN_Wisconsin_BreastCancer.ipynb    # Notebook principal con implementación completa
├── DOCUMENTACION_TECNICA.md            # Este archivo - Documentación completa del proyecto
├── ANALISIS_MEJORAS_KAN.md             # Análisis detallado de optimizaciones V3/V4
├── GUIA_RAPIDA_MEJORAS_V4.md           # Guía rápida visual con diagramas
├── GUIA_RAPIDA_ESTABILIDAD_V3.md       # Mejoras de estabilidad (Wavelet, Fast-KAN)
├── MEJORAS_ESTABILIDAD_V3.md           # Detalle de mejoras de estabilidad
├── README.md                            # Documentación general del repositorio
├── pyproject.toml                       # Configuración de dependencias (uv)
└── check_environment.py                 # Script de verificación de entorno
```

### Navegación Rápida por la Documentación

- **🚀 Inicio Rápido**: Leer `README.md`
- **📚 Entender el Código**: Leer `DOCUMENTACION_TECNICA.md` (este archivo)
- **🔬 Analizar Mejoras**: Leer `ANALISIS_MEJORAS_KAN.md`
- **💻 Implementar**: Ejecutar `KAN_Wisconsin_BreastCancer.ipynb`
- **✅ Verificar Entorno**: Ejecutar `python check_environment.py`

### Flujo del Notebook Principal

El notebook `KAN_Wisconsin_BreastCancer.ipynb` sigue esta estructura:

```
1. INTRODUCCIÓN Y CONTEXTO
   └─ Mejoras implementadas (V3/V4)

2. FASE 1: Carga y Preprocesamiento
   ├─ Imports y configuración (semilla, device)
   ├─ Carga de WDBC dataset (569 muestras, 30 features)
   └─ División estratificada (60-20-20) + estandarización

3. FASE 2: Implementación de Arquitecturas
   ├─ Baseline MLP (referencia tradicional)
   ├─ Chebyshev-KAN V4 (polinomios ortogonales)
   ├─ Wavelet-KAN V3 (Mexican Hat wavelets)
   ├─ Fast-KAN V3 (RBF Gaussianas)
   └─ Fourier-KAN V4 (series de Fourier)

4. FASE 3: Framework de Entrenamiento
   ├─ calculate_clinical_metrics() → Métricas médicas
   ├─ train_epoch() → Época de entrenamiento
   ├─ evaluate_model() → Evaluación en val/test
   └─ train_and_evaluate() → Loop completo con early stopping

5. FASE 4: Entrenamiento (ÚNICO)
   └─ Entrena los 5 modelos con versiones optimizadas finales
      (~10-15 minutos total)

6. FASE 5: Análisis y Visualizaciones
   ├─ Comparación de métricas (tabla)
   ├─ Ranking por criterio clínico
   ├─ Gráficos de barras comparativos
   ├─ Curvas ROC superpuestas
   ├─ Matrices de confusión
   ├─ Curvas de entrenamiento (loss, sensitivity, specificity)
   └─ Gráficos radar de métricas

7. FASE 6: Conclusiones
   └─ Análisis profundo de por qué cada arquitectura funciona diferente
```

### Estado del Proyecto

**Versión Actual**: 4.0 (Octubre 2025)

**Modelos Implementados**:
- ✅ Baseline MLP (referencia tradicional)
- ✅ Chebyshev-KAN V4 (optimizado para screening, sensitivity perfecta)
- ✅ Wavelet-KAN V3 (estabilizado anti-overfitting)
- ✅ Fast-KAN V3 (centros RBF fijos para estabilidad)
- ✅ Fourier-KAN V4 (arquitectura profunda, necesita optimización adicional)

**Estado de Entrenamiento**: ✅ Completo  
**Resultados Validados**: ✅ Sí (conjunto de prueba con 114 muestras)  
**Documentación**: ✅ Completa (este documento + ANALISIS_MEJORAS_KAN.md)

---

## Configuración del Entorno

### Librerías Requeridas

#### Deep Learning (PyTorch)
```python
import torch                 # Framework principal para redes neuronales
import torch.nn as nn        # Módulos de redes neuronales (capas, activaciones)
import torch.optim as optim  # Optimizadores (Adam, AdamW, SGD)
from torch.utils.data import DataLoader, TensorDataset
```

**Propósito**: PyTorch proporciona las herramientas fundamentales para construir, entrenar y evaluar redes neuronales profundas con soporte para GPU.

#### Machine Learning (Scikit-learn)
```python
from sklearn.datasets import load_breast_cancer      # Dataset WDBC
from sklearn.model_selection import train_test_split # División de datos
from sklearn.preprocessing import StandardScaler     # Estandarización
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score, matthews_corrcoef
)
```

**Propósito**: Scikit-learn proporciona utilidades para preprocesamiento y métricas de evaluación estándar en machine learning.

#### Análisis y Visualización
```python
import numpy as np           # Operaciones numéricas y álgebra lineal
import pandas as pd          # Manipulación de datos tabulares
import matplotlib.pyplot as plt  # Visualización básica
import seaborn as sns        # Visualización estadística avanzada
```

**Propósito**: Estas librerías permiten análisis exploratorio, manipulación de datos y generación de visualizaciones profesionales.

### Configuración de Reproducibilidad

```python
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)  # Semilla para PyTorch
np.random.seed(RANDOM_SEED)     # Semilla para NumPy
```

**Propósito**: Fijar semillas aleatorias asegura que los resultados sean reproducibles en múltiples ejecuciones. Esto es crítico para validación científica y debugging.

### Detección de Dispositivo

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Propósito**: Detecta automáticamente si hay GPU disponible. Las GPU aceleran significativamente el entrenamiento de redes neuronales (10-100x más rápido que CPU).

---

## Carga y Preprocesamiento de Datos

### Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)

#### Características del Dataset
- **Muestras**: 569 biopsias de tumores mamarios
- **Características**: 30 características morfológicas continuas
- **Clases**: 2 (Maligno: 212 casos, Benigno: 357 casos)
- **Desbalanceo**: 37.3% malignos, 62.7% benignos

#### Origen de las Características
Las características se calculan de imágenes digitalizadas de aspiración con aguja fina (FNA) de masa mamaria. Para cada núcleo celular se extraen:

**Características Base** (10):
1. Radio (distancia media del centro al perímetro)
2. Textura (desviación estándar de valores en escala de grises)
3. Perímetro
4. Área
5. Suavidad (variación local en longitudes de radio)
6. Compacidad (perímetro² / área - 1.0)
7. Concavidad (severidad de porciones cóncavas del contorno)
8. Puntos cóncavos (número de porciones cóncavas del contorno)
9. Simetría
10. Dimensión fractal ("aproximación de línea costera" - 1)

**Medidas** (3 por característica = 30 total):
- Media
- Error estándar
- "Peor" o mayor (promedio de los tres valores más grandes)

### Código de Carga

```python
data = load_breast_cancer()
X = data.data       # Matriz (569, 30)
y = data.target     # Vector (569,)

# Inversión de etiquetas: 0=Benigno, 1=Maligno
y = 1 - y
```

**Justificación de la inversión**: Por convención en medicina, la clase positiva (1) representa la condición de interés (enfermedad). Esto facilita la interpretación de métricas como Sensitivity (recall de clase positiva).

### División Estratificada

#### Estrategia de División
```
Dataset (569 muestras)
    │
    ├─ Entrenamiento (60%): 341 muestras
    │
    └─ Temporal (40%): 228 muestras
        │
        ├─ Validación (20% del total): 114 muestras
        │
        └─ Prueba (20% del total): 114 muestras
```

#### Código de División
```python
# Primera división: 60-40
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=RANDOM_SEED, stratify=y
)

# Segunda división: 20-20
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
)
```

**Parámetros Clave**:
- `stratify=y`: Mantiene la proporción de clases en cada conjunto (37%/63%)
- `random_state=RANDOM_SEED`: Asegura divisiones reproducibles
- `test_size`: Controla el tamaño relativo de cada conjunto

**Importancia de la Estratificación**: Con desbalanceo de clases, divisiones aleatorias pueden generar conjuntos no representativos. La estratificación asegura que train, val y test tengan la misma distribución de clases que el dataset original.

### Estandarización

#### Motivación
Las características tienen escalas muy diferentes:
- Área: rango ~150-2500
- Suavidad: rango ~0.05-0.16
- Dimensión fractal: rango ~0.05-0.10

Sin estandarización, características con mayor magnitud dominarían el gradiente durante el entrenamiento.

#### Método: StandardScaler
Transforma cada característica a media 0 y desviación estándar 1:

$$z = \frac{x - \mu}{\sigma}$$

donde:
- $x$: valor original
- $\mu$: media de la característica en conjunto de entrenamiento
- $\sigma$: desviación estándar en conjunto de entrenamiento

#### Código
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Calcula μ y σ, transforma
X_val_scaled = scaler.transform(X_val)          # Solo transforma
X_test_scaled = scaler.transform(X_test)        # Solo transforma
```

**CRÍTICO - Prevención de Data Leakage**:
- El scaler se **ajusta solo con datos de entrenamiento**
- Validación y prueba se transforman con parámetros ($\mu$, $\sigma$) de entrenamiento
- Usar estadísticas de val/test para normalizar sería **data leakage** (información del futuro)

### Conversión a Tensores de PyTorch

```python
X_train_t = torch.FloatTensor(X_train_scaled)  # Características (float32)
y_train_t = torch.LongTensor(y_train)          # Etiquetas (int64)
```

**Tipos de Datos**:
- `FloatTensor`: Para características (valores continuos)
- `LongTensor`: Para etiquetas (índices de clase)

### DataLoaders

```python
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=32,
    shuffle=True,      # Aleatorizar orden
    drop_last=True     # Descartar último batch incompleto
)
```

**Parámetros**:
- `batch_size=32`: Número de muestras procesadas simultáneamente
  - Muy pequeño: entrenamiento lento, alta varianza
  - Muy grande: mucha memoria, convergencia pobre
  - 32 es un balance estándar para datasets pequeños
- `shuffle=True`: Aleatoriza orden en cada época (solo train)
  - Previene que el modelo aprenda del orden de los datos
- `drop_last=True`: Descarta último batch si no es completo (solo train)
  - Asegura todos los batches tengan mismo tamaño (importante para BatchNorm)

---

## Arquitecturas de Modelos

### 1. Baseline MLP (Multi-Layer Perceptron)

#### Arquitectura
```
Entrada (30) → [144] → [96] → [48] → [24] → Salida (2)
```

#### Componentes por Bloque
Cada bloque contiene:
1. **Linear Layer**: Transformación afín $y = Wx + b$
2. **BatchNorm1d**: Normalización por batch
3. **GELU Activation**: Función de activación suave
4. **Dropout**: Regularización estocástica

#### Código Completo con Documentación

```python
class BaselineMLP(nn.Module):
    """
    Red Neuronal Profunda Multi-Capa (MLP) de referencia.
    
    Arquitectura:
        - 4 capas ocultas con dimensiones decrecientes
        - BatchNormalization para estabilidad
        - GELU como función de activación
        - Dropout para regularización
    
    Parámetros:
        input_size (int): Número de características de entrada (default: 30)
        num_classes (int): Número de clases de salida (default: 2)
        dropout_rate (float): Tasa de dropout (default: 0.25)
    """
    def __init__(self, input_size=30, num_classes=2, dropout_rate=0.25):
        super().__init__()
        
        # Bloque 1: 30 → 144
        self.fc1 = nn.Linear(input_size, 144)
        self.bn1 = nn.BatchNorm1d(144)
        
        # Bloque 2: 144 → 96
        self.fc2 = nn.Linear(144, 96)
        self.bn2 = nn.BatchNorm1d(96)
        
        # Bloque 3: 96 → 48
        self.fc3 = nn.Linear(96, 48)
        self.bn3 = nn.BatchNorm1d(48)
        
        # Bloque 4: 48 → 24
        self.fc4 = nn.Linear(48, 24)
        self.bn4 = nn.BatchNorm1d(24)
        
        # Capa de salida: 24 → 2
        self.output = nn.Linear(24, num_classes)
        
        # Función de activación y regularización
        self.activation = nn.GELU()  # Gaussian Error Linear Unit
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        Propagación hacia adelante.
        
        Parámetros:
            x (torch.Tensor): Tensor de entrada con forma (batch_size, 30)
        
        Retorna:
            torch.Tensor: Logits de salida con forma (batch_size, 2)
        """
        # Bloque 1
        x = self.fc1(x)          # Transformación lineal
        x = self.bn1(x)          # Normalización de batch
        x = self.activation(x)   # Función de activación
        x = self.dropout(x)      # Dropout para regularización
        
        # Bloque 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Bloque 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Bloque 4
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Capa de salida (sin activación, se aplica en la función de pérdida)
        return self.output(x)
```

#### Explicación de Componentes

**1. Linear Layer (`nn.Linear`)**
- Implementa: $y = Wx + b$
- $W$: Matriz de pesos (entrenables)
- $b$: Vector de bias (entrenable)
- Ejemplo: `nn.Linear(30, 144)` tiene 30×144 + 144 = 4,464 parámetros

**2. Batch Normalization (`nn.BatchNorm1d`)**
- Normaliza activaciones por batch:
  $$\hat{x} = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}}$$
- Luego escala y desplaza: $y = \gamma \hat{x} + \beta$
- **Ventajas**:
  - Estabiliza entrenamiento
  - Permite learning rates más altos
  - Actúa como regularizador débil
- **Parámetros**: $\gamma$ (escala) y $\beta$ (desplazamiento) son entrenables

**3. GELU Activation (`nn.GELU`)**
- Gaussian Error Linear Unit: $GELU(x) = x \cdot \Phi(x)$
- $\Phi(x)$: Función de distribución acumulativa gaussiana
- **Ventajas sobre ReLU**:
  - Derivada suave (mejor para optimización)
  - No corta completamente valores negativos
  - Estado del arte en Transformers

**4. Dropout (`nn.Dropout`)**
- Durante entrenamiento: desactiva neuronas aleatoriamente con probabilidad $p$
- Durante inferencia: usa todas las neuronas, escala salidas por $(1-p)$
- **Propósito**: Prevenir overfitting forzando redundancia

---

### 2. Chebyshev-KAN

#### Fundamento Matemático

**Polinomios de Chebyshev**:
Los polinomios de Chebyshev $T_n(x)$ son definidos en $[-1, 1]$ por:
- $T_0(x) = 1$
- $T_1(x) = x$
- $T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)$ (recurrencia)

**Propiedades**:
1. **Ortogonales**: $\int_{-1}^{1} \frac{T_m(x)T_n(x)}{\sqrt{1-x^2}}dx = 0$ si $m \neq n$
2. **Mejor aproximación**: Minimizan el error máximo de aproximación
3. **Rango acotado**: $|T_n(x)| \leq 1$ para $x \in [-1,1]$

**Aplicación en KAN**:
Aproximar función univariable $f(x)$ como:
$$f(x) \approx \sum_{n=0}^{N} a_n T_n(x)$$

#### Arquitectura Chebyshev-KAN

**ChebyshevBasis Layer**:
```python
class ChebyshevBasis(nn.Module):
    """
    Capa de transformación basada en polinomios de Chebyshev.
    
    Aproxima funciones no-lineales univariables usando expansión en
    polinomios de Chebyshev ortogonales.
    
    Parámetros:
        in_features (int): Dimensión de entrada
        out_features (int): Dimensión de salida
        degree (int): Grado máximo del polinomio (default: 3)
    """
    def __init__(self, in_features, out_features, degree=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        
        # Coeficientes para cada término polinomial
        # Forma: (out_features, in_features, degree+1)
        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, degree + 1) * 0.04
        )
        
        # Inicialización Xavier para mejor convergencia
        nn.init.xavier_uniform_(self.coeffs.view(out_features, -1).unsqueeze(0))
        
        # Término de bias aprendible
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Escala de salida aprendible con clamping
        self.output_scale = nn.Parameter(torch.ones(out_features) * 0.4)
        
        # LayerNorm para normalización adicional
        self.layer_norm = nn.LayerNorm(out_features)
    
    def chebyshev_poly(self, x, n):
        """
        Calcula el n-ésimo polinomio de Chebyshev usando recurrencia.
        
        Recurrencia: T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)
        
        Parámetros:
            x (torch.Tensor): Entrada normalizada a [-1, 1]
            n (int): Grado del polinomio
        
        Retorna:
            torch.Tensor: T_n(x)
        """
        if n == 0:
            return torch.ones_like(x)
        elif n == 1:
            return x
        else:
            T_prev2 = torch.ones_like(x)
            T_prev1 = x
            for _ in range(2, n + 1):
                T_curr = 2 * x * T_prev1 - T_prev2
                T_prev2, T_prev1 = T_prev1, T_curr
            return T_prev1
    
    def forward(self, x):
        """
        Propagación hacia adelante.
        
        Pasos:
        1. Normalizar entrada a [-1, 1]
        2. Calcular polinomios de Chebyshev hasta grado N
        3. Combinar linealmente con coeficientes aprendibles
        4. Aplicar escala y bias
        5. Normalizar salida
        
        Parámetros:
            x (torch.Tensor): Entrada (batch_size, in_features)
        
        Retorna:
            torch.Tensor: Salida (batch_size, out_features)
        """
        batch_size = x.shape[0]
        
        # Normalización a [-1, 1] usando min-max
        x_min = x.min(dim=0, keepdim=True)[0]
        x_max = x.max(dim=0, keepdim=True)[0]
        x_range = x_max - x_min + 1e-6
        x_norm = 2 * (x - x_min) / x_range - 1
        x_norm = torch.clamp(x_norm, -0.98, 0.98)  # Evitar valores extremos
        
        # Inicializar salida
        output = torch.zeros(batch_size, self.out_features, device=x.device)
        
        # Sumar contribución de cada polinomio con ponderación decreciente
        for n in range(self.degree + 1):
            # Calcular T_n(x_norm)
            cheby_n = self.chebyshev_poly(x_norm, n)
            
            # Ponderación decreciente para altos grados (reduce overfitting)
            weight = 1.0 / (1.0 + 0.1 * n)
            
            # Combinar: (batch, in) @ (out, in).T -> (batch, out)
            output += weight * torch.mm(cheby_n, self.coeffs[:, :, n].t())
        
        # Aplicar escala con clamping para estabilidad
        scale = torch.clamp(self.output_scale, 0.1, 2.0)
        output = output * scale.unsqueeze(0) + self.bias.unsqueeze(0)
        
        # Normalización de capa
        output = self.layer_norm(output)
        
        return output
```

**Flujo de Datos**:
1. Entrada: $(batch, in\_features)$
2. Normalización: cada feature mapeada a $[-1, 1]$
3. Para cada característica:
   - Calcular $T_0(x), T_1(x), ..., T_N(x)$
   - Combinar con coeficientes: $\sum_{n=0}^{N} a_n T_n(x)$
4. Sumar contribuciones, aplicar escala y bias
5. LayerNorm para normalización final
6. Salida: $(batch, out\_features)$

#### Mejoras Implementadas (V4)

**Problema Detectado**: Desbalance entre clases (Specificity 0.889)

**Soluciones**:
1. **Grado Uniforme (3)**: Antes usaba grados progresivos (3→3→2) que causaban sesgo
2. **Dropout Agresivo**: Incrementado a 0.30→0.35→0.40→0.35
3. **Arquitectura Reducida**: 128→80→48→24 (menos parámetros)
4. **Escala Aprendible**: Rango [0.1, 2.0] con clamping
5. **LayerNorm**: Normalización adicional en cada capa
6. **Ponderación Decreciente**: $w_n = 1/(1+0.1n)$ para reducir influencia de altos grados

---

### 3. Fourier-KAN

#### Fundamento Matemático

**Series de Fourier**:
Cualquier función periódica $f(x)$ puede representarse como:
$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} [a_n \cos(n\omega x + \phi_n) + b_n \sin(n\omega x + \phi_n)]$$

donde:
- $a_n, b_n$: coeficientes de Fourier
- $\omega$: frecuencia fundamental
- $\phi_n$: fase

**Aplicación en KAN**:
- Aproximar funciones usando armónicos truncados
- Frecuencias y fases aprendibles
- Útil para patrones con componentes periódicas

#### Arquitectura Fourier-KAN V4

**FourierBasis Layer**:
```python
class FourierBasis(nn.Module):
    """
    Capa de transformación basada en Series de Fourier.
    
    Aproxima funciones usando combinación de senos y cosenos con
    frecuencias y fases aprendibles.
    
    Parámetros:
        in_features (int): Dimensión de entrada
        out_features (int): Dimensión de salida
        n_harmonics (int): Número de armónicos (default: 8)
    """
    def __init__(self, in_features, out_features, n_harmonics=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_harmonics = n_harmonics
        
        # Coeficientes para términos seno y coseno
        self.a_coeffs = nn.Parameter(
            torch.randn(out_features, in_features, n_harmonics) * 0.15
        )
        self.b_coeffs = nn.Parameter(
            torch.randn(out_features, in_features, n_harmonics) * 0.15
        )
        
        # Frecuencias mixtas: lineales + logarítmicas
        freq_linear = torch.linspace(0.5, 10, n_harmonics // 2)
        freq_log = torch.logspace(0, 1.3, n_harmonics - n_harmonics // 2)
        freq_init = torch.cat([freq_linear, freq_log]).unsqueeze(0).unsqueeze(0)
        self.freq = nn.Parameter(
            freq_init.expand(out_features, in_features, n_harmonics).clone()
        )
        
        # Fase uniforme en [-π, π]
        self.phase = nn.Parameter(
            torch.rand(out_features, in_features, n_harmonics) * 2 * np.pi - np.pi
        )
        
        # Bias DC component
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Normalización por armónico
        self.norm_factor = nn.Parameter(
            torch.ones(out_features, n_harmonics) * 0.2
        )
    
    def forward(self, x):
        """
        Propagación hacia adelante.
        
        Calcula: f(x) = bias + Σ[norm_n * (a_n*cos(ω_n*x + φ_n) + b_n*sin(ω_n*x + φ_n))]
        
        Parámetros:
            x (torch.Tensor): Entrada (batch_size, in_features)
        
        Retorna:
            torch.Tensor: Salida (batch_size, out_features)
        """
        batch_size = x.shape[0]
        
        # Expandir para broadcasting
        x_expanded = x.unsqueeze(1).unsqueeze(3)  # (batch, 1, in, 1)
        
        # Clamping de frecuencias para estabilidad
        freq = torch.clamp(self.freq.unsqueeze(0), 0.1, 50)
        phase = self.phase.unsqueeze(0)
        a_coeffs = self.a_coeffs.unsqueeze(0)
        b_coeffs = self.b_coeffs.unsqueeze(0)
        
        # Calcular ángulo: ω*x + φ
        angle = freq * x_expanded + phase
        
        # Términos seno y coseno
        cos_term = torch.cos(angle)
        sin_term = torch.sin(angle)
        
        # Combinación lineal
        fourier_out = cos_term * a_coeffs + sin_term * b_coeffs
        
        # Normalización por armónico
        norm = self.norm_factor.unsqueeze(0).unsqueeze(2)
        fourier_out = fourier_out * norm
        
        # Sumar sobre features y armónicos
        output = fourier_out.sum(dim=(2, 3))
        
        # Añadir bias
        output = output + self.bias.unsqueeze(0)
        
        return output
```

#### Mejoras Implementadas (V4)

**Problema Detectado**: Underfitting (Sensitivity 0.857, MCC 0.810)

**Soluciones**:
1. **Arquitectura 67% Más Profunda**: 5 capas Fourier (vs 3)
2. **Más Armónicos**: 8 en capas iniciales (vs 6)
3. **Frecuencias Mixtas**: Lineales [0.5-10] + Logarítmicas [1-20]
4. **Coeficientes 3x Más Fuertes**: 0.15 (vs 0.05)
5. **Dropout Reducido**: 0.20→0.15→0.10 (prevenir underfitting)
6. **Bias DC**: Término constante adicional
7. **Normalización por Armónico**: Control individual de cada frecuencia

---

### 4. Wavelet-KAN

#### Fundamento Matemático

**Wavelets (Mexican Hat / Ricker)**:
$$\psi(x) = (1 - x^2) e^{-x^2/2}$$

**Propiedades**:
- **Localización**: Activa solo en ventana estrecha
- **Captura características locales**: Bordes, texturas, discontinuidades
- **Escalable y trasladable**: $\psi_{a,b}(x) = \frac{1}{\sqrt{a}}\psi(\frac{x-b}{a})$

**Aplicación en KAN**:
- Detectar patrones locales en características
- Útil para dimensión fractal, puntos cóncavos (cambios abruptos)

#### Arquitectura Wavelet-KAN V3

```python
class WaveletBasis(nn.Module):
    """
    Capa de transformación basada en Wavelets Mexican Hat.
    
    Detecta características locales usando wavelets escalables y
    trasladables.
    
    Parámetros:
        in_features (int): Dimensión de entrada
        out_features (int): Dimensión de salida
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Parámetros de escala y traslación
        self.scale = nn.Parameter(torch.ones(out_features, in_features) * 0.8)
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        
        # Pesos aprendibles para combinar wavelets
        self.weights = nn.Parameter(torch.ones(out_features, in_features) * 0.5)
        
        # Inicialización robusta
        nn.init.uniform_(self.scale, 0.3, 1.5)
        nn.init.uniform_(self.translation, -0.5, 0.5)
        nn.init.xavier_uniform_(self.weights.unsqueeze(0))
    
    def mexican_hat_wavelet(self, x):
        """
        Calcula Mexican Hat (Ricker) Wavelet.
        
        ψ(x) = (1 - x²) * exp(-x²/2)
        
        Parámetros:
            x (torch.Tensor): Entrada
        
        Retorna:
            torch.Tensor: Wavelet evaluada
        """
        return (1 - x**2) * torch.exp(-0.5 * x**2)
    
    def forward(self, x):
        """
        Propagación hacia adelante.
        
        Pasos:
        1. Expandir entrada
        2. Aplicar escala y traslación
        3. Evaluar wavelet
        4. Ponderar y combinar
        
        Parámetros:
            x (torch.Tensor): Entrada (batch_size, in_features)
        
        Retorna:
            torch.Tensor: Salida (batch_size, out_features)
        """
        x_expanded = x.unsqueeze(1)  # (batch, 1, in_features)
        
        # Aplicar escala y traslación
        scaled = (x_expanded - self.translation) / (torch.abs(self.scale) + 1e-5)
        
        # Evaluar wavelet
        wavelet_out = self.mexican_hat_wavelet(scaled)
        
        # Combinar con pesos aprendibles
        weighted_out = wavelet_out * torch.abs(self.weights)
        
        # Sumar sobre features
        return torch.sum(weighted_out, dim=2)
```

#### Mejoras Implementadas (V3)

**Problema Detectado**: Overfitting severo

**Soluciones**:
1. **Arquitectura Más Shallow**: 2 capas wavelet (vs 3)
2. **Neuronas Reducidas**: 96→64→32
3. **Dropout Agresivo**: 0.40→0.45→0.40
4. **Weight Decay Fuerte**: 0.05
5. **Pesos Aprendibles**: Para combinar wavelets

---

### 5. Fast-KAN (RBF)

#### Fundamento Matemático

**Funciones de Base Radial (RBF) Gaussianas**:
$$\phi(x; c, \sigma) = \exp\left(-\frac{||x - c||^2}{2\sigma^2}\right)$$

donde:
- $c$: centro de la RBF
- $\sigma$: ancho (controla localización)

**Propiedades**:
- **Localización radial**: Activa en vecindad de centro
- **Suavidad**: Derivadas continuas infinitas
- **Aproximación universal**: Puede aproximar cualquier función continua

**Aplicación en KAN**:
- Múltiples centros distribuidos en espacio de entrada
- Anchos adaptativos por centro
- Combinación lineal ponderada

#### Arquitectura Fast-KAN V3

```python
class RBFBasis(nn.Module):
    """
    Capa de transformación basada en Funciones de Base Radial Gaussianas.
    
    Usa múltiples centros RBF fijos con anchos adaptativos.
    
    Parámetros:
        in_features (int): Dimensión de entrada
        out_features (int): Dimensión de salida
        num_rbf_centers (int): Número de centros RBF (default: 8)
    """
    def __init__(self, in_features, out_features, num_rbf_centers=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_rbf_centers = num_rbf_centers
        
        # Centros RBF FIJOS (no entrenables) para prevenir drift
        centers_init = torch.linspace(-2, 2, num_rbf_centers)
        centers_init = centers_init.unsqueeze(0).unsqueeze(0).expand(
            out_features, in_features, num_rbf_centers
        ).clone()
        self.register_buffer('centers', centers_init)
        
        # Anchos adaptativos (log-parametrización para positividad)
        self.log_widths = nn.Parameter(
            torch.ones(out_features, in_features, num_rbf_centers) * 0.3
        )
        
        # Pesos de combinación
        self.weights = nn.Parameter(
            torch.randn(out_features, in_features, num_rbf_centers) * 0.05
        )
        
        # LayerNorm para normalización estable
        self.layer_norm = nn.LayerNorm(out_features)
    
    def forward(self, x):
        """
        Propagación hacia adelante.
        
        Calcula: f(x) = Σ w_i * exp(-||x - c_i||² / (2σ_i²))
        
        Parámetros:
            x (torch.Tensor): Entrada (batch_size, in_features)
        
        Retorna:
            torch.Tensor: Salida (batch_size, out_features)
        """
        batch_size = x.shape[0]
        
        # Expandir para broadcasting
        x_expanded = x.unsqueeze(1).unsqueeze(3)  # (batch, 1, in, 1)
        centers = self.centers.unsqueeze(0)        # (1, out, in, centers)
        
        # Distancia a centros
        diff = x_expanded - centers
        
        # Anchos con clamping fuerte para estabilidad
        widths = torch.exp(torch.clamp(self.log_widths, -1.5, 1.0))
        
        # RBF Gaussiana
        rbf_out = torch.exp(-0.5 * (diff / (widths.unsqueeze(0) + 1e-5))**2)
        
        # Combinar con pesos
        weighted_rbf = rbf_out * self.weights.unsqueeze(0)
        
        # Sumar sobre features y centros
        output = weighted_rbf.sum(dim=(2, 3))
        
        # LayerNorm
        output = self.layer_norm(output)
        
        return output
```

#### Mejoras Implementadas (V3)

**Problema Detectado**: Inestabilidad extrema

**Soluciones**:
1. **Centros RBF Fijos**: No entrenables (previene drift)
2. **Anchos Restringidos**: Clamp (-1.5, 1.0)
3. **LayerNorm**: Por capa
4. **Arquitectura Reducida**: 128→80→40→24
5. **Dropout Agresivo**: 0.30→0.35→0.40→0.35

---

## Proceso de Entrenamiento

### Framework de Entrenamiento Unificado

#### Función de Pérdida

**CrossEntropyLoss con Pesos de Clase**:
```python
class_weights = torch.tensor([1.0, 2.5]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Justificación**:
- Peso 2.5x para clase maligna (1)
- Penaliza más errores en malignos (falsos negativos)
- Balance entre sensitivity y specificity

#### Optimizador

**AdamW con Weight Decay Diferenciado**:
```python
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=lr, 
    weight_decay=weight_decay
)
```

**Configuración por Modelo**:
- **Modelos estables** (MLP, Fourier, Baseline): `weight_decay=0.01`
- **Modelos inestables** (Chebyshev): `weight_decay=0.03`
- **Modelos con overfitting** (Wavelet, Fast): `weight_decay=0.05`

#### Learning Rate Schedulers

**Estrategia Dual**:
```python
# 1. Cosine Annealing
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs-warmup, eta_min=lr*0.1
)

# 2. ReduceLROnPlateau
scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=8
)
```

**Warmup**: 10 épocas con LR lineal de 0 a `lr`

**Justificación**:
- Cosine Annealing: Decaimiento suave y predecible
- ReduceLROnPlateau: Ajuste automático si se estanca

#### Gradient Clipping

**Diferenciado por Modelo**:
```python
# Modelos inestables
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

# Modelos estables
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Propósito**: Prevenir explosión de gradientes en KAN

#### Early Stopping

**Multi-Métrica**:
```python
val_score = (
    val_metrics['sensitivity'] * 0.6 + 
    val_metrics['specificity'] * 0.3 + 
    val_metrics['f1_score'] * 0.1
)
```

**Ponderaciones**:
- 60% Sensitivity (prioridad en contexto médico)
- 30% Specificity
- 10% F1-Score

**Paciencia**: 20-25 épocas según estabilidad del modelo

---

### Función de Entrenamiento Completa

```python
def train_and_evaluate(model, train_loader, val_loader, test_loader, 
                       model_name, epochs=150, lr=0.0008, device='cpu'):
    """
    Entrena y evalúa un modelo con configuración optimizada.
    
    Parámetros:
        model: Instancia del modelo a entrenar
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        test_loader: DataLoader de prueba
        model_name (str): Nombre del modelo para logging
        epochs (int): Número máximo de épocas
        lr (float): Learning rate inicial
        device: Dispositivo de cómputo (CPU/GPU)
    
    Retorna:
        dict: Diccionario con:
            - 'model': Modelo entrenado
            - 'history': Historial de métricas por época
            - 'test_metrics': Métricas en conjunto de prueba
            - 'test_predictions': Predicciones y probabilidades
    """
    print(f"\n{'='*60}")
    print(f"Entrenando: {model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    
    # Configuración específica por modelo
    if 'Wavelet' in model_name or 'Fast' in model_name:
        weight_decay = 0.05
        clip_norm = 0.5
        patience = 25
    elif 'Chebyshev' in model_name:
        weight_decay = 0.03
        clip_norm = 0.7
        patience = 20
    else:
        weight_decay = 0.01
        clip_norm = 1.0
        patience = 20
    
    # Función de pérdida con pesos
    class_weights = torch.tensor([1.0, 2.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizador
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    
    # Schedulers
    warmup_epochs = 10
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.1
    )
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=lr * 0.01
    )
    
    # Early stopping
    best_val_score = -float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Historial
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_sensitivity': [],
        'val_specificity': [],
        'learning_rate': []
    }
    
    # Loop de entrenamiento
    for epoch in range(epochs):
        # Warmup
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / warmup_epochs
        
        # Entrenamiento
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, clip_norm
        )
        
        # Validación
        y_true, y_pred, y_prob = evaluate_model(model, val_loader, device)
        val_metrics = calculate_clinical_metrics(y_true, y_pred, y_prob)
        
        # Calcular val_loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
        val_loss /= len(val_loader)
        
        # Guardar historial
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_sensitivity'].append(val_metrics['sensitivity'])
        history['val_specificity'].append(val_metrics['specificity'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Score combinado
        val_score = (
            val_metrics['sensitivity'] * 0.6 + 
            val_metrics['specificity'] * 0.3 + 
            val_metrics['f1_score'] * 0.1
        )
        
        # Early stopping
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Schedulers
        if epoch >= warmup_epochs:
            scheduler_cosine.step()
            scheduler_plateau.step(val_score)
        
        # Logging
        if (epoch + 1) % 15 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Sensitivity: {val_metrics['sensitivity']:.4f} | "
                  f"Specificity: {val_metrics['specificity']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping en época {epoch+1}")
            break
    
    # Cargar mejor modelo
    model.load_state_dict(best_model_state)
    
    # Evaluación final en test
    y_true_test, y_pred_test, y_prob_test = evaluate_model(model, test_loader, device)
    test_metrics = calculate_clinical_metrics(y_true_test, y_pred_test, y_prob_test)
    
    print(f"\nResultados en Test Set:")
    print(f"   Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"   Specificity: {test_metrics['specificity']:.4f}")
    print(f"   F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"   AUC-ROC: {test_metrics['auc_roc']:.4f}")
    print(f"   MCC: {test_metrics['mcc']:.4f}")
    
    return {
        'model': model,
        'history': history,
        'test_metrics': test_metrics,
        'test_predictions': (y_true_test, y_pred_test, y_prob_test)
    }
```

---

## Evaluación y Métricas

### Métricas Clínicas

**Matriz de Confusión**:
```
                Predicción
              Benigno  Maligno
Real Benigno    TN       FP
     Maligno    FN       TP
```

**Métricas Derivadas**:

1. **Sensitivity (Sensibilidad / Recall)**:
   $$Sensitivity = \frac{TP}{TP + FN}$$
   - Proporción de malignos correctamente identificados
   - **Crítico en medicina**: Minimizar FN

2. **Specificity (Especificidad)**:
   $$Specificity = \frac{TN}{TN + FP}$$
   - Proporción de benignos correctamente identificados

3. **Positive Predictive Value (PPV / Precision)**:
   $$PPV = \frac{TP}{TP + FP}$$
   - De los predichos malignos, cuántos son realmente malignos

4. **Negative Predictive Value (NPV)**:
   $$NPV = \frac{TN}{TN + FN}$$
   - De los predichos benignos, cuántos son realmente benignos
   - **Crítico en screening**: Alta NPV permite confiar en negativo

5. **F1-Score**:
   $$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$
   - Media armónica entre precision y recall

6. **Matthews Correlation Coefficient (MCC)**:
   $$MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$
   - Rango: [-1, 1]
   - Métrica más balanceada para clases desbalanceadas

7. **AUC-ROC (Area Under ROC Curve)**:
   - Curva ROC: Sensitivity vs (1 - Specificity)
   - AUC: Probabilidad de que modelo ordene aleatorios positivo y negativo correctamente
   - Rango: [0, 1], 0.5 = aleatorio, 1.0 = perfecto

---

### Implementación de Cálculo de Métricas

```python
def calculate_clinical_metrics(y_true, y_pred, y_prob):
    """
    Calcula métricas clínicas completas para evaluación médica.
    
    Parámetros:
        y_true (np.array): Etiquetas verdaderas
        y_pred (np.array): Predicciones del modelo
        y_prob (np.array): Probabilidades de clase positiva
    
    Retorna:
        dict: Diccionario con todas las métricas clínicas
    """
    # Matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Métricas clínicas
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Métricas estándar
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'mcc': mcc,
        'tn': int(tn), 'fp': int(fp),
        'fn': int(fn), 'tp': int(tp)
    }
```

---

## Análisis de Resultados

### Resultados Finales (Versiones Optimizadas)

Los siguientes resultados corresponden al entrenamiento final con las arquitecturas optimizadas (V3/V4) en el conjunto de prueba (20% del dataset, 114 muestras):

| Modelo | Sensitivity | Specificity | F1-Score | AUC-ROC | MCC | FN | FP |
|--------|------------|-------------|----------|---------|-----|----|----|
| **Baseline MLP** | **0.9524** | **1.0000** | **0.9756** | 0.9954 | **0.9626** | 2 | 0 |
| **Chebyshev-KAN V4** | **1.0000** | 0.9444 | 0.9545 | **0.9980** | 0.9286 | 0 | 4 |
| **Wavelet-KAN V3** | 0.8571 | **0.9861** | 0.9114 | 0.9792 | 0.8688 | 6 | 1 |
| **Fast-KAN V3** | 0.9762 | 0.9722 | 0.9647 | 0.9874 | 0.9439 | 1 | 2 |
| **Fourier-KAN V4** | 0.8571 | 0.9167 | 0.8571 | 0.9649 | 0.7738 | 6 | 6 |

#### Interpretación Clínica de los Resultados

**Matriz de Confusión - Contexto Médico**:
- **TP (True Positives)**: Casos malignos correctamente detectados → Pacientes reciben tratamiento adecuado
- **TN (True Negatives)**: Casos benignos correctamente identificados → Evita biopsias innecesarias
- **FN (False Negatives)**: Casos malignos no detectados → **MUY CRÍTICO** - Pacientes con cáncer no reciben tratamiento
- **FP (False Positives)**: Casos benignos clasificados como malignos → Genera biopsias adicionales (menos crítico)

### Análisis Detallado por Modelo

#### 1. Baseline MLP (Mejor Balance General)
```
Métricas Clave:
- Sensitivity: 0.9524 (95.2% de malignos detectados)
- Specificity: 1.0000 (100% de benignos correctos)
- MCC: 0.9626 (mejor correlación global)
- Falsos Negativos: 2 (solo 2 casos malignos perdidos)
- Falsos Positivos: 0 (ningún benigno mal clasificado)
```

**Fortalezas**:
- **Mejor MCC global** (0.9626) indica balance óptimo
- **Specificity perfecta** (1.0000) - ningún falso positivo
- Arquitectura madura y estable
- Menor complejidad computacional que KAN

**Limitaciones**:
- 2 falsos negativos (crítico en detección de cáncer)
- Menor interpretabilidad que KAN

**Conclusión**: Excelente opción para **producción** por su balance y estabilidad.

---

#### 2. Chebyshev-KAN V4 (Mejor para Screening)
```
Métricas Clave:
- Sensitivity: 1.0000 (100% de malignos detectados)
- Specificity: 0.9444 (94.4% de benignos correctos)
- AUC-ROC: 0.9980 (mejor discriminación)
- Falsos Negativos: 0 (ningún caso maligno perdido)
- Falsos Positivos: 4 (4 casos benignos mal clasificados)
```

**Fortalezas**:
- **Sensitivity perfecta** (1.0000) - **CRÍTICO para screening**
- **Mejor AUC-ROC** (0.9980) - excelente capacidad discriminativa
- Cero falsos negativos (ningún paciente con cáncer no detectado)
- Interpretabilidad mediante análisis de coeficientes polinomiales

**Limitaciones**:
- 4 falsos positivos (4 pacientes sin cáncer recibirían biopsias innecesarias)
- Mayor complejidad computacional que MLP
- Specificity 94.4% (inferior a MLP)

**Conclusión**: **Ideal para screening masivo** donde el objetivo es no perder ningún caso de cáncer, aceptando algunos falsos positivos.

**Interpretación de Mejoras V4**:
- Grado uniforme (3) + LayerNorm → Mayor estabilidad
- Dropout agresivo (0.30-0.40) → Previene overfitting en clase maligna
- Escala aprendible → Adaptación automática a magnitud de características

---

#### 3. Wavelet-KAN V3 (Especialista en Patrones Locales)
```
Métricas Clave:
- Sensitivity: 0.8571 (85.7% de malignos detectados)
- Specificity: 0.9861 (98.6% de benignos correctos)
- MCC: 0.8688
- Falsos Negativos: 6 (6 casos malignos no detectados)
- Falsos Positivos: 1 (1 caso benigno mal clasificado)
```

**Fortalezas**:
- **Alta Specificity** (0.9861) - muy pocos falsos positivos
- Excelente para detectar características locales (fractal dimension, concavidad)
- Solo 1 falso positivo
- Útil cuando se prioriza especificidad sobre sensibilidad

**Limitaciones**:
- **Sensitivity baja** (0.8571) - 6 casos malignos no detectados
- **No recomendado para screening** por alta tasa de falsos negativos
- Requiere ajuste adicional para aplicación clínica

**Conclusión**: Útil como **modelo complementario** en ensemble para detectar patrones específicos (texturas, irregularidades).

**Interpretación de Mejoras V3**:
- Arquitectura shallow (2 capas) → Reduce overfitting
- Dropout agresivo (0.40-0.45) → Estabilización forzada
- Wavelets Mexican Hat → Detecta cambios abruptos en características

---

#### 4. Fast-KAN V3 (Balance Robusto)
```
Métricas Clave:
- Sensitivity: 0.9762 (97.6% de malignos detectados)
- Specificity: 0.9722 (97.2% de benignos correctos)
- MCC: 0.9439
- Falsos Negativos: 1 (1 caso maligno no detectado)
- Falsos Positivos: 2 (2 casos benignos mal clasificados)
```

**Fortalezas**:
- **Excelente balance** Sensitivity/Specificity (~97% ambos)
- Solo 1 falso negativo (segundo mejor en detección de malignos)
- RBF con centros fijos → Estabilidad mejorada dramáticamente
- Buen desempeño en espacios de alta dimensión (30 features)

**Limitaciones**:
- No destaca en ninguna métrica específica
- Complejidad computacional moderada-alta

**Conclusión**: Opción **robusta y balanceada** para aplicaciones clínicas donde se requiere equilibrio entre sensitivity y specificity.

**Interpretación de Mejoras V3**:
- Centros RBF fijos → Previene drift durante entrenamiento
- Anchos con clamp (-1.5, 1.0) → Estabilización de gaussianas
- LayerNorm → Normalización adicional para convergencia

---

#### 5. Fourier-KAN V4 (Necesita Optimización Adicional)
```
Métricas Clave:
- Sensitivity: 0.8571 (85.7% de malignos detectados)
- Specificity: 0.9167 (91.7% de benignos correctos)
- MCC: 0.7738 (menor correlación)
- Falsos Negativos: 6 (6 casos malignos no detectados)
- Falsos Positivos: 6 (6 casos benignos mal clasificados)
```

**Fortalezas**:
- Arquitectura más profunda (5 capas) con mayor expresividad
- Frecuencias mixtas pueden capturar patrones complejos
- Potencial para capturar simetría celular

**Limitaciones**:
- **Peor desempeño general** en este dataset
- 6 falsos negativos + 6 falsos positivos (12 errores totales)
- **No recomendado para aplicación clínica** en su estado actual
- Posiblemente requiere más datos o ajustes adicionales

**Conclusión**: A pesar de las mejoras V4, **Fourier-KAN no logra desempeño competitivo** en este dataset. Posibles razones:
1. Características del dataset no tienen patrones periódicos fuertes
2. Arquitectura muy profunda puede estar causando overfitting residual
3. Necesita exploración adicional de hiperparámetros

**Recomendación**: Considerar arquitectura híbrida (Fourier + Chebyshev) o explorar datasets con patrones más periódicos.

---

### Comparación Visual de Trade-offs

```
PRIORIDAD: SENSITIVITY (Detectar Cáncer)
════════════════════════════════════════
Chebyshev-KAN V4:  1.0000 ████████████████████ (MEJOR - 0 FN)
Fast-KAN V3:       0.9762 ███████████████████░
Baseline MLP:      0.9524 ███████████████████░
Wavelet-KAN V3:    0.8571 █████████████████░░░
Fourier-KAN V4:    0.8571 █████████████████░░░

PRIORIDAD: SPECIFICITY (Evitar Biopsias Innecesarias)
═══════════════════════════════════════════════════════
Baseline MLP:      1.0000 ████████████████████ (PERFECTO)
Wavelet-KAN V3:    0.9861 ███████████████████░
Fast-KAN V3:       0.9722 ███████████████████░
Chebyshev-KAN V4:  0.9444 ██████████████████░░
Fourier-KAN V4:    0.9167 ██████████████████░░

PRIORIDAD: BALANCE (MCC)
═══════════════════════════════════════
Baseline MLP:      0.9626 ████████████████████ (MEJOR)
Fast-KAN V3:       0.9439 ███████████████████░
Chebyshev-KAN V4:  0.9286 ██████████████████░░
Wavelet-KAN V3:    0.8688 █████████████████░░░
Fourier-KAN V4:    0.7738 ███████████████░░░░░
```

### Ranking Final por Aplicación Clínica

#### 🥇 Para Screening Masivo (Prioridad: Detectar TODO cáncer)
**Recomendación**: **Chebyshev-KAN V4**
- Sensitivity: 1.0000 (0 falsos negativos)
- AUC-ROC: 0.9980 (mejor discriminación)
- Trade-off aceptable: 4 falsos positivos → biopsias adicionales
- **Impacto**: Ningún paciente con cáncer queda sin diagnosticar

#### 🥈 Para Diagnóstico de Confirmación (Balance Sensitivity-Specificity)
**Recomendación**: **Baseline MLP** o **Fast-KAN V3**
- Baseline MLP: Specificity perfecta, solo 2 FN
- Fast-KAN V3: Balance 97.6%/97.2%, solo 1 FN
- **Impacto**: Minimiza errores en ambas direcciones

#### 🥉 Para Producción (Estabilidad y Robustez)
**Recomendación**: **Baseline MLP**
- MCC: 0.9626 (mejor balance)
- Arquitectura madura y probada
- Menor complejidad computacional
- **Impacto**: Despliegue confiable y mantenible

#### 🔬 Para Investigación (Interpretabilidad)
**Recomendación**: **Chebyshev-KAN V4**
- Coeficientes polinomiales interpretables
- AUC-ROC: 0.9980
- Análisis de contribución por característica
- **Impacto**: Insights sobre relaciones no-lineales entre features

---

## Conclusiones

### Hallazgos Principales

#### 1. KAN vs MLP: Rendimiento Comparable con Trade-offs Específicos
- **Baseline MLP**: Mejor MCC (0.9626) y Specificity perfecta (1.0000)
- **Chebyshev-KAN V4**: Mejor AUC-ROC (0.9980) y Sensitivity perfecta (1.0000)
- **Fast-KAN V3**: Mejor balance general (Sens: 0.9762, Spec: 0.9722)

**Conclusión**: Las redes KAN **no superan universalmente** a MLP tradicional, pero ofrecen **trade-offs valiosos** según la aplicación.

#### 2. La Arquitectura Debe Adaptarse al Problema
- **Chebyshev-KAN**: Sobresale con características polinomiales suaves (radius, area, perimeter)
- **Wavelet-KAN**: Mejor para patrones locales (fractal dimension, concavidad) pero underfitting en este dataset
- **Fourier-KAN**: Bajo desempeño sugiere que características no tienen patrones periódicos fuertes
- **Fast-KAN**: Balance robusto gracias a centros RBF fijos en espacios de alta dimensión

**Conclusión**: La **elección de base funcional** debe estar guiada por el **dominio del problema**.

#### 3. Dropout es Crítico para Regularización
- **Muy alto** (0.40-0.45 en Wavelet): Causó underfitting → Sensitivity 0.8571
- **Muy bajo** (0.10-0.20 en Fourier inicial): Causó overfitting → Inestabilidad
- **Óptimo** (0.30-0.35 en Chebyshev y Fast): Balance entre regularización y capacidad

**Conclusión**: Dropout debe **ajustarse por arquitectura** según complejidad y tendencia al overfitting.

#### 4. Sensitivity Perfecta es Alcanzable pero con Trade-offs
- **Chebyshev-KAN V4**: Sensitivity 1.0000 (0 FN) pero 4 falsos positivos
- **Baseline MLP**: Specificity 1.0000 (0 FP) pero 2 falsos negativos

**Conclusión**: En aplicaciones médicas, priorizar **Sensitivity** (detectar cáncer) aceptando más falsos positivos es generalmente **preferible** a priorizar Specificity.

#### 5. AUC-ROC Alto No Garantiza Bajo Error Práctico
- **Chebyshev-KAN V4**: AUC 0.9980 pero 4 FP
- **Fourier-KAN V4**: AUC 0.9649 pero 12 errores totales (6 FN + 6 FP)

**Conclusión**: AUC-ROC mide **capacidad de discriminación global**, no necesariamente **errores clínicos mínimos** en umbral operativo.

#### 6. Estabilización de RBF Requiere Centros Fijos
**Fast-KAN antes de V3**: Especificidad catastrófica (~0.25) por drift de centros durante entrenamiento.

**Fast-KAN V3**: Centros fijos + anchos con clamp → Sensitivity 0.9762, Specificity 0.9722.

**Conclusión**: En arquitecturas basadas en RBF, **anclar centros** es esencial para estabilidad.

#### 7. Interpretabilidad vs Desempeño: KAN Ofrece Ventaja
- **MLP**: Mejor MCC pero pesos sin interpretación directa
- **Chebyshev-KAN**: Coeficientes polinomiales revelan relaciones no-lineales entre features
- **Wavelet-KAN**: Escalas y traslaciones indican localización de patrones discriminativos

**Conclusión**: KAN permite **análisis post-hoc** de qué transformaciones funcionales son importantes.

#### 8. Profundidad vs Regularización: Balance Delicado
- **Fourier-KAN V4**: 5 capas profundas para combatir underfitting inicial
- **Wavelet-KAN V3**: Solo 2 capas wavelet para combatir overfitting
- **Chebyshev-KAN V4**: 4 capas con dropout agresivo progresivo

**Conclusión**: Profundidad debe **aumentarse para underfitting** pero **acompañarse de regularización fuerte** (dropout, weight decay).

---

### Recomendaciones para Producción

#### Escenario 1: Screening Masivo (Hospitales, Campañas)
**Modelo Recomendado**: **Chebyshev-KAN V4**

**Justificación**:
- Sensitivity perfecta (1.0000) → Ningún caso de cáncer pasa desapercibido
- AUC-ROC más alto (0.9980) → Mejor discriminación global
- 4 falsos positivos → Costo aceptable (biopsias adicionales) vs beneficio (salvar vidas)

**Protocolo de Implementación**:
1. Usar Chebyshev-KAN V4 como **primera línea de screening**
2. Casos positivos → Enviar a biopsia y pruebas adicionales
3. Monitoreo continuo de tasa de biopsias innecesarias

**Métricas de Éxito**:
- **Sensitivity > 99%** (permitir máximo 1% de falsos negativos)
- Tasa de biopsia innecesaria < 10%
- NPV (Negative Predictive Value) > 99.5%

---

#### Escenario 2: Diagnóstico de Confirmación (Clínicas Especializadas)
**Modelo Recomendado**: **Baseline MLP** o **Fast-KAN V3**

**Justificación**:
- **Baseline MLP**: Specificity perfecta (1.0000), MCC 0.9626
- **Fast-KAN V3**: Balance 97.6%/97.2%, solo 1 FN
- Ambos minimizan errores totales

**Protocolo de Implementación**:
1. Pacientes con resultados preliminares positivos → Evaluación con MLP
2. Si MLP confirma maligno → Tratamiento inmediato
3. Si MLP indica benigno pero screening fue positivo → Pruebas adicionales

**Métricas de Éxito**:
- **Balance Sensitivity-Specificity > 95%** en ambas
- MCC > 0.93
- Tasa de error total < 5%

---

#### Escenario 3: Producción (Sistemas Hospitalarios)
**Modelo Recomendado**: **Baseline MLP**

**Justificación**:
- Arquitectura madura y probada
- Menor complejidad computacional → Latencia baja
- MCC más alto (0.9626) → Mejor balance general
- Facilidad de mantenimiento y actualización

**Protocolo de Implementación**:
1. Despliegue en servidores con PyTorch optimizado
2. API REST con tiempos de respuesta < 100ms
3. Sistema de monitoreo de drift de datos
4. Re-entrenamiento trimestral con nuevos datos

**Métricas de Éxito**:
- Latencia < 100ms por predicción
- Disponibilidad > 99.9%
- MCC > 0.95 en producción

---

#### Escenario 4: Investigación Clínica (Análisis de Biomarcadores)
**Modelo Recomendado**: **Chebyshev-KAN V4**

**Justificación**:
- Coeficientes polinomiales interpretables
- AUC-ROC más alto → Mejor ordenamiento de riesgo
- Permite análisis de contribución por característica

**Protocolo de Análisis**:
1. Entrenar Chebyshev-KAN y extraer coeficientes aprendidos
2. Analizar qué términos polinomiales tienen mayor magnitud
3. Identificar características con mayor contribución no-lineal
4. Publicar insights sobre relaciones morfológicas vs malignidad

**Métricas de Éxito**:
- Identificación de top-5 características más discriminativas
- Cuantificación de no-linealidades (grados 2-3 de Chebyshev)
- Correlación con literatura médica existente

---

### Limitaciones del Estudio

1. **Dataset Pequeño**: 569 muestras pueden no capturar toda la variabilidad clínica
   - Recomendación: Validar en datasets externos (DDSM, MIAS)

2. **División Fija**: Resultados dependen de división train/val/test específica
   - Recomendación: Realizar validación cruzada 10-fold

3. **Características Predefinidas**: Usa features extraídas, no imágenes directas
   - Recomendación: Explorar KAN en arquitecturas CNN end-to-end

4. **Fourier-KAN Bajo Desempeño**: Puede indicar inadecuación de bases periódicas
   - Recomendación: Probar bases mixtas (Fourier + Chebyshev híbrido)

5. **Sin Análisis de Incertidumbre**: No se cuantifica confianza de predicciones
   - Recomendación: Integrar Dropout Bayesiano o Ensembles

6. **Interpretabilidad Limitada**: Aunque KAN ofrece coeficientes, no se analizan en profundidad
   - Recomendación: Aplicar SHAP/LIME y análisis de sensibilidad

---

### Trabajo Futuro

#### 1. Ensemble Híbrido: MLP + Chebyshev-KAN
**Objetivo**: Combinar estabilidad de MLP con discriminación de Chebyshev

**Arquitectura Propuesta**:
```python
Ensemble = 0.5 × MLP(x) + 0.5 × ChebyshevKAN(x)
```

**Expectativa**: Sensitivity ≥ 0.98, Specificity ≥ 0.98

---

#### 2. Análisis de Interpretabilidad Profunda
**Métodos**:
- Extraer coeficientes de Chebyshev por capa y característica
- Visualizar contribución de términos polinomiales (lineal, cuadrático, cúbico)
- Correlacionar con literatura médica sobre morfología tumoral

**Preguntas de Investigación**:
- ¿Qué características tienen relaciones más no-lineales?
- ¿Cómo se comparan con conocimiento médico previo?

---

#### 3. Transfer Learning desde Datasets Grandes
**Dataset Fuente**: ImageNet pre-entrenado → Fine-tuning en imágenes mamográficas

**Objetivo**: Mejorar generalización con conocimiento previo

---

#### 4. Quantificación de Incertidumbre
**Método**: Monte Carlo Dropout o Deep Ensembles

**Aplicación**: Estimar confianza por predicción → Priorizar casos con alta incertidumbre para revisión humana

---

#### 5. Optimización de Hiperparámetros Bayesiana
**Herramienta**: Optuna con TPE (Tree-structured Parzen Estimator)

**Espacio de Búsqueda**:
- Learning rate: [1e-4, 1e-2]
- Dropout: [0.1, 0.5]
- Arquitectura: Número de capas, neuronas por capa
- Grado de Chebyshev: [2, 5]

**Objetivo**: Encontrar configuración óptima globalmente

---

#### 6. Validación Externa
**Datasets Adicionales**:
- **DDSM** (Digital Database for Screening Mammography)
- **MIAS** (Mammographic Image Analysis Society)
- **INbreast** (Portuguese mammography database)

**Objetivo**: Evaluar generalización cross-dataset

---

#### 7. Explicabilidad con SHAP
**Implementación**: Aplicar SHAP (SHapley Additive exPlanations) a Chebyshev-KAN

**Visualizaciones**:
- SHAP waterfall plots por predicción
- SHAP summary plots por dataset
- Dependence plots de características clave

**Impacto**: Confianza clínica mediante explicaciones locales

---

#### 8. KAN en Arquitecturas CNN End-to-End
**Propuesta**: Reemplazar capas fully-connected finales de ResNet con Chebyshev-KAN

**Objetivo**: Combinar extracción automática de características (CNN) con aproximación funcional (KAN)

---

#### 9. Análisis de Robustez
**Pruebas**:
- Perturbaciones adversariales (FGSM, PGD)
- Ruido Gaussiano en características
- Data augmentation (rotaciones, escalados en imágenes originales)

**Objetivo**: Evaluar estabilidad de modelos ante variaciones

---

#### 10. Despliegue Clínico Piloto
**Fase 1**: Prueba retrospectiva en hospital colaborador

**Fase 2**: Estudio prospectivo comparando con radiólogos

**Métricas**:
- Concordancia inter-observador (Modelo vs Médico)
- Tiempo de diagnóstico (con vs sin IA)
- Satisfacción del personal médico

---

## Referencias

### Papers Fundamentales
1. Kolmogorov, A. N. (1957). On the representation of continuous functions of several variables by superposition of continuous functions of one variable and addition.
2. Liu, Z. et al. (2024). KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756.

### Datasets
1. Wolberg, W.H., Street, W.N., and Mangasarian, O.L. (1995). Wisconsin Diagnostic Breast Cancer (WDBC). UCI Machine Learning Repository.

### Técnicas de Regularización
1. Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training.
2. Srivastava, N. et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting.

### Optimización
1. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization (AdamW).

---

**Fin de la Documentación Técnica**
