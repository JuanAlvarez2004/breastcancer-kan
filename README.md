# Variantes KAN (Kolmogorov-Arnold Networks)

Implementación de diferentes variantes de KAN para clasificación de imágenes médicas.

---

## Introducción

Este proyecto implementa y evalúa diferentes variantes de Kolmogorov-Arnold Networks (KAN) para la clasificación binaria de tumores mamarios utilizando el dataset Wisconsin Diagnostic Breast Cancer (WDBC).

### Objetivo Principal
Comparar el desempeño de cuatro variantes de redes KAN (Chebyshev, Wavelet, Fast-RBF, Fourier) contra una arquitectura MLP tradicional en un problema de clasificación médica real.

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

## Variantes implementadas

1. **Wave-KAN**: Utiliza wavelets (Morlet) como funciones base
2. **Chebyshev-KAN**: Usa polinomios de Chebyshev para mejor aproximación
3. **Fast-KAN (RBF)**: Implementación rápida con funciones de base radial
4. **Fourier-KAN**: Captura patrones frecuentes con series de Fourier

---

## Configuración del proyecto con UV

Este proyecto usa `uv` como administrador de paquetes para gestionar dependencias de manera rápida y eficiente.

### Instalación inicial

```bash
# Inicializar el entorno virtual y sincronizar dependencias
uv sync
```

### Activar el entorno virtual

```bash
# En Windows PowerShell
.venv\Scripts\Activate.ps1

# En Windows CMD
.venv\Scripts\activate.bat

# En Linux/Mac
source .venv/bin/activate
```

### Gestión de dependencias

#### Instalar nuevas dependencias

```bash
# Agregar una dependencia de producción
uv add nombre-paquete

# Ejemplo: agregar pandas
uv add pandas

# Agregar versión específica
uv add pandas==2.0.0

# Agregar dependencia de desarrollo
uv add --dev pytest
```

#### Eliminar dependencias

```bash
# Remover una dependencia
uv remove nombre-paquete

# Ejemplo: remover pandas
uv remove pandas
```

#### Actualizar dependencias

```bash
# Actualizar todas las dependencias
uv lock --upgrade

# Actualizar una dependencia específica
uv lock --upgrade-package nombre-paquete
```

#### Ver dependencias instaladas

```bash
# Listar todas las dependencias
uv pip list

# Ver árbol de dependencias
uv tree
```

## Trabajar con Jupyter Notebooks

### Opción 1: Usar el kernel del entorno virtual

```bash
# El kernel de IPython ya está instalado como dependencia
# VS Code detectará automáticamente el entorno .venv
```

### Opción 2: Ejecutar desde la terminal

```bash
# Activar el entorno
.venv\Scripts\Activate.ps1

# Abrir Jupyter
uv run jupyter notebook

# O usar Jupyter Lab
uv run jupyter lab
```

### Instalar paquetes dentro del notebook

Si necesitas instalar algo rápidamente desde el notebook:

```python
# En una celda del notebook
import sys
import subprocess

# Instalar con uv
subprocess.check_call([sys.executable, "-m", "pip", "install", "nombre-paquete"])

# O mejor aún, usa el comando mágico de Jupyter
%pip install nombre-paquete
```

**⚠️ RECOMENDACIÓN**: Mejor agregar dependencias con `uv add` desde la terminal para mantener el archivo `pyproject.toml` actualizado.

## Comandos útiles de UV

```bash
# Ejecutar un comando en el entorno sin activarlo
uv run python script.py
uv run jupyter notebook

# Sincronizar el entorno con pyproject.toml
uv sync

# Crear lock file sin instalar
uv lock

# Limpiar cache de uv
uv cache clean

# Ver información del entorno
uv python list
uv python pin 3.11  # Fijar versión de Python
```

## Dependencias principales

- **torch**: Framework de deep learning
- **numpy**: Operaciones numéricas
- **scikit-learn**: Utilidades de ML (datasets, métricas, preprocesamiento)
- **matplotlib**: Visualización
- **seaborn**: Gráficos estadísticos
- **tqdm**: Barras de progreso
- **ipykernel**: Kernel de Jupyter para notebooks

## Troubleshooting

### El kernel no encuentra los paquetes

1. Asegúrate de que VS Code esté usando el kernel correcto (`.venv`)
2. Reinicia el kernel del notebook
3. Si persiste, ejecuta en una celda:
   ```python
   import sys
   print(sys.executable)  # Debe apuntar a .venv
   ```

### Error al instalar PyTorch con GPU

```bash
# Para CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Para CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Para CPU only
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Actualizar UV

```bash
# Actualizar uv a la última versión
pip install --upgrade uv

# O con pipx
pipx upgrade uv
```
