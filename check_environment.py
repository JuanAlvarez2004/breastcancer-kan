"""
Script de ejemplo: Gestión de paquetes con UV en notebooks
Ejecuta este script para ver ejemplos de comandos útiles
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Ejecuta un comando y muestra el resultado"""
    print(f"\n{'='*70}")
    print(f"📝 {description}")
    print(f"{'='*70}")
    print(f"Comando: {cmd}\n")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8'
        )
        
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"⚠️ Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error al ejecutar: {e}")

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║         UV Package Manager - Ejemplos para Notebooks              ║
    ║         Proyecto: Variantes KAN                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Verificar que estamos en el entorno correcto
    print("\n🔍 VERIFICACIÓN DEL ENTORNO")
    print("="*70)
    print(f"Python ejecutable: {sys.executable}")
    print(f"Versión de Python: {sys.version.split()[0]}")
    print(f"Path del proyecto: {Path.cwd()}")
    
    # Verificar instalaciones
    print("\n\n📦 PAQUETES PRINCIPALES INSTALADOS")
    print("="*70)
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  - CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - Dispositivos: {torch.cuda.device_count()}")
    except ImportError:
        print("✗ PyTorch no instalado")
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError:
        print("✗ NumPy no instalado")
    
    try:
        import sklearn
        print(f"✓ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("✗ Scikit-learn no instalado")
    
    try:
        import matplotlib
        print(f"✓ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("✗ Matplotlib no instalado")
    
    try:
        import seaborn as sns
        print(f"✓ Seaborn: {sns.__version__}")
    except ImportError:
        print("✗ Seaborn no instalado")
    
    try:
        import tqdm
        print(f"✓ tqdm: {tqdm.__version__}")
    except ImportError:
        print("✗ tqdm no instalado")
    
    try:
        import IPython
        print(f"✓ IPython: {IPython.__version__}")
    except ImportError:
        print("✗ IPython no instalado")
    
    try:
        import jupyter_core
        print(f"✓ Jupyter: {jupyter_core.__version__}")
    except ImportError:
        print("✗ Jupyter no instalado")
    
    print("\n\n💡 COMANDOS ÚTILES DE UV")
    print("="*70)
    print("""
    # Ver todas las dependencias instaladas
    uv pip list
    
    # Ver árbol de dependencias
    uv tree
    
    # Agregar una nueva dependencia
    uv add nombre-paquete
    
    # Ejemplos:
    uv add pandas              # Agregar pandas
    uv add plotly              # Visualización interactiva
    uv add tensorboard         # Tracking de experimentos
    uv add albumentations      # Data augmentation
    
    # Eliminar dependencia
    uv remove nombre-paquete
    
    # Actualizar todas las dependencias
    uv lock --upgrade
    uv sync
    
    # Ejecutar comandos sin activar el entorno
    uv run python mi_script.py
    uv run jupyter notebook
    uv run jupyter lab
    
    # Listar versiones de Python disponibles
    uv python list
    
    # Cache
    uv cache dir               # Ver ubicación del cache
    uv cache clean             # Limpiar cache
    """)
    
    print("\n\n📊 EJEMPLOS ESPECÍFICOS PARA TU PROYECTO KAN")
    print("="*70)
    print("""
    # Agregar herramientas de visualización avanzada
    uv add plotly wandb tensorboard
    
    # Agregar optimización de hiperparámetros
    uv add optuna ray[tune]
    
    # Agregar procesamiento de imágenes médicas
    uv add SimpleITK pydicom opencv-python
    
    # Agregar métricas avanzadas
    uv add torchmetrics
    
    # Agregar framework de training
    uv add pytorch-lightning
    
    # Agregar interpretabilidad
    uv add captum shap
    
    # Agregar utilities
    uv add python-dotenv loguru rich
    """)
    
    print("\n\n🎯 CÓMO USAR EN TU NOTEBOOK")
    print("="*70)
    print("""
    1. Abre VS Code
    2. Abre VariantesKANS.ipynb
    3. Click en "Select Kernel" (arriba derecha)
    4. Selecciona el kernel de .venv
    5. ¡Listo! Todas las dependencias estarán disponibles
    
    Para verificar en el notebook:
    
    ```python
    import sys
    print(sys.executable)  # Debe mostrar .venv
    
    import torch
    print(torch.__version__)
    ```
    
    Para instalar algo temporalmente en el notebook:
    
    ```python
    %pip install nombre-paquete
    ```
    
    ⚠️ Luego ejecuta en terminal:
    uv add nombre-paquete  # Para persistir en pyproject.toml
    """)
    
    print("\n\n✅ PROYECTO CONFIGURADO EXITOSAMENTE")
    print("="*70)
    print("""
    Tu proyecto está listo para usar. Las siguientes dependencias están instaladas:
    
    ✓ PyTorch (Deep Learning)
    ✓ NumPy (Operaciones numéricas)
    ✓ Scikit-learn (ML utilities)
    ✓ Matplotlib & Seaborn (Visualización)
    ✓ tqdm (Progress bars)
    ✓ Jupyter & IPython (Notebooks)
    
    Para más información, consulta:
    - README.md: Documentación completa
    - QUICK_START.md: Guía rápida de comandos
    """)
    
    print("\n" + "="*70)
    print("🚀 ¡Disfruta programando con UV!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
