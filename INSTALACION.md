# Guía de Instalación - Hugging Face Transformers

## Opción 1: Instalación básica (recomendada)

```bash
pip install -r requirements.txt
```

## Opción 2: Instalación mínima (solo lo esencial)

```bash
pip install transformers torch huggingface-hub
```

## Opción 3: Instalación con GPU (CUDA)

Para sistemas con GPU NVIDIA y CUDA instalado:

```bash
# Para CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Luego instalar el resto
pip install transformers huggingface-hub tokenizers sentencepiece
```

```bash
# Para CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Luego instalar el resto
pip install transformers huggingface-hub tokenizers sentencepiece
```

## Opción 4: Instalación con soporte para audio

```bash
pip install transformers[torch,audio]
pip install librosa soundfile
```

## Opción 5: Instalación con soporte para visión

```bash
pip install transformers[torch,vision]
pip install Pillow timm
```

## Opción 6: Instalación completa (todo incluido)

```bash
pip install transformers[torch,sentencepiece,tokenizers,audio,vision]
pip install datasets accelerate jupyter notebook
```

## Verificación de instalación

Después de instalar, verifica que todo funcione correctamente:

```python
import transformers
import torch

print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

# Si tienes GPU, verifica la versión de CUDA
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Número de GPUs: {torch.cuda.device_count()}")
```

## Entorno virtual (recomendado)

Se recomienda usar un entorno virtual para evitar conflictos:

### Con venv:
```bash
# Crear entorno virtual
python -m venv venv_huggingface

# Activar (Windows)
venv_huggingface\Scripts\activate

# Activar (Linux/Mac)
source venv_huggingface/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Con conda:
```bash
# Crear entorno virtual
conda create -n huggingface python=3.10

# Activar
conda activate huggingface

# Instalar PyTorch con conda (recomendado para GPU)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Instalar el resto con pip
pip install transformers huggingface-hub tokenizers sentencepiece
```

## Google Colab

Si usas Google Colab, las librerías principales ya están instaladas. Solo necesitas:

```python
# En una celda de Colab
!pip install --upgrade transformers huggingface-hub
```

## Solución de problemas comunes

### Error: "No module named 'transformers'"
```bash
pip install --upgrade transformers
```

### Error con tokenizers en Windows
```bash
pip install --upgrade tokenizers --no-cache-dir
```

### Error: "CUDA out of memory"
- Reducir batch_size en tus modelos
- Usar modelos más pequeños (distil*)
- Usar device="cpu" en lugar de GPU

### Error con sentencepiece
```bash
pip install sentencepiece --no-binary sentencepiece
```

## Requisitos del sistema

- Python 3.8 o superior
- 4 GB RAM mínimo (8 GB recomendado)
- 10 GB espacio en disco para modelos
- GPU opcional pero recomendada para modelos grandes

## Caché de modelos

Los modelos se descargan en:
- Linux/Mac: `~/.cache/huggingface/`
- Windows: `C:\Users\{usuario}\.cache\huggingface\`

Para cambiar la ubicación:
```bash
export HF_HOME=/tu/ruta/personalizada
```