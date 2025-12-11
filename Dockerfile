# =================================================================
# 1. IMAGEN BASE (LA PRIMERA LÍNEA ACTIVA)
# =================================================================
FROM nvcr.io/nvidia/pytorch:23.10-py3

# =================================================================
# 2. CONFIGURACIÓN DEL ENTORNO
# =================================================================
# Aumentar timeout para descargas (Aunque ahora usamos cache local, es buena práctica)
ENV HF_TIMEOUT=600
ENV DEBIAN_FRONTEND=noninteractive

# Directorio de trabajo
WORKDIR /app

# =================================================================
# 3. INSTALAR DEPENDENCIAS CRÍTICAS PARA DETR
# =================================================================
# 'transformers', 'timm', y 'scipy' (necesario para el Matching Húngaro de DETR)
RUN pip install --no-cache-dir \
    datasets==2.16.1 \
    fsspec==2023.10.0 \
    transformers==4.35.2 \
    timm==0.9.10 \
    scipy \
    torchmetrics \
    torchvision \
    tqdm

# =================================================================
# 4. COPIAR SCRIPT
# =================================================================
COPY Bench_Coco_Det_Tra.py /app/

# =================================================================
# 5. COMANDO DE EJECUCIÓN
# =================================================================
CMD ["python", "Bench_Coco_Det_Tra.py"]