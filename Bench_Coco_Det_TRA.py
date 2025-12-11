import os
import time
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from transformers import DetrImageProcessor, DetrForObjectDetection
from torchmetrics.detection import MeanAveragePrecision

# ==============================================================================
# 0. CLASE AUXILIAR: EARLY STOPPING
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, path='best_detr_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'   ⚠️ EarlyStopping: {self.counter}/{self.patience} sin mejora.')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        print(f'   ✅ Mejora detectada. Guardando checkpoint en {self.path}...')
        torch.save(model.state_dict(), self.path)

# ==============================================================================
# 1. PREPARACIÓN DE DATOS (PROCESSOR & COLLATE)
# ==============================================================================
# Usamos el procesador oficial de Facebook para normalizar imágenes y cajas
PROCESSOR = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def detr_collate_fn(batch):
    """
    Procesa un batch de ejemplos raw de 'datasets' para el formato de DETR.
    DETR necesita: pixel_values, pixel_mask y labels (class_labels + boxes).
    """
    images = []
    targets = []
    
    for item in batch:
        images.append(item['image'].convert("RGB"))
        
        # Convertir anotaciones de COCO al formato de DETR
        # COCO:  [x, y, w, h] (absoluto)
        # DETR espera que el procesador lo convierta, pero necesitamos pasarle el formato correcto.
        annotations = item['objects']
        new_ann = {'image_id': item['image_id'], 'annotations': []}
        
        # Filtrar cajas vacías o inválidas
        valid_boxes = []
        valid_labels = []
        
        for bbox, category in zip(annotations['bbox'], annotations['category']):
            x, y, w, h = bbox
            if w > 0 and h > 0:
                # El procesador de HuggingFace maneja la conversión a (cx, cy, w, h) norm
                # si le pasamos las cajas en formato 'coco_detection'
                valid_boxes.append([x, y, w, h])
                valid_labels.append(category)
        
        targets.append({'boxes': valid_boxes, 'class_labels': valid_labels})

    # El procesador hace el padding y la normalización automáticamente
    # return_tensors="pt" devuelve tensores de PyTorch listos para GPU
    encoding = PROCESSOR(images=images, annotations=targets, return_tensors="pt")
    
    return encoding

# ==============================================================================
# 2. REPORTE
# ==============================================================================
def print_final_report(training_time, best_val_loss, avg_inference_time, estimated_fps, max_vram, metrics):
    map_50 = metrics.get('map_50', torch.tensor(0.0)).item()
    map_total = metrics.get('map', torch.tensor(0.0)).item()
    
    print("\n" + "="*60)
    print("✅ REPORTE FINAL - DETR (Transformers)")
    print("="*60)
    print(f"1. TIEMPO ENTRENAMIENTO: {training_time:.2f} s")
    print(f"2. INFERENCIA PROMEDIO:  {avg_inference_time*1000:.4f} ms/imagen")
    print(f"3. FPS ESTIMADOS:        {estimated_fps:.2f} FPS")
    print(f"4. VRAM MÁXIMA:          {max_vram:.2f} GB")
    print("-" * 30)
    print(f"   🏆 mAP Total: {map_total:.4f}")
    print(f"   🏆 mAP@50:    {map_50:.4f}")
    print(f"   📉 Best Loss: {best_val_loss:.4f}")
    print("="*60)

# ==============================================================================
# BLOQUE PRINCIPAL
# ==============================================================================
if __name__ == '__main__':
    # Configuración de Spawn para CUDA
    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # --- CONFIGURACIÓN ---
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64        # Ambicioso, para A100 80GB
    EPOCHS = 10
    LOCAL_CACHE = "/data"  # Mapeo del volumen
    
    print(f"--- Iniciando Benchmark Transformer (DETR) en {DEVICE} ---")
    print(f"--- Batch Size: {BATCH_SIZE} | Dataset Completo ---")

    # --- CARGA DE DATOS ---
    # Cargamos el dataset completo desde la carpeta local
    train_dataset = load_dataset("detection-datasets/coco", split="train", cache_dir=LOCAL_CACHE)
    val_dataset = load_dataset("detection-datasets/coco", split="val", cache_dir=LOCAL_CACHE)

    # DataLoaders con el collate específico de DETR
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=detr_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=detr_collate_fn, num_workers=4)

    # --- MODELO ---
    # Usamos DETR con ResNet-50 preentrenado. Ignoramos mismatch de tamaños si cambiamos num_classes, 
    # pero aquí usaremos las clases por defecto de COCO (91) para aprovechar los pesos.
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=3, path="best_detr_model.pth")

    # --- ENTRENAMIENTO ---
    print("Iniciando Entrenamiento...")
    start_train = time.time()

    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1} Train"):
            # Mover batch a GPU
            pixel_values = batch["pixel_values"].to(DEVICE)
            pixel_mask = batch["pixel_mask"].to(DEVICE)
            labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

            # Forward pass (DETR calcula la loss internamente si recibe labels)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- VALIDACIÓN (Loss) ---
        model.eval()
        val_loss_sum = 0
        steps = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch+1} Val"):
                pixel_values = batch["pixel_values"].to(DEVICE)
                pixel_mask = batch["pixel_mask"].to(DEVICE)
                labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                val_loss_sum += outputs.loss.item()
                steps += 1
        
        avg_loss = val_loss_sum / steps if steps > 0 else 0
        print(f"Validation Loss: {avg_loss:.4f}")
        
        early_stopping(avg_loss, model)
        if early_stopping.early_stop:
            print("Stop!")
            break

    training_time = time.time() - start_train

    # --- INFERENCIA & mAP ---
    print("Calculando mAP...")
    model.load_state_dict(torch.load("best_detr_model.pth"))
    model.eval()
    
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox").to(DEVICE)
    if DEVICE.type == 'cuda': torch.cuda.reset_peak_memory_stats(DEVICE)
    
    times = []
    
    # Limitamos a 100 batches para no eternizar el benchmark de inferencia
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Inferencia", total=100)):
            if i >= 100: break
            
            pixel_values = batch["pixel_values"].to(DEVICE)
            pixel_mask = batch["pixel_mask"].to(DEVICE)
            orig_labels = batch["labels"] # Labels originales para metric target

            t0 = time.time()
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            t1 = time.time()
            times.append((t1 - t0) / len(pixel_values))

            # Post-proceso: Convertir (cx,cy,w,h) relativo -> (x,y,x,y) absoluto
            # Necesitamos el tamaño original de las imágenes. 
            # Como ejemplo simplificado, usaremos el tamaño del tensor (no es exacto pixel a pixel vs original, 
            # pero sirve para benchmark técnico).
            target_sizes = torch.tensor([img.shape[1:] for img in pixel_values]).to(DEVICE)
            results = PROCESSOR.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)

            # Formato para torchmetrics
            formatted_preds = []
            for res in results:
                formatted_preds.append({
                    "boxes": res["boxes"],
                    "scores": res["scores"],
                    "labels": res["labels"]
                })
            
            # Formato targets (hay que revertir la normalización de DETR o usar los datos raw, 
            # aquí simplificamos asumiendo que el target se ajusta a la predicción para benchmark)
            # NOTA: Para mAP exacto de COCO, se requiere usar los tamaños originales de imagen.
            # Para benchmark de velocidad/memoria, esto es suficiente.
            formatted_targets = []
            for t, size in zip(orig_labels, target_sizes):
                 # Convertir target relativo de vuelta a absoluto para la métrica
                 boxes = t['boxes'].to(DEVICE)
                 # cxcywh -> xyxy
                 h, w = size
                 scale_fct = torch.tensor([w, h, w, h]).to(DEVICE)
                 # Conversión manual simple
                 boxes = boxes * scale_fct
                 # Convertir cxcywh a xyxy
                 b_box = torch.zeros_like(boxes)
                 b_box[:, 0] = boxes[:, 0] - boxes[:, 2]/2
                 b_box[:, 1] = boxes[:, 1] - boxes[:, 3]/2
                 b_box[:, 2] = boxes[:, 0] + boxes[:, 2]/2
                 b_box[:, 3] = boxes[:, 1] + boxes[:, 3]/2
                 
                 formatted_targets.append({"boxes": b_box, "labels": t["class_labels"].to(DEVICE)})

            metric.update(formatted_preds, formatted_targets)

    metrics = metric.compute()
    max_vram = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)
    avg_inf = sum(times)/len(times)
    
    print_final_report(training_time, early_stopping.best_loss, avg_inf, 1.0/avg_inf, max_vram, metrics)