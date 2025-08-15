# Documentación del Archivo pruebas.ipynb

## Descripción General

El archivo `pruebas.ipynb` es un Jupyter Notebook que implementa un pipeline completo de machine learning para la clasificación de la enfermedad de Parkinson utilizando un enfoque multimodal. El notebook combina datos clínicos, imágenes de resonancia magnética (MRI), y imágenes SPECT (tanto reales como sintéticas).

## Estructura del Notebook

### 1. Instalación de Dependencias
```python
%pip install -r /home/Data/franklin_pupils/yeferson/ccyclegan/requirements.txt
```
Instala todas las dependencias necesarias para el proyecto.

### 2. Sección: Dataset SPECT Sintético

#### Propósito
Procesa y organiza las imágenes SPECT sintéticas generadas para crear un dataset estructurado.

#### Funcionalidad
- **Exploración de Directorios**: Recorre los directorios de imágenes sintéticas
- **Filtrado de Archivos**: Selecciona solo archivos PNG de conjuntos 'train' y 'test'
- **Extracción de Metadatos**: 
  - `tipo_spect`: Tipo de imagen SPECT
  - `clase_spect`: Clasificación binaria (0=Control, 1=Parkinson)
  - `paciente_spect`: Identificador del paciente
- **Exportación**: Guarda el dataset procesado en CSV

#### Código Clave
```python
# Búsqueda recursiva de imágenes PNG
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.png'):
            if root.split('/')[-3] in coincidencias:
                imgs.append(os.path.join(root, file))

# Extracción de características de la ruta
spect_sinteticos['clase_spect'] = spect_sinteticos['path_spect'].apply(
    lambda x: 1 if 'parkinson' in x else 0
)
```

### 3. Sección: CycleGAN

#### Propósito
Implementa un Conditional CycleGAN para la generación de imágenes SPECT sintéticas de alta calidad.

#### Arquitectura del Modelo

##### CCycleGAN Class
Clase principal que encapsula toda la arquitectura del modelo:

**Parámetros de Configuración:**
- `img_rows`, `img_cols`: Dimensiones de imagen (256x256)
- `channels`: Número de canales (1 para escala de grises)
- `num_classes`: Número de clases (2: Control/Parkinson)
- Pesos de pérdida: `d_gan_loss_w`, `d_cl_loss_w`, `g_gan_loss_w`, etc.
- Parámetros de optimización Adam

##### Discriminador
```python
def build_discriminator(self):
    """Construcción del Discriminador con dos salidas:
       - Validez: indica si la imagen es real/falsa.
       - Clasificación: predice la etiqueta (0 o 1) de la imagen."""
```

**Características:**
- **Arquitectura**: Capas convolucionales con LeakyReLU
- **Normalización**: LayerNormalization para estabilidad
- **Doble Salida**: 
  1. Validación GAN (real/fake)
  2. Clasificación (Control/Parkinson)
- **Embedding de Etiquetas**: Para condicionamiento

##### Generador (Encoder-Decoder)
```python
def build_generator_enc_dec(self):
    """Construcción del Generador tipo Encoder-Decoder"""
```

**Características:**
- **Encoder**: Reduce dimensionalidad progresivamente
- **Decoder**: Reconstruye imágenes condicionadas por etiquetas
- **Skip Connections**: Para preservar detalles finos
- **Condicionamiento**: Incorpora etiquetas de clase en la generación

#### Funciones de Pérdida

**Pérdida Combinada:**
1. **GAN Loss**: `binary_crossentropy` para discriminación real/fake
2. **Classification Loss**: `sparse_categorical_crossentropy` para clasificación
3. **Reconstruction Loss**: `mae` para consistencia cíclica

#### DataLoader Personalizado
```python
class DataLoader:
    def __init__(self, csv_path, img_res):
        # Carga y procesa el dataset
        # Maneja tanto imágenes reales como sintéticas
```

**Funcionalidades:**
- Carga imágenes desde rutas CSV
- Redimensionamiento automático
- Normalización de píxeles
- Balanceado de clases
- Generación de batches

### 4. Sección: Entrenamiento del CycleGAN

#### Proceso de Entrenamiento
```python
cyclegan.train(epochs=200, batch_size=4, save_interval=50)
```

**Características del Entrenamiento:**
- **Épocas**: 200 (configurable)
- **Batch Size**: 4 (ajustable según GPU)
- **Save Interval**: Guarda checkpoints cada 50 épocas
- **Monitoreo**: Pérdidas del discriminador y generador
- **Validación**: Generación de muestras periódicas

#### Métricas de Seguimiento
- `d_loss_real`: Pérdida del discriminador en imágenes reales
- `d_loss_fake`: Pérdida del discriminador en imágenes falsas
- `g_loss`: Pérdida del generador
- `class_acc`: Precisión de clasificación

### 5. Sección: Vision Transformer Multimodal

#### Arquitectura del Modelo

##### MultiModalVisionTransformer
```python
class MultiModalVisionTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, 
                 num_classes=2, embed_dim=768, depth=12, num_heads=12,
                 clinical_dim=14, use_embeddings=True):
```

**Componentes Principales:**

1. **Patch Embedding**: Convierte imágenes en secuencias de patches
2. **Position Embedding**: Codificación posicional para Transformer
3. **Clinical Embedding**: Procesamiento de datos clínicos
4. **Multi-Head Attention**: Mecanismo de atención entre modalidades
5. **Classification Head**: Capa final para predicción

##### ClinicalImageDataset
```python
class ClinicalImageDataset(Dataset):
    def __init__(self, img_csv_path, clinical_control_csv, clinical_pd_excel,
                 img_res=(256, 256), mode='train', img_col='mri',
                 clinical_features=None, spect_synth_path=None):
```

**Funcionalidades:**
- **Multimodal Loading**: Carga MRI, SPECT y datos clínicos
- **Augmentation**: Transformaciones de imágenes
- **Sincronización**: Alineación de datos por paciente
- **Preprocesamiento**: Normalización y estandarización

#### Características Clínicas Utilizadas
```python
clinical_features = ['DBSYN', 'NP3SPCH', 'NP3FACXP', 'NP3RIGN', 
                     'NP3RIGRU', 'NP3RIGLU', 'NP3RIGRL', 'NP3RIGLL', 
                     'NP3HMOVR', 'NP3HMOVL', 'NP3GAIT', 'NP3FRZGT', 
                     'NP3PSTBL', 'NP3POSTR']
```

### 6. Sección: Entrenamiento y Evaluación

#### Configuración del Entrenamiento
```python
# Configuración del modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalVisionTransformer(
    img_size=256, patch_size=16, in_chans=1, num_classes=2, 
    embed_dim=768, depth=12, num_heads=12,
    clinical_dim=len(train_dataset.clinical_features), 
    use_embeddings=True
).to(device)

# Función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

#### Loop de Entrenamiento
```python
for epoch in range(epochs):
    model.train()
    # Entrenamiento por batches con tqdm para progreso
    for mri_images, spect_images, clinical_data, labels in train_loader:
        # Forward pass
        outputs = model(mri_images, spect_images, clinical_data)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

#### Evaluación del Modelo
```python
# Evaluación en conjunto de test
model.eval()
with torch.no_grad():
    for mri_images, spect_images, clinical_data, labels in test_loader:
        outputs = model(mri_images, spect_images, clinical_data)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
```

#### Métricas de Evaluación
- **Classification Report**: Precision, Recall, F1-Score por clase
- **AUC-ROC**: Área bajo la curva ROC
- **Confusion Matrix**: Matriz de confusión implícita
- **Training Curves**: Visualización de pérdida y accuracy

#### Visualización de Resultados
```python
# Gráficos de entrenamiento
plt.figure(figsize=(12, 5))

# Curva de pérdida
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")

# Curva de accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label="Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curve")
```

## Flujo de Trabajo Completo

1. **Preparación de Datos**: 
   - Procesamiento de imágenes SPECT sintéticas
   - Organización de datasets multimodales

2. **Generación de Datos Sintéticos**: 
   - Entrenamiento del Conditional CycleGAN
   - Generación de imágenes SPECT adicionales

3. **Entrenamiento Multimodal**: 
   - Carga de datos MRI, SPECT y clínicos
   - Entrenamiento del Vision Transformer

4. **Evaluación y Análisis**: 
   - Métricas de clasificación
   - Visualización de resultados
   - Guardado del modelo entrenado

## Requisitos Computacionales

- **GPU**: Recomendada para entrenamiento (CUDA compatible)
- **Memoria**: Mínimo 16GB RAM, 8GB VRAM
- **Almacenamiento**: Suficiente para datasets de imágenes
- **Python**: 3.8+ con PyTorch, TensorFlow, y dependencias científicas

## Archivos Generados

- **Checkpoints del CycleGAN**: Modelos guardados periódicamente
- **Imágenes Sintéticas**: Muestras generadas para evaluación
- **Modelo Final**: `{nombre}.pth` con pesos del Vision Transformer
- **Métricas de Entrenamiento**: Logs y visualizaciones

## Consideraciones Técnicas

- **Memory Management**: Optimización para GPUs con memoria limitada
- **Data Augmentation**: Transformaciones para aumentar robustez
- **Regularización**: Dropout y técnicas para evitar overfitting
- **Balanceado de Clases**: Estratificación para datasets desbalanceados

Este notebook representa un pipeline de investigación completo que combina generación de datos sintéticos con clasificación multimodal, proporcionando una herramienta poderosa para el diagnóstico asistido por computadora de la enfermedad de Parkinson.