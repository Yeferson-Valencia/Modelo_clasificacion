# Modelo de Clasificación para Enfermedad de Parkinson

Este repositorio contiene implementaciones de modelos de clasificación para la detección de la enfermedad de Parkinson utilizando datos multimodales incluyendo imágenes médicas (MRI, SPECT) y datos clínicos.

## Archivos Principales

- `pruebas.ipynb`: Notebook principal con implementaciones de modelos avanzados
- `pruebas.py`: Script de clasificación utilizando datos clínicos con validación cruzada

## Documentación del Notebook `pruebas.ipynb`

### Descripción General

El notebook `pruebas.ipynb` contiene un flujo de trabajo completo para la clasificación de la enfermedad de Parkinson utilizando múltiples enfoques y modalidades de datos. Incluye generación de datos sintéticos, transformaciones de imágenes y clasificación multimodal.

### Secciones del Notebook

#### 1. Instalación de Dependencias
```python
%pip install -r /home/Data/franklin_pupils/yeferson/ccyclegan/requirements.txt
```
Instala las dependencias necesarias para ejecutar el notebook.

#### 2. Dataset SPECT Sintético

Esta sección se encarga de procesar y organizar imágenes SPECT sintéticas para el entrenamiento.

**Funcionalidad:**
- Escanea directorios para encontrar imágenes PNG
- Filtra imágenes de entrenamiento y prueba
- Crea un DataFrame con rutas de imágenes y etiquetas
- Clasifica automáticamente entre casos control (0) y Parkinson (1)
- Extrae información del paciente desde las rutas de archivos

**Salida:**
- Archivo CSV: `spect_sinteticos.csv` con columnas:
  - `path_spect`: Ruta completa a la imagen
  - `tipo_spect`: Tipo de imagen SPECT
  - `clase_spect`: Etiqueta binaria (0=Control, 1=Parkinson)
  - `paciente_spect`: ID del paciente

#### 3. CycleGAN Condicional

Implementación completa de una red CycleGAN condicional para transformación de imágenes médicas.

**Componentes Principales:**

##### Arquitectura del Modelo:
- **Generador**: Arquitectura encoder-decoder con embedding de etiquetas
- **Discriminador**: Red con dos salidas (validez + clasificación)
- **Modelo Combinado**: Integra generador y discriminador para entrenamiento end-to-end

##### Características Técnicas:
- **Dimensiones de imagen**: 256x256 píxeles, 1 canal (escala de grises)
- **Clases**: 2 (Control y Parkinson)
- **Optimizador**: Adam (lr=0.0002, beta1=0.5, beta2=0.999)
- **Funciones de pérdida**:
  - Binary cross-entropy para validez
  - Sparse categorical cross-entropy para clasificación
  - Mean Absolute Error (MAE) para reconstrucción

##### Funciones de Pérdida Configurables:
- `d_gan_loss_w`: Peso para pérdida adversarial del discriminador
- `d_cl_loss_w`: Peso para pérdida de clasificación del discriminador
- `g_gan_loss_w`: Peso para pérdida adversarial del generador
- `g_cl_loss_w`: Peso para pérdida de clasificación del generador
- `rec_loss_w`: Peso para pérdida de reconstrucción

##### DataLoader Personalizado:
- Carga y preprocesa imágenes automáticamente
- Normalización a rango [-1, 1]
- Manejo de lotes balanceados
- Soporte para imágenes de diferentes tamaños

#### 4. Clasificación Multimodal

Múltiples implementaciones de clasificadores que combinan diferentes tipos de datos.

##### 4.1 Clasificación con SPECT Sintético
- **Modelo**: Vision Transformer (ViT) modificado
- **Datos**: Solo imágenes SPECT sintéticas
- **Arquitectura**: 
  - Tamaño de imagen: 256x256
  - Tamaño de patch: 16x16
  - Dimensión de embedding: 768
  - Profundidad: 12 capas
  - Número de heads: 12

##### 4.2 Clasificación con Datos Clínicos
- **Características clínicas utilizadas**: 
  - `DBSYN`, `NP3SPCH`, `NP3FACXP`, `NP3RIGN`
  - `NP3RIGRU`, `NP3RIGLU`, `NP3RIGRL`, `NP3RIGLL`
  - `NP3HMOVR`, `NP3HMOVL`, `NP3GAIT`, `NP3FRZGT`
  - `NP3PSTBL`, `NP3POSTR`
- **Fuentes de datos**:
  - Control: `embcExtension_control.csv`
  - Parkinson: `embcExtension.xlsx` (hoja 'PD')

##### 4.3 Clasificación Multimodal (MRI + SPECT + Clínico)
- **Arquitectura**: MultiModalVisionTransformer
- **Modalidades**:
  - Imágenes MRI (256x256)
  - Imágenes SPECT sintéticas (256x256)
  - Datos clínicos (14 características)
- **Características**:
  - Fusión de características multimodales
  - Embeddings especializados para cada modalidad
  - Atención cruzada entre modalidades

#### 5. Información Motora

Procesamiento y análisis de datos motores relacionados con la enfermedad de Parkinson.

**Datos procesados:**
- Escalas MDS-UPDRS Parte III
- Métricas de rigidez y movimientos
- Evaluación de marcha y estabilidad postural

### Métricas de Evaluación

Todos los modelos son evaluados utilizando:
- **Accuracy**: Precisión general del modelo
- **AUC-ROC**: Área bajo la curva ROC
- **Precision**: Precisión por clase (Control/Parkinson)
- **Recall**: Sensibilidad por clase
- **F1-Score**: Media armónica de precision y recall
- **Classification Report**: Reporte detallado por clase

### Visualizaciones Incluidas

- Curvas de pérdida durante el entrenamiento
- Curvas de precisión por época
- Métricas de rendimiento comparativas
- Análisis de gradientes (Grad-CAM)

## Requisitos del Sistema

### Dependencias Principales:
- Python 3.8+
- PyTorch 1.8+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- matplotlib
- tqdm
- openpyxl

### Hardware Recomendado:
- GPU compatible con CUDA (recomendado para entrenamiento)
- Mínimo 16GB RAM
- Espacio de almacenamiento: >50GB para datasets completos

## Estructura de Datos

### Datasets Requeridos:
1. **Imágenes MRI**: Formato PNG/DICOM, 256x256 píxeles
2. **Imágenes SPECT**: Sintéticas y reales, escala de grises
3. **Datos Clínicos**: Archivos CSV/Excel con escalas MDS-UPDRS
4. **Estructura de directorios**:
   ```
   /home/Data/Datasets/Parkinson/
   ├── radiological/PPMI/tmpSpect/
   │   ├── train/
   │   └── test/
   ├── embcExtension_control.csv
   ├── embcExtension.xlsx
   └── MDS-UPDRS_Part_III_*.csv
   ```

## Uso del Notebook

1. **Preparación del entorno**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuración de rutas**:
   - Ajustar las rutas de datos en las celdas correspondientes
   - Verificar la disponibilidad de GPU

3. **Ejecución secuencial**:
   - Ejecutar las celdas en orden
   - Cada sección puede ejecutarse independientemente
   - Los modelos entrenados se guardan automáticamente

4. **Personalización**:
   - Modificar hiperparámetros en las celdas de configuración
   - Ajustar arquitecturas de red según necesidades
   - Cambiar métricas de evaluación

## Notas Técnicas

- El notebook utiliza rutas absolutas específicas del sistema
- Se requiere acceso a datasets propietarios para ejecución completa
- Los modelos están optimizados para clasificación binaria
- Soporte para GPUs NVIDIA con CUDA

## Contribución

Para contribuir al proyecto:
1. Fork del repositorio
2. Crear rama de características
3. Implementar cambios
4. Ejecutar pruebas
5. Enviar pull request

## Licencia

Este proyecto está destinado para uso académico e investigación en el área de diagnóstico médico asistido por IA.
