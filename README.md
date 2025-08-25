# Modelo de Clasificación para la Enfermedad de Parkinson

Este proyecto implementa modelos de aprendizaje automático para la clasificación y detección de la enfermedad de Parkinson utilizando datos clínicos, imágenes de resonancia magnética (MRI) y imágenes SPECT (tanto reales como sintéticas generadas mediante GANs).

## Descripción del Proyecto

El proyecto utiliza un enfoque multimodal que combina:

1. **Datos Clínicos**: Características clínicas del MDS-UPDRS Part III
2. **Imágenes MRI**: Imágenes de resonancia magnética cerebral
3. **Imágenes SPECT**: Imágenes de transportador de dopamina (DaTscan)
4. **Imágenes SPECT Sintéticas**: Generadas usando Conditional CycleGAN

## Archivos Principales

### `pruebas.ipynb`
Notebook principal que contiene la implementación completa del pipeline de machine learning, incluyendo:
- Procesamiento de imágenes SPECT sintéticas
- Implementación de Conditional CycleGAN para generación de imágenes
- Modelo Vision Transformer multimodal
- Entrenamiento y evaluación de modelos

### `pruebas.py`
Script de Python que implementa un modelo de red neuronal más simple para clasificación binaria usando únicamente datos clínicos:
- Validación cruzada estratificada con 20 folds
- Red neuronal de 3 capas con PyTorch
- Métricas de evaluación completas (accuracy, AUC, precision, recall, F1-score)

## Estructura de Datos

### Datos Clínicos
El proyecto utiliza características clínicas del MDS-UPDRS Part III con diferentes configuraciones según el modelo:

**Modelo Simple (pruebas.py)**: 6 variables de rigidez y movimientos de manos
**Modelo Multimodal (pruebas.ipynb)**: 14 variables completas del MDS-UPDRS Part III

*Ver sección [Preprocesamiento de Variables Clínicas](#preprocesamiento-de-variables-clínicas) para detalles completos.*

### Archivos de Datos
- `embcExtension_control.csv`: Datos de sujetos control
- `embcExtension.xlsx`: Datos de pacientes con Parkinson (hoja 'PD')
- `spect_sinteticos.csv`: Rutas de imágenes SPECT sintéticas
- Varios archivos CSV con datos clínicos adicionales

## Modelos Implementados

### 1. Conditional CycleGAN
- Generación de imágenes SPECT sintéticas
- Arquitectura encoder-decoder con discriminador
- Pérdidas combinadas: GAN + clasificación + reconstrucción
- Implementado en TensorFlow/Keras

### 2. Vision Transformer Multimodal
- Procesa simultáneamente imágenes MRI, SPECT y datos clínicos
- Arquitectura Transformer adaptada para datos multimodales
- Embeddings separados para cada modalidad
- Implementado en PyTorch

### 3. Red Neuronal Simple (pruebas.py)
- **Clase**: `Net` (hereda de `nn.Module`)
- **Arquitectura**: 6 → 64 → 32 → 1 neuronas
- **Funciones de activación**: ReLU + Sigmoid
- **Regularización**: Dropout (0.3) en cada capa oculta
- **Optimizador**: Adam (lr=0.001)
- **Función de pérdida**: Binary Cross Entropy (BCELoss)
- **Validación**: 20-fold estratificada con StandardScaler por fold

## Instalación y Uso

### Requisitos
```bash
pip install pandas numpy scikit-learn torch torchvision tensorflow matplotlib tqdm openpyxl
```

## Resultados y Métricas

El modelo evalúa el rendimiento usando:
- **Accuracy**: Precisión general del modelo
- **AUC-ROC**: Área bajo la curva ROC
- **Precision/Recall/F1-Score**: Para ambas clases (Control/Parkinson)
- **Validación Cruzada**: 20-fold stratified para robustez estadística

## Preprocesamiento de Variables Clínicas

### Variables del MDS-UPDRS Part III Utilizadas

#### Modelo 1: Red Neuronal Simple (pruebas.py)
El modelo simple utiliza **6 variables clínicas** específicas del MDS-UPDRS Part III:

- `NP3RIGRU`: Rigidez extremidad superior derecha
- `NP3RIGLU`: Rigidez extremidad superior izquierda  
- `NP3RIGRL`: Rigidez extremidad inferior derecha
- `NP3RIGLL`: Rigidez extremidad inferior izquierda
- `NP3HMOVR`: Movimientos de manos derecha
- `NP3HMOVL`: Movimientos de manos izquierda

#### Modelo 2: Vision Transformer Multimodal (pruebas.ipynb)
El modelo multimodal utiliza **14 variables clínicas** del MDS-UPDRS Part III:

- `DBSYN`: Síntomas de discinesia
- `NP3SPCH`: Habla
- `NP3FACXP`: Expresión facial
- `NP3RIGN`: Rigidez en cuello
- `NP3RIGRU`: Rigidez extremidad superior derecha
- `NP3RIGLU`: Rigidez extremidad superior izquierda
- `NP3RIGRL`: Rigidez extremidad inferior derecha
- `NP3RIGLL`: Rigidez extremidad inferior izquierda
- `NP3HMOVR`: Movimientos de manos derecha
- `NP3HMOVL`: Movimientos de manos izquierda
- `NP3GAIT`: Marcha
- `NP3FRZGT`: Congelamiento de la marcha
- `NP3PSTBL`: Estabilidad postural
- `NP3POSTR`: Postura

### Pipeline de Preprocesamiento

#### 1. Manejo de Valores Faltantes
```python
# Imputación robusta por mediana para variables numéricas
for col in columnas_a_usar:
    if combined_df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(combined_df[col]):
            median_val = combined_df[col].median()
            combined_df[col] = combined_df[col].fillna(median_val)
```

#### 2. Normalización por Fold
```python
# Normalización independiente para cada fold (evita data leakage)
scaler = StandardScaler()
X_train_scaled_fold = scaler.fit_transform(X_train_fold)
X_test_scaled_fold = scaler.transform(X_test_fold)
```

#### 3. Características del Preprocesamiento

- **Robustez**: Imputación por mediana para valores faltantes
- **Normalización**: StandardScaler para evitar dominancia de variables con mayor escala
- **Validación**: Estratificación para mantener proporciones de clases
- **Sincronización**: Alineación de datos clínicos con imágenes por ID de paciente (modelo multimodal)

## Integración en los Modelos de Clasificación

### Modelo 1: Red Neuronal Simple (pruebas.py)

#### Arquitectura:
```python
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(input_size, 64)    # 6 variables → 64 neuronas
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.3)
        self.layer_2 = nn.Linear(64, 32)            # 64 → 32 neuronas
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(0.3)
        self.output_layer = nn.Linear(32, 1)        # 32 → 1 salida
        self.sigmoid = nn.Sigmoid()
```

#### Flujo de integración:
- Las 6 variables clínicas se alimentan directamente como entrada
- Validación cruzada estratificada de 20 folds
- Normalización independiente en cada fold
- Entrenamiento con Adam optimizer (lr=0.001)

### Modelo 2: Vision Transformer Multimodal (pruebas.ipynb)

#### Integración multimodal:
```python
class ClinicalImageDataset(Dataset):
    # Carga simultánea de:
    # - Imágenes MRI
    # - Imágenes SPECT (reales y sintéticas)
    # - Variables clínicas (las 14 del MDS-UPDRS)
```

#### Procesamiento conjunto:
- **Imágenes**: Redimensionadas a 256x256, normalizadas
- **Variables clínicas**: Normalizadas con StandardScaler
- **Fusión**: Vision Transformer procesa embeddings separados para cada modalidad
- **Combinación**: Embeddings concatenados antes de la clasificación final

### Métricas de Evaluación:
- Accuracy, AUC-ROC, Precision, Recall, F1-Score
- Validación cruzada de 20 folds para robustez estadística
- Evaluación separada por clase (Control vs Parkinson)

Este enfoque garantiza que las variables clínicas del MDS-UPDRS Part III se integren de manera efectiva tanto en modelos unimodales como multimodales, manteniendo la integridad y calidad de los datos a lo largo del pipeline de machine learning.

## Metodología

### Preprocesamiento de Datos
1. **Limpieza**: Conversión a tipos numéricos y manejo de valores faltantes
2. **Imputación**: Valores faltantes reemplazados por la mediana
3. **Normalización**: StandardScaler para características numéricas
4. **Balanceado**: Validación cruzada estratificada para mantener proporciones de clases

### Entrenamiento
1. **División de Datos**: Validación cruzada 20-fold estratificada
2. **Arquitectura**: Redes profundas con regularización (dropout)
3. **Optimización**: Adam optimizer con learning rate adaptativo
4. **Early Stopping**: Implementado para evitar overfitting

## Estructura del Proyecto

```
Modelo_clasificacion/
├── README.md                           # Este archivo
├── pruebas.ipynb                       # Notebook principal
├── pruebas.py                          # Script de clasificación simple
├── embcExtension_control.csv           # Datos control
├── embcExtension.xlsx                  # Datos Parkinson
├── spect_sinteticos.csv               # Imágenes SPECT sintéticas
└── [otros archivos de datos]
```

## Contribuciones

Para contribuir al proyecto:
1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -am 'Agrega nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crea un Pull Request

## Licencia

[Especificar licencia del proyecto]

## Referencias

- MDS-UPDRS: Movement Disorder Society-Unified Parkinson's Disease Rating Scale
- PPMI: Parkinson's Progression Markers Initiative
- Vision Transformer: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- CycleGAN: "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"

## Contacto

[Información de contacto del desarrollador]
