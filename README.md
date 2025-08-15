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
El proyecto utiliza las siguientes características clínicas del MDS-UPDRS Part III:
- `NP3RIGRU`: Rigidez extremidad superior derecha
- `NP3RIGLU`: Rigidez extremidad superior izquierda  
- `NP3RIGRL`: Rigidez extremidad inferior derecha
- `NP3RIGLL`: Rigidez extremidad inferior izquierda
- `NP3HMOVR`: Movimientos de manos derecha
- `NP3HMOVL`: Movimientos de manos izquierda

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
- Arquitectura: 6 → 64 → 32 → 1
- Funciones de activación: ReLU + Sigmoid
- Regularización: Dropout (0.3)
- Optimizador: Adam (lr=0.001)

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
