# Documentación del Archivo pruebas.py

## Descripción General

El archivo `pruebas.py` implementa un modelo de red neuronal simple para la clasificación binaria de la enfermedad de Parkinson utilizando únicamente características clínicas del MDS-UPDRS Part III. Este script proporciona una implementación robusta con validación cruzada estratificada y evaluación completa de métricas.

## Arquitectura del Modelo

### Red Neuronal (Clase Net)

```python
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(input_size, 64)      # Capa entrada: 6 → 64
        self.relu_1 = nn.ReLU()                       # Activación ReLU
        self.dropout_1 = nn.Dropout(0.3)              # Regularización 30%
        self.layer_2 = nn.Linear(64, 32)              # Capa oculta: 64 → 32
        self.relu_2 = nn.ReLU()                       # Activación ReLU
        self.dropout_2 = nn.Dropout(0.3)              # Regularización 30%
        self.output_layer = nn.Linear(32, 1)          # Capa salida: 32 → 1
        self.sigmoid = nn.Sigmoid()                   # Activación sigmoid
```

**Características de la Arquitectura:**
- **Capas**: 3 capas completamente conectadas (6 → 64 → 32 → 1)
- **Activación**: ReLU para capas ocultas, Sigmoid para salida
- **Regularización**: Dropout (30%) en cada capa oculta
- **Salida**: Probabilidad única para clasificación binaria

### Forward Pass
```python
def forward(self, x):
    x = self.dropout_1(self.relu_1(self.layer_1(x)))    # Primera capa + activación + dropout
    x = self.dropout_2(self.relu_2(self.layer_2(x)))    # Segunda capa + activación + dropout
    x = self.sigmoid(self.output_layer(x))               # Capa de salida + sigmoid
    return x
```

## Preprocesamiento de Datos

### 1. Carga de Datos
```python
# Datos de sujetos control
df_control = pd.read_csv("/path/to/embcExtension_control.csv")

# Datos de pacientes con Parkinson
df_parkinson = pd.read_excel('/path/to/embcExtension.xlsx', sheet_name='PD')
```

### 2. Características Utilizadas
```python
columnas_a_usar = ['NP3RIGRU', 'NP3RIGLU', 
                   'NP3RIGRL', 'NP3RIGLL', 
                   'NP3HMOVR', 'NP3HMOVL']
```

**Descripción de las Características:**
- `NP3RIGRU`: Rigidez de extremidad superior derecha
- `NP3RIGLU`: Rigidez de extremidad superior izquierda
- `NP3RIGRL`: Rigidez de extremidad inferior derecha
- `NP3RIGLL`: Rigidez de extremidad inferior izquierda
- `NP3HMOVR`: Movimientos de mano derecha
- `NP3HMOVL`: Movimientos de mano izquierda

### 3. Etiquetado de Clases
```python
df_control['target'] = 0      # Sujetos control
df_parkinson['target'] = 1    # Pacientes con Parkinson
```

### 4. Combinación de Datasets
```python
combined_df = pd.concat([
    df_control[columnas_a_usar + ['target']], 
    df_parkinson[columnas_a_usar + ['target']]
], ignore_index=True)
```

### 5. Manejo de Valores Faltantes
```python
for col in columnas_a_usar:
    if combined_df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(combined_df[col]):
            median_val = combined_df[col].median()
            combined_df[col] = combined_df[col].fillna(median_val)
        else:
            combined_df[col] = combined_df[col].fillna(0)
```

**Estrategia de Imputación:**
- **Datos Numéricos**: Reemplazo por la mediana
- **Datos No Numéricos**: Reemplazo por 0 (con advertencia)

### 6. Conversión a Tipos Numéricos
```python
for col in columnas_a_usar:
    if not pd.api.types.is_numeric_dtype(combined_df[col]):
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        if combined_df[col].isnull().any(): 
             combined_df[col] = combined_df[col].fillna(combined_df[col].median())
```

## Configuración del Modelo

### Parámetros de Entrenamiento
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU si disponible
input_dims = X.shape[1]          # Dimensiones de entrada (6 características)
epochs = 100                     # Épocas de entrenamiento por fold
batch_size = 16                  # Tamaño de batch
lr = 0.001                       # Learning rate
```

### Configuración de Validación Cruzada
```python
n_splits = 20                    # Número de folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
```

**Características de la Validación Cruzada:**
- **Tipo**: Estratificada (mantiene proporciones de clases)
- **Folds**: 20 divisiones
- **Reproducibilidad**: Semilla fija (random_state=42)
- **Shuffle**: Datos mezclados antes de dividir

## Pipeline de Entrenamiento

### 1. Loop Principal de Validación Cruzada
```python
for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(X, y), total=n_splits)):
    # División de datos para el fold actual
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
```

### 2. Normalización por Fold
```python
scaler = StandardScaler()
X_train_scaled_fold = scaler.fit_transform(X_train_fold)
X_test_scaled_fold = scaler.transform(X_test_fold)
```

**Importante**: La normalización se hace independientemente para cada fold para evitar data leakage.

### 3. Conversión a Tensores PyTorch
```python
X_train_tensor = torch.FloatTensor(X_train_scaled_fold).to(device)
y_train_tensor = torch.FloatTensor(y_train_fold.values.reshape(-1, 1)).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled_fold).to(device)
y_test_tensor = torch.FloatTensor(y_test_fold.values.reshape(-1, 1)).to(device)
```

### 4. Creación de DataLoader
```python
train_dataset_fold = TensorDataset(X_train_tensor, y_train_tensor)
train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
```

### 5. Inicialización del Modelo para cada Fold
```python
model = Net(input_dims).to(device)
criterion = nn.BCELoss()                    # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=lr)
```

### 6. Loop de Entrenamiento
```python
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader_fold:
        optimizer.zero_grad()               # Resetear gradientes
        outputs = model(batch_X)            # Forward pass
        loss = criterion(outputs, batch_y)  # Calcular pérdida
        loss.backward()                     # Backward pass
        optimizer.step()                    # Actualizar pesos
```

### 7. Evaluación del Modelo
```python
model.eval()
with torch.no_grad():
    y_pred_proba_tensor = model(X_test_tensor)
    test_loss = criterion(y_pred_proba_tensor, y_test_tensor).item()
    y_pred_proba = y_pred_proba_tensor.cpu().numpy()
    y_pred_classes = (y_pred_proba > 0.5).astype(int)  # Umbral 0.5
```

## Métricas de Evaluación

### Cálculo de Métricas por Fold
```python
accuracy = accuracy_score(y_test_fold.values, y_pred_classes)
sklearn_auc = roc_auc_score(y_test_fold.values, y_pred_proba)
report = classification_report(y_test_fold.values, y_pred_classes, 
                             target_names=['Control', 'Parkinson'], 
                             output_dict=True, zero_division=0)
```

### Almacenamiento de Resultados
```python
fold_test_losses.append(test_loss)
fold_accuracies.append(accuracy)
fold_aucs.append(sklearn_auc)
fold_classification_reports.append(report)
```

## Análisis de Resultados

### Métricas Promediadas
```python
print(f"\n--- Cross-Validation Results ({n_splits} folds) ---")
print(f"Average Test Loss: {np.mean(fold_test_losses):.4f} +/- {np.std(fold_test_losses):.4f}")
print(f"Average Test Accuracy: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")
print(f"Average Test AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
```

### Métricas Detalladas por Clase
```python
# Control
avg_precision_control = np.mean([rep['Control']['precision'] for rep in fold_classification_reports])
avg_recall_control = np.mean([rep['Control']['recall'] for rep in fold_classification_reports])
avg_f1_control = np.mean([rep['Control']['f1-score'] for rep in fold_classification_reports])

# Parkinson
avg_precision_parkinson = np.mean([rep['Parkinson']['precision'] for rep in fold_classification_reports])
avg_recall_parkinson = np.mean([rep['Parkinson']['recall'] for rep in fold_classification_reports])
avg_f1_parkinson = np.mean([rep['Parkinson']['f1-score'] for rep in fold_classification_reports])
```

## Interpretación de Resultados

### Ejemplo de Salida
```
--- Cross-Validation Results (20 folds) ---
Average Test Loss: 0.6234 +/- 0.0892
Average Test Accuracy: 0.7456 +/- 0.1123
Average precision: 0.7234 +/- 0.1034
Average recall: 0.7891 +/- 0.0987
Average F1-score: 0.7545 +/- 0.0876
Average Test AUC: 0.8012 +/- 0.0654

Average Classification Metrics:
  Control - Precision: 0.7234, Recall: 0.7891, F1-score: 0.7545
  Parkinson - Precision: 0.7789, Recall: 0.7123, F1-score: 0.7434
```

### Significado de las Métricas

**Loss (Pérdida):**
- Mide qué tan lejos están las predicciones de las etiquetas reales
- Valores más bajos indican mejor ajuste

**Accuracy (Precisión):**
- Porcentaje de predicciones correctas
- Rango: 0-1 (0%-100%)

**AUC-ROC:**
- Área bajo la curva ROC
- Mide la capacidad de discriminación del modelo
- Rango: 0-1 (0.5 = random, 1.0 = perfecto)

**Precision:**
- De todas las predicciones positivas, cuántas fueron correctas
- Importante para minimizar falsos positivos

**Recall (Sensitivity):**
- De todos los casos positivos reales, cuántos fueron detectados
- Importante para minimizar falsos negativos

**F1-Score:**
- Media armónica entre precision y recall
- Útil cuando las clases están desbalanceadas

## Fortalezas del Modelo

1. **Robustez Estadística**: 20-fold cross-validation proporciona estimaciones confiables
2. **Simplicidad**: Modelo interpretable con pocas capas
3. **Regularización**: Dropout previene overfitting
4. **Evaluación Completa**: Múltiples métricas para análisis integral
5. **Reproducibilidad**: Semillas fijas para resultados consistentes

## Limitaciones y Consideraciones

1. **Datos Limitados**: Solo 6 características clínicas
2. **Arquitectura Simple**: Podría beneficiarse de mayor complejidad
3. **Desbalance de Clases**: No implementa técnicas específicas de balanceado
4. **Hiperparámetros Fijos**: No incluye optimización de hiperparámetros
5. **Validación**: No incluye conjunto de validación separado

## Posibles Mejoras

1. **Feature Engineering**: Agregar más características clínicas
2. **Ensemble Methods**: Combinar múltiples modelos
3. **Hyperparameter Tuning**: Optimización sistemática de parámetros
4. **Class Balancing**: SMOTE u otras técnicas de balanceado
5. **Early Stopping**: Implementar parada temprana durante entrenamiento

## Uso Recomendado

Este script es ideal para:
- **Baseline Model**: Establecer una línea base de rendimiento
- **Prototipado Rápido**: Evaluación rápida de nuevas características
- **Validación de Datos**: Verificar calidad y consistencia de los datos
- **Análisis Estadístico**: Obtener métricas robustas con intervalos de confianza
- **Investigación Clínica**: Evaluación de marcadores clínicos simples