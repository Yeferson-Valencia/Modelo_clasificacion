"""
Modelo de Clasificación Simple para Enfermedad de Parkinson

Este script implementa una red neuronal feedforward para clasificar
pacientes con enfermedad de Parkinson basándose en características
clínicas del MDS-UPDRS Part III.

Características principales:
- Validación cruzada estratificada de 20 folds
- Red neuronal de 3 capas con regularización
- Evaluación completa con múltiples métricas
- Manejo robusto de datos faltantes

Autor: [Yeferson Valencia]
Fecha: [2024]
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# ========================================
# CONFIGURACIÓN Y CARGA DE DATOS
# ========================================

# Load the control data
df_control = pd.read_csv("/home/Data/franklin_pupils/yeferson/Modelo_clasificacion/embcExtension_control.csv")

# Load the Parkinson's data
df_parkinson = pd.read_excel('/home/Data/franklin_pupils/yeferson/Modelo_clasificacion/embcExtension.xlsx', sheet_name='PD')

# Define the list of columns to use for classification
# Estas características corresponden a elementos del MDS-UPDRS Part III:
# - NP3RIGRU/LU/RL/LL: Rigidez en extremidades (superior derecha/izquierda, inferior derecha/izquierda)
# - NP3HMOVR/L: Movimientos de manos (derecha/izquierda)
columnas_a_usar = ['NP3RIGRU', 'NP3RIGLU', 
                   'NP3RIGRL', 'NP3RIGLL', 'NP3HMOVR', 'NP3HMOVL']

# ========================================
# PREPROCESAMIENTO DE DATOS
# ========================================

# Preprocessing df_control
df_control['target'] = 0  # Sujetos control = clase 0

# Preprocessing df_parkinson
df_parkinson['target'] = 1  # Pacientes Parkinson = clase 1


# Combine dataframes, selecting only the required columns
combined_df = pd.concat([
    df_control[columnas_a_usar + ['target']], 
    df_parkinson[columnas_a_usar + ['target']]
], ignore_index=True)

# ========================================
# MANEJO DE VALORES FALTANTES Y LIMPIEZA
# ========================================


for col in columnas_a_usar:
    if combined_df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(combined_df[col]):
            median_val = combined_df[col].median()
            combined_df[col] = combined_df[col].fillna(median_val)
        else:
            print(f"Warning: Column {col} is not numeric and has NaNs. Filling with 0. Review imputation strategy.")
            combined_df[col] = combined_df[col].fillna(0) 

# Ensure all feature columns are numeric
for col in columnas_a_usar:
    if not pd.api.types.is_numeric_dtype(combined_df[col]):
        print(f"Error: Column {col} is not numeric after preprocessing. Type: {combined_df[col].dtype}")
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        if combined_df[col].isnull().any(): 
             combined_df[col] = combined_df[col].fillna(combined_df[col].median()) 


# ========================================
# CONFIGURACIÓN DEL MODELO Y PARÁMETROS
# ========================================

# Prepare data for modeling
X = combined_df[columnas_a_usar]
y = combined_df['target']

# PyTorch specific part:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Hiperparámetros del modelo
input_dims = X.shape[1]          # 6 características de entrada
epochs = 100                     # Épocas de entrenamiento por fold
batch_size = 16                  # Tamaño del batch
lr = 0.001                       # Learning rate para Adam optimizer

print(f"Configuración del modelo:")
print(f"- Características de entrada: {input_dims}")
print(f"- Épocas por fold: {epochs}")
print(f"- Tamaño de batch: {batch_size}")
print(f"- Learning rate: {lr}")

# ========================================
# DEFINICIÓN DE LA ARQUITECTURA DEL MODELO
# ========================================

# Define the PyTorch neural network model
class Net(nn.Module):
    """
    Red neuronal feedforward para clasificación binaria de Parkinson.
    
    Arquitectura:
    - Capa 1: 6 → 64 neuronas + ReLU + Dropout(0.3)
    - Capa 2: 64 → 32 neuronas + ReLU + Dropout(0.3)  
    - Salida: 32 → 1 neurona + Sigmoid
    
    La arquitectura está diseñada para:
    - Capturar patrones no lineales en los datos clínicos
    - Prevenir overfitting con regularización (dropout)
    - Producir probabilidades para clasificación binaria
    """
    def __init__(self, input_size):
        super(Net, self).__init__()
        # Primera capa oculta
        self.layer_1 = nn.Linear(input_size, 64)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.3)
        
        # Segunda capa oculta
        self.layer_2 = nn.Linear(64, 32)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(0.3)
        
        # Capa de salida
        self.output_layer = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Propagación hacia adelante.
        
        Args:
            x: Tensor de entrada [batch_size, input_size]
            
        Returns:
            Probabilidades de pertenencia a la clase positiva (Parkinson)
        """
        x = self.dropout_1(self.relu_1(self.layer_1(x)))
        x = self.dropout_2(self.relu_2(self.layer_2(x)))
        x = self.sigmoid(self.output_layer(x))
        return x

# ========================================
# CONFIGURACIÓN DE VALIDACIÓN CRUZADA
# ========================================

# Cross-validation setup
n_splits = 20  # Número de folds para validación cruzada
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"Validación cruzada estratificada con {n_splits} folds")
print(f"Total de muestras: {len(X)}")
print(f"Distribución de clases: Control={sum(y==0)}, Parkinson={sum(y==1)}")

# Listas para almacenar métricas de cada fold
fold_test_losses = []
fold_accuracies = []
fold_aucs = []
fold_classification_reports = []

# ========================================
# ENTRENAMIENTO CON VALIDACIÓN CRUZADA
# ========================================


for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(X, y), total=n_splits, desc="Cross-validation Folds")):
    """
    Entrenamiento y evaluación para cada fold de la validación cruzada.
    
    Para cada fold:
    1. División de datos (entrenamiento/prueba)
    2. Normalización independiente de características
    3. Conversión a tensores PyTorch
    4. Entrenamiento del modelo
    5. Evaluación y cálculo de métricas
    """
    # print(f"\n--- Fold {fold+1}/{n_splits} ---")
    
    # División de datos para el fold actual
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    # Scale numerical features for the current fold
    # IMPORTANTE: El escalado se hace independientemente para cada fold
    # para evitar data leakage entre conjuntos de entrenamiento y prueba
    scaler = StandardScaler()
    X_train_scaled_fold = scaler.fit_transform(X_train_fold)
    X_test_scaled_fold = scaler.transform(X_test_fold)

    # Convert data to PyTorch Tensors for the current fold
    X_train_tensor = torch.FloatTensor(X_train_scaled_fold).to(device)
    y_train_tensor = torch.FloatTensor(y_train_fold.values.reshape(-1, 1)).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled_fold).to(device)
    y_test_tensor = torch.FloatTensor(y_test_fold.values.reshape(-1, 1)).to(device)

    # Create DataLoader for training for the current fold
    train_dataset_fold = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)

    # Instantiate the model, define loss function and optimizer for the current fold
    model = Net(input_dims).to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy para clasificación binaria
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model for the current fold
    for epoch in range(epochs): # tqdm(range(epochs), desc=f"Fold {fold+1} Training", leave=False):
        model.train()  # Modo entrenamiento (activa dropout)
        for batch_X, batch_y in train_loader_fold:
            optimizer.zero_grad()           # Resetear gradientes acumulados
            outputs = model(batch_X)        # Forward pass
            loss = criterion(outputs, batch_y)  # Calcular pérdida
            loss.backward()                 # Backward pass (calcular gradientes)
            optimizer.step()                # Actualizar pesos del modelo
    
    # Evaluate the model for the current fold
    model.eval()  # Modo evaluación (desactiva dropout)
    with torch.no_grad():  # Desactivar cálculo de gradientes para evaluación
        y_pred_proba_tensor = model(X_test_tensor)
        test_loss = criterion(y_pred_proba_tensor, y_test_tensor).item()
        y_pred_proba = y_pred_proba_tensor.cpu().numpy()
        y_pred_classes = (y_pred_proba > 0.5).astype(int)  # Umbral de decisión 0.5

    # Calculate metrics for the current fold
    accuracy = accuracy_score(y_test_fold.values, y_pred_classes)
    sklearn_auc = roc_auc_score(y_test_fold.values, y_pred_proba)
    report = classification_report(y_test_fold.values, y_pred_classes, 
                                 target_names=['Control', 'Parkinson'], 
                                 output_dict=True, zero_division=0)

    # Almacenar métricas del fold actual
    fold_test_losses.append(test_loss)
    fold_accuracies.append(accuracy)
    fold_aucs.append(sklearn_auc)
    fold_classification_reports.append(report)
    
    # Opcional: imprimir métricas por fold (comentado para output más limpio)
    # print(f"Fold {fold+1} Test Loss: {test_loss:.4f}")
    # print(f"Fold {fold+1} Test Accuracy: {accuracy:.4f}")
    # print(f"Fold {fold+1} Test AUC: {sklearn_auc:.4f}")
    # print(f"Fold {fold+1} Classification Report:\n{classification_report(y_test_fold.values, y_pred_classes, target_names=['Control', 'Parkinson'], zero_division=0)}")

# ========================================
# ANÁLISIS DE RESULTADOS FINALES
# ========================================


# Calculate and print average metrics across all folds
print(f"\n{'='*60}")
print(f"RESULTADOS FINALES - VALIDACIÓN CRUZADA ({n_splits} folds)")
print(f"{'='*60}")

print(f"Average Test Loss: {np.mean(fold_test_losses):.4f} +/- {np.std(fold_test_losses):.4f}")
print(f"Average Test Accuracy: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")
print(f"Average precision: {np.mean([rep['Control']['precision'] for rep in fold_classification_reports]):.4f} +/- {np.std([rep['Control']['precision'] for rep in fold_classification_reports]):.4f}")
print(f"Average recall: {np.mean([rep['Control']['recall'] for rep in fold_classification_reports]):.4f} +/- {np.std([rep['Control']['recall'] for rep in fold_classification_reports]):.4f}")
print(f"Average F1-score: {np.mean([rep['Control']['f1-score'] for rep in fold_classification_reports]):.4f} +/- {np.std([rep['Control']['f1-score'] for rep in fold_classification_reports]):.4f}")
print(f"Average Test AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")

# ========================================
# MÉTRICAS DETALLADAS POR CLASE
# ========================================

# Averaging classification report metrics (example for 'Parkinson' class precision)
avg_precision_parkinson = np.mean([rep['Parkinson']['precision'] for rep in fold_classification_reports])
avg_recall_parkinson = np.mean([rep['Parkinson']['recall'] for rep in fold_classification_reports])
avg_f1_parkinson = np.mean([rep['Parkinson']['f1-score'] for rep in fold_classification_reports])

avg_precision_control = np.mean([rep['Control']['precision'] for rep in fold_classification_reports])
avg_recall_control = np.mean([rep['Control']['recall'] for rep in fold_classification_reports])
avg_f1_control = np.mean([rep['Control']['f1-score'] for rep in fold_classification_reports])

print("\nMétricas Promedio por Clase:")
print(f"  Control - Precision: {avg_precision_control:.4f}, Recall: {avg_recall_control:.4f}, F1-score: {avg_f1_control:.4f}")
print(f"  Parkinson - Precision: {avg_precision_parkinson:.4f}, Recall: {avg_recall_parkinson:.4f}, F1-score: {avg_f1_parkinson:.4f}")

print(f"\n{'='*60}")
print("INTERPRETACIÓN DE RESULTADOS:")
print(f"{'='*60}")
print(f"• AUC = {np.mean(fold_aucs):.4f}: {'Excelente' if np.mean(fold_aucs) > 0.8 else 'Bueno' if np.mean(fold_aucs) > 0.7 else 'Moderado'} capacidad discriminativa")
print(f"• Accuracy = {np.mean(fold_accuracies):.4f}: {np.mean(fold_accuracies)*100:.1f}% de predicciones correctas")
print(f"• La desviación estándar indica la {'alta' if np.std(fold_aucs) > 0.1 else 'baja'} variabilidad entre folds")
print(f"• Modelo {'balanceado' if abs(avg_precision_control - avg_precision_parkinson) < 0.1 else 'desbalanceado'} entre clases")
print(f"{'='*60}")