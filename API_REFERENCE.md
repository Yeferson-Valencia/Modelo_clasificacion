# API y Referencia Técnica

## Clases y Funciones Principales

### 1. pruebas.py

#### Clase `Net`
```python
class Net(nn.Module):
    """
    Red neuronal feedforward para clasificación binaria de Parkinson.
    
    Arquitectura: 6 → 64 → 32 → 1
    Activaciones: ReLU (ocultas), Sigmoid (salida)
    Regularización: Dropout (0.3)
    """
    
    def __init__(self, input_size: int):
        """
        Inicializa la red neuronal.
        
        Args:
            input_size (int): Número de características de entrada (6)
        """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante.
        
        Args:
            x (torch.Tensor): Tensor de entrada [batch_size, input_size]
            
        Returns:
            torch.Tensor: Probabilidades de salida [batch_size, 1]
        """
```

### 2. pruebas.ipynb

#### Clase `CCycleGAN`
```python
class CCycleGAN:
    """
    Conditional CycleGAN para generación de imágenes SPECT sintéticas.
    
    Implementa un GAN condicional que puede generar imágenes SPECT 
    de diferentes clases (Control/Parkinson) con consistencia cíclica.
    """
    
    def __init__(self, csv_path: str, img_rows: int = 256, img_cols: int = 256,
                 channels: int = 1, num_classes: int = 2, **kwargs):
        """
        Inicializa el Conditional CycleGAN.
        
        Args:
            csv_path (str): Ruta al archivo CSV con datos de entrenamiento
            img_rows (int): Alto de las imágenes (default: 256)
            img_cols (int): Ancho de las imágenes (default: 256) 
            channels (int): Número de canales (default: 1)
            num_classes (int): Número de clases (default: 2)
            **kwargs: Parámetros adicionales de configuración
        """
    
    def build_discriminator(self) -> Model:
        """
        Construye el discriminador con doble salida.
        
        Returns:
            Model: Modelo de Keras del discriminador
        """
    
    def build_generator_enc_dec(self) -> Tuple[Model, Model]:
        """
        Construye el generador (encoder-decoder).
        
        Returns:
            Tuple[Model, Model]: (encoder, decoder)
        """
    
    def train(self, epochs: int, batch_size: int = 1, save_interval: int = 50):
        """
        Entrena el modelo CycleGAN.
        
        Args:
            epochs (int): Número de épocas de entrenamiento
            batch_size (int): Tamaño del batch (default: 1)
            save_interval (int): Intervalo para guardar checkpoints (default: 50)
        """
```

#### Clase `MultiModalVisionTransformer`
```python
class MultiModalVisionTransformer(nn.Module):
    """
    Vision Transformer multimodal para clasificación de Parkinson.
    
    Combina imágenes MRI, SPECT y datos clínicos utilizando 
    arquitectura Transformer con atención multi-cabeza.
    """
    
    def __init__(self, img_size: int = 256, patch_size: int = 16, 
                 in_chans: int = 1, num_classes: int = 2, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, clinical_dim: int = 14,
                 use_embeddings: bool = True):
        """
        Inicializa el Vision Transformer multimodal.
        
        Args:
            img_size (int): Tamaño de las imágenes (default: 256)
            patch_size (int): Tamaño de los patches (default: 16)
            in_chans (int): Canales de entrada (default: 1)
            num_classes (int): Número de clases de salida (default: 2)
            embed_dim (int): Dimensión de embedding (default: 768)
            depth (int): Número de capas del transformer (default: 12)
            num_heads (int): Número de cabezas de atención (default: 12)
            clinical_dim (int): Dimensión de datos clínicos (default: 14)
            use_embeddings (bool): Usar embeddings para datos clínicos (default: True)
        """
    
    def forward(self, mri_img: torch.Tensor, spect_img: torch.Tensor, 
                clinical: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante multimodal.
        
        Args:
            mri_img (torch.Tensor): Imágenes MRI [batch_size, 1, 256, 256]
            spect_img (torch.Tensor): Imágenes SPECT [batch_size, 1, 256, 256]
            clinical (torch.Tensor): Datos clínicos [batch_size, clinical_dim]
            
        Returns:
            torch.Tensor: Logits de clasificación [batch_size, num_classes]
        """
```

#### Clase `ClinicalImageDataset`
```python
class ClinicalImageDataset(Dataset):
    """
    Dataset personalizado para datos multimodales de Parkinson.
    
    Maneja la carga y sincronización de imágenes MRI, SPECT 
    y datos clínicos para entrenamiento multimodal.
    """
    
    def __init__(self, img_csv_path: str, clinical_control_csv: str,
                 clinical_pd_excel: str, img_res: Tuple[int, int] = (256, 256),
                 mode: str = 'train', img_col: str = 'mri',
                 clinical_features: List[str] = None, 
                 spect_synth_path: str = None):
        """
        Inicializa el dataset multimodal.
        
        Args:
            img_csv_path (str): Ruta al CSV con rutas de imágenes
            clinical_control_csv (str): Ruta al CSV de datos control
            clinical_pd_excel (str): Ruta al Excel de datos Parkinson
            img_res (Tuple[int, int]): Resolución de imágenes (default: (256, 256))
            mode (str): Modo del dataset ('train'/'test', default: 'train')
            img_col (str): Columna de imágenes a usar (default: 'mri')
            clinical_features (List[str]): Lista de características clínicas
            spect_synth_path (str): Ruta a imágenes SPECT sintéticas
        """
    
    def __len__(self) -> int:
        """Retorna el tamaño del dataset."""
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, 
                                           torch.Tensor, torch.Tensor]:
        """
        Obtiene un elemento del dataset.
        
        Args:
            idx (int): Índice del elemento
            
        Returns:
            Tuple: (mri_tensor, spect_tensor, clinical_tensor, label_tensor)
        """
```

## Funciones Utilitarias

### Preprocesamiento de Datos

```python
def preprocess_clinical_data(df: pd.DataFrame, 
                           features: List[str]) -> pd.DataFrame:
    """
    Preprocesa datos clínicos para el modelo.
    
    Args:
        df (pd.DataFrame): DataFrame con datos clínicos
        features (List[str]): Lista de características a usar
        
    Returns:
        pd.DataFrame: DataFrame preprocesado
    """
    for col in features:
        # Conversión a numérico
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Imputación de valores faltantes con mediana
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df
```

```python
def create_combined_dataset(control_df: pd.DataFrame, 
                          parkinson_df: pd.DataFrame,
                          features: List[str]) -> pd.DataFrame:
    """
    Combina datasets de control y Parkinson.
    
    Args:
        control_df (pd.DataFrame): Datos de sujetos control
        parkinson_df (pd.DataFrame): Datos de pacientes con Parkinson
        features (List[str]): Características a incluir
        
    Returns:
        pd.DataFrame: Dataset combinado con etiquetas
    """
    control_df['target'] = 0
    parkinson_df['target'] = 1
    
    combined_df = pd.concat([
        control_df[features + ['target']], 
        parkinson_df[features + ['target']]
    ], ignore_index=True)
    
    return combined_df
```

### Evaluación y Métricas

```python
def evaluate_model_cv(model_class, X: pd.DataFrame, y: pd.Series,
                     n_splits: int = 20, random_state: int = 42,
                     **model_params) -> Dict[str, List[float]]:
    """
    Evalúa un modelo usando validación cruzada estratificada.
    
    Args:
        model_class: Clase del modelo a evaluar
        X (pd.DataFrame): Características de entrada
        y (pd.Series): Etiquetas de destino
        n_splits (int): Número de folds (default: 20)
        random_state (int): Semilla para reproducibilidad (default: 42)
        **model_params: Parámetros adicionales del modelo
        
    Returns:
        Dict[str, List[float]]: Diccionario con métricas por fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                         random_state=random_state)
    
    results = {
        'test_losses': [],
        'accuracies': [],
        'aucs': [],
        'precisions': [],
        'recalls': [],
        'f1_scores': []
    }
    
    for train_idx, test_idx in skf.split(X, y):
        # Implementación de entrenamiento y evaluación
        pass
    
    return results
```

```python
def calculate_average_metrics(results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Calcula métricas promedio y desviación estándar.
    
    Args:
        results (Dict[str, List[float]]): Resultados de validación cruzada
        
    Returns:
        Dict[str, Dict[str, float]]: Métricas promedio con std
    """
    avg_metrics = {}
    
    for metric, values in results.items():
        avg_metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return avg_metrics
```

### Visualización

```python
def plot_training_curves(train_losses: List[float], 
                        train_accuracies: List[float],
                        val_losses: List[float] = None,
                        val_accuracies: List[float] = None,
                        figsize: Tuple[int, int] = (12, 5)):
    """
    Visualiza curvas de entrenamiento.
    
    Args:
        train_losses (List[float]): Pérdidas de entrenamiento
        train_accuracies (List[float]): Accuracies de entrenamiento
        val_losses (List[float], optional): Pérdidas de validación
        val_accuracies (List[float], optional): Accuracies de validación
        figsize (Tuple[int, int]): Tamaño de la figura (default: (12, 5))
    """
    epochs_range = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=figsize)
    
    # Gráfico de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Training Loss")
    if val_losses:
        plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    
    # Gráfico de accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label="Training Accuracy")
    if val_accuracies:
        plt.plot(epochs_range, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
```

## Configuraciones Recomendadas

### Para pruebas.py (Modelo Simple)
```python
# Configuración óptima para datasets pequeños
CONFIG_SIMPLE = {
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 0.001,
    'dropout_rate': 0.3,
    'n_splits': 20,
    'random_state': 42
}
```

### Para Vision Transformer (Modelo Complejo)
```python
# Configuración para modelo multimodal
CONFIG_TRANSFORMER = {
    'img_size': 256,
    'patch_size': 16,
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'epochs': 10,
    'batch_size': 4,  # Reducido por memoria GPU
    'learning_rate': 0.001
}
```

### Para CycleGAN
```python
# Configuración para generación de imágenes
CONFIG_CYCLEGAN = {
    'img_size': (256, 256),
    'channels': 1,
    'num_classes': 2,
    'epochs': 200,
    'batch_size': 4,
    'd_gan_loss_w': 1,
    'd_cl_loss_w': 1,
    'g_gan_loss_w': 1,
    'g_cl_loss_w': 1,
    'rec_loss_w': 1,
    'adam_lr': 0.0002,
    'adam_beta_1': 0.5,
    'adam_beta_2': 0.999
}
```

## Mejores Prácticas

### 1. Manejo de Memoria
```python
# Para GPUs con memoria limitada
torch.cuda.empty_cache()  # Liberar memoria GPU
batch_size = 4           # Reducir tamaño de batch
gradient_accumulation = 4 # Acumular gradientes
```

### 2. Reproducibilidad
```python
# Establecer semillas para reproducibilidad
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
```

### 3. Monitoreo de Entrenamiento
```python
# Usar callbacks para monitoreo
from torch.utils.tensorboard import SummaryWriter

def log_metrics(writer: SummaryWriter, metrics: Dict, epoch: int):
    """Log métricas a TensorBoard."""
    for name, value in metrics.items():
        writer.add_scalar(name, value, epoch)
```

### 4. Validación Robusta
```python
# Implementar métricas adicionales
from sklearn.metrics import confusion_matrix, roc_curve

def comprehensive_evaluation(y_true, y_pred, y_prob):
    """Evaluación completa del modelo."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred)
    }
    return metrics
```

Esta API proporciona una referencia completa para utilizar y extender los modelos implementados en el proyecto.