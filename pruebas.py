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

# Load the control data
df_control = pd.read_csv("/home/Data/franklin_pupils/yeferson/Modelo_clasificacion/embcExtension_control.csv")

# Load the Parkinson's data
df_parkinson = pd.read_excel('/home/Data/franklin_pupils/yeferson/Modelo_clasificacion/embcExtension.xlsx', sheet_name='PD')

# Define the list of columns to use for classification
columnas_a_usar = ['NP3RIGRU', 'NP3RIGLU', 
                   'NP3RIGRL', 'NP3RIGLL', 'NP3HMOVR', 'NP3HMOVL']

# Preprocessing df_control
df_control['target'] = 0

# Preprocessing df_parkinson
df_parkinson['target'] = 1


# Combine dataframes, selecting only the required columns
combined_df = pd.concat([
    df_control[columnas_a_usar + ['target']], 
    df_parkinson[columnas_a_usar + ['target']]
], ignore_index=True)


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


# Prepare data for modeling
X = combined_df[columnas_a_usar]
y = combined_df['target']

# PyTorch specific part:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dims = X.shape[1]
epochs = 100 # Number of epochs for training each fold
batch_size = 16
lr = 0.001

# Define the PyTorch neural network model
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(input_size, 64)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.3)
        self.layer_2 = nn.Linear(64, 32)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(0.3)
        self.output_layer = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout_1(self.relu_1(self.layer_1(x)))
        x = self.dropout_2(self.relu_2(self.layer_2(x)))
        x = self.sigmoid(self.output_layer(x))
        return x

# Cross-validation setup
n_splits = 20 # Number of folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_test_losses = []
fold_accuracies = []
fold_aucs = []
fold_classification_reports = []


for fold, (train_idx, test_idx) in enumerate(tqdm(skf.split(X, y), total=n_splits, desc="Cross-validation Folds")):
    # print(f"\n--- Fold {fold+1}/{n_splits} ---")
    
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    # Scale numerical features for the current fold
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
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model for the current fold
    for epoch in range(epochs): # tqdm(range(epochs), desc=f"Fold {fold+1} Training", leave=False):
        model.train()
        for batch_X, batch_y in train_loader_fold:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluate the model for the current fold
    model.eval()
    with torch.no_grad():
        y_pred_proba_tensor = model(X_test_tensor)
        test_loss = criterion(y_pred_proba_tensor, y_test_tensor).item()
        y_pred_proba = y_pred_proba_tensor.cpu().numpy()
        y_pred_classes = (y_pred_proba > 0.5).astype(int)

    # Calculate metrics for the current fold
    accuracy = accuracy_score(y_test_fold.values, y_pred_classes)
    sklearn_auc = roc_auc_score(y_test_fold.values, y_pred_proba)
    report = classification_report(y_test_fold.values, y_pred_classes, target_names=['Control', 'Parkinson'], output_dict=True, zero_division=0)


    fold_test_losses.append(test_loss)
    fold_accuracies.append(accuracy)
    fold_aucs.append(sklearn_auc)
    fold_classification_reports.append(report)
    
    # print(f"Fold {fold+1} Test Loss: {test_loss:.4f}")
    # print(f"Fold {fold+1} Test Accuracy: {accuracy:.4f}")
    # print(f"Fold {fold+1} Test AUC: {sklearn_auc:.4f}")
    # print(f"Fold {fold+1} Classification Report:\n{classification_report(y_test_fold.values, y_pred_classes, target_names=['Control', 'Parkinson'], zero_division=0)}")


# Calculate and print average metrics across all folds
print(f"\n--- Cross-Validation Results ({n_splits} folds) ---")
print(f"Average Test Loss: {np.mean(fold_test_losses):.4f} +/- {np.std(fold_test_losses):.4f}")
print(f"Average Test Accuracy: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")
print(f"Average precision: {np.mean([rep['Control']['precision'] for rep in fold_classification_reports]):.4f} +/- {np.std([rep['Control']['precision'] for rep in fold_classification_reports]):.4f}")
print(f"Average recall: {np.mean([rep['Control']['recall'] for rep in fold_classification_reports]):.4f} +/- {np.std([rep['Control']['recall'] for rep in fold_classification_reports]):.4f}")
print(f"Average F1-score: {np.mean([rep['Control']['f1-score'] for rep in fold_classification_reports]):.4f} +/- {np.std([rep['Control']['f1-score'] for rep in fold_classification_reports]):.4f}")
print(f"Average Test AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")

# Averaging classification report metrics (example for 'Parkinson' class precision)
avg_precision_parkinson = np.mean([rep['Parkinson']['precision'] for rep in fold_classification_reports])
avg_recall_parkinson = np.mean([rep['Parkinson']['recall'] for rep in fold_classification_reports])
avg_f1_parkinson = np.mean([rep['Parkinson']['f1-score'] for rep in fold_classification_reports])

avg_precision_control = np.mean([rep['Control']['precision'] for rep in fold_classification_reports])
avg_recall_control = np.mean([rep['Control']['recall'] for rep in fold_classification_reports])
avg_f1_control = np.mean([rep['Control']['f1-score'] for rep in fold_classification_reports])

print("\nAverage Classification Metrics:")
print(f"  Control - Precision: {avg_precision_control:.4f}, Recall: {avg_recall_control:.4f}, F1-score: {avg_f1_control:.4f}")
print(f"  Parkinson - Precision: {avg_precision_parkinson:.4f}, Recall: {avg_recall_parkinson:.4f}, F1-score: {avg_f1_parkinson:.4f}")