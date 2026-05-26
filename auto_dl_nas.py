import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DynamicMLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, dropout, is_classification):
        super().__init__()
        self.is_classification = is_classification
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            
        if is_classification:
            layers.append(nn.Linear(hidden_dim, output_dim)) # Logits
        else:
            layers.append(nn.Linear(hidden_dim, 1)) # Regression
            
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def objective_nas(trial, X, y, problem_type, device, w1, w2, w3):
    # 1. Optuna suggests architecture
    num_layers = trial.suggest_int('dl_num_layers', 1, 4)
    hidden_dim = trial.suggest_categorical('dl_hidden_dim', [32, 64, 128, 256])
    dropout = trial.suggest_float('dl_dropout', 0.1, 0.5)
    lr = trial.suggest_float('dl_lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('dl_batch_size', [32, 64, 128, 256])
    
    # 2. Prepare Data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_t = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32).to(device)
    X_val_t = torch.tensor(scaler.transform(X_val), dtype=torch.float32).to(device)
    
    is_clf = (problem_type == 'classification')
    out_dim = len(np.unique(y)) if is_clf else 1
    
    if is_clf:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_val_enc = le.transform(y_val)
        y_train_t = torch.tensor(y_train_enc, dtype=torch.long).to(device)
        y_val_t = torch.tensor(y_val_enc, dtype=torch.long).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
        criterion = nn.MSELoss()

    # 3. Initialize Model & Optimizer
    model = DynamicMLP(X.shape[1], out_dim, num_layers, hidden_dim, dropout, is_clf).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training Loop (Keep epochs low for AutoML speed)
    epochs = 30 
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    
    start_time = time.time()
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start_time
            
    # 5. Evaluation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_loss = criterion(val_preds, y_val_t).item()
        
    # Convert loss to a comparable "score"
    if is_clf:
        _, predicted = torch.max(val_preds, 1)
        score = (predicted == y_val_t).float().mean().item() # Accuracy
    else:
        score = -np.sqrt(val_loss) # Negative RMSE to match sklearn convention
        
    # 6. Calculate Utility (Matching your Multi-Objective logic)
    norm_acc = max(0, min(1, score)) if is_clf else 1.0 / (1.0 + abs(score))
    norm_speed = 1.0 / (1.0 + train_time)
    complexity = sum(p.numel() for p in model.parameters())
    norm_complexity = 1.0 / (1.0 + complexity / 10000) # Scale down params
    
    utility = (w1 * norm_acc) + (w2 * norm_speed) + (w3 * norm_complexity)
    
    return utility
