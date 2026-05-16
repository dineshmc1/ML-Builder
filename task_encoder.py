"""
task_encoder.py

Trains a Siamese MLP encoder to map 17-dimensional handcrafted dataset meta-features
into a learned 32-dimensional embedding space using contrastive loss based on
the best-performing model family.
"""

import os
import random
import logging
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    raise ImportError("torch is not installed. Run: pip install torch")

from cold_start import MemoryStore

logger = logging.getLogger(__name__)

# --- Configuration ---

@dataclass
class TaskEncoderConfig:
    input_dim: int = 10
    hidden_dim: int = 64
    output_dim: int = 32
    margin: float = 1.0
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 32
    early_stopping_patience: int = 20
    encoder_save_path: str = "task_encoder.pt"

MODEL_FAMILY = {
    "tree_based": [
        "rf", "et_clf", "gb", "lgbm_clf", "xgb_clf", 
        "rf_reg", "et_reg", "gb_reg", "lgbm_reg", "xgb_reg", 
        "ada_clf", "ada_reg"
    ],
    "linear": [
        "logistic", "ridge", "lasso", "elastic", 
        "sgd_clf", "sgd_reg"
    ],
    "distance": [
        "knn_clf", "knn_reg"
    ],
    "neural_ml": [
        "mlp_clf", "mlp_reg"
    ],
    "kernel": [
        "svc", "svr"
    ],
    "bagging": [
        "bag_clf", "bag_reg"
    ]
}

def get_family(model_name: str) -> str:
    for fam, models in MODEL_FAMILY.items():
        if model_name in models:
            return fam
    return "unknown"

# --- Architecture ---

class SiameseEncoder(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # L2 Normalize
        return F.normalize(x, p=2, dim=1)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        # label: 1 for positive pair, 0 for negative pair
        euclidean_distance = F.pairwise_distance(out1, out2)
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

# --- Dataset ---

class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        v1, v2, label = self.pairs[idx]
        return (
            torch.tensor(v1, dtype=torch.float32), 
            torch.tensor(v2, dtype=torch.float32), 
            torch.tensor(label, dtype=torch.float32)
        )

def generate_pairs(store: MemoryStore) -> list:
    records = store.records
    if len(records) < 50:
        logger.warning(f"Only {len(records)} datasets available for pair generation. Results may be unstable.")

    family_map = {}
    for r in records:
        if not r.models:
            continue
        best_model = r.models[0]
        fam = get_family(best_model)
        if fam == "unknown":
            continue
        if fam not in family_map:
            family_map[fam] = []
        family_map[fam].append(r.embedding)

    if len(family_map) < 2:
        raise ValueError("MemoryStore must contain datasets representing at least 2 distinct model families to generate pairs.")

    pairs = []
    # 1:2 ratio positive to negative
    families = list(family_map.keys())
    
    # Generate Positives
    for fam, vectors in family_map.items():
        if len(vectors) > 1:
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    pairs.append((vectors[i], vectors[j], 1))
    
    num_pos = len(pairs)
    if num_pos == 0:
        logger.warning("No positive pairs could be generated (only 1 dataset per family).")
    
    target_neg = max(num_pos * 2, 100) # At least 100 negative pairs if pos is low
    neg_pairs = []
    attempts = 0
    while len(neg_pairs) < target_neg and attempts < target_neg * 10:
        attempts += 1
        f1, f2 = random.sample(families, 2)
        v1 = random.choice(family_map[f1])
        v2 = random.choice(family_map[f2])
        neg_pairs.append((v1, v2, 0))
        
    pairs.extend(neg_pairs)
    random.shuffle(pairs)
    return pairs

# --- Public API ---

def train_encoder(store: MemoryStore, config: TaskEncoderConfig = None, force_retrain: bool = False) -> Tuple[nn.Module, dict]:
    """Train the Siamese encoder on dataset pairs from the MemoryStore."""
    if config is None:
        config = TaskEncoderConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SiameseEncoder(
        input_dim=config.input_dim, 
        hidden_dim=config.hidden_dim, 
        output_dim=config.output_dim
    ).to(device)

    history_dict = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": 0,
        "stopped_early": False
    }

    if not force_retrain and os.path.exists(config.encoder_save_path):
        encoder.load_state_dict(torch.load(config.encoder_save_path, map_location=device, weights_only=True))
        encoder.eval()
        print(f"[TaskEncoder] Loaded existing encoder from {config.encoder_save_path}")
        return encoder, history_dict

    pairs = generate_pairs(store)
    if not pairs:
        raise ValueError("Failed to generate any pairs for training.")

    train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(PairDataset(train_pairs), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(PairDataset(val_pairs), batch_size=config.batch_size, shuffle=False)

    criterion = ContrastiveLoss(margin=config.margin)
    optimizer = optim.Adam(encoder.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, mode='min')

    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"[TaskEncoder] Starting training on {device}...")
    for epoch in range(1, config.epochs + 1):
        encoder.train()
        total_train_loss = 0.0
        for v1, v2, label in train_loader:
            v1, v2, label = v1.to(device), v2.to(device), label.to(device)
            optimizer.zero_grad()
            out1 = encoder(v1)
            out2 = encoder(v2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * v1.size(0)
            
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        history_dict["train_loss"].append(avg_train_loss)

        encoder.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for v1, v2, label in val_loader:
                v1, v2, label = v1.to(device), v2.to(device), label.to(device)
                out1 = encoder(v1)
                out2 = encoder(v2)
                loss = criterion(out1, out2, label)
                total_val_loss += loss.item() * v1.size(0)
                
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        history_dict["val_loss"].append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if epoch % 10 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            history_dict["best_epoch"] = epoch
            epochs_no_improve = 0
            torch.save(encoder.state_dict(), config.encoder_save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.early_stopping_patience:
                print(f"[TaskEncoder] Early stopping triggered at epoch {epoch}")
                history_dict["stopped_early"] = True
                break

    encoder.load_state_dict(torch.load(config.encoder_save_path, map_location=device, weights_only=True))
    encoder.eval()
    return encoder, history_dict

def encode_dataset(raw_vec: np.ndarray, encoder: nn.Module) -> np.ndarray:
    """Forward pass raw 17-dim vector through the Siamese encoder."""
    device = next(encoder.parameters()).device
    encoder.eval()
    
    # Handle both (17,) and (1, 17)
    tensor = torch.tensor(raw_vec, dtype=torch.float32).to(device)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
        
    with torch.no_grad():
        out = encoder(tensor)
        
    return out.cpu().numpy().flatten()

def encode_all(store: MemoryStore, encoder: nn.Module) -> dict:
    """Encode all datasets currently in memory and return a mapping of key -> 32-dim vector."""
    learned_vectors = {}
    for r in store.records:
        learned_vectors[r.key] = encode_dataset(r.embedding, encoder)
    return learned_vectors
