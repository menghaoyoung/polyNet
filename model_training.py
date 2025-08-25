import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import json
import os
import time
from datetime import datetime
import warnings
import argparse

warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Basic settings
tokenizer = AutoTokenizer.from_pretrained('kuelumbus/polyBERT')
polyBERT = AutoModel.from_pretrained('kuelumbus/polyBERT').to(device)

def mean_pooling(model_output, attention_mask):
    """Mean pooling for polyBERT embeddings"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_polybert_dim():
    """Get actual output dimension of polyBERT"""
    test_smiles = "CCO"
    encoded_input = tokenizer([test_smiles], padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = polyBERT(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    actual_dim = embedding.shape[1]
    print(f"polyBERT actual output dimension: {actual_dim}")
    return actual_dim

class SMILESDataset(Dataset):
    """Dataset class for SMILES and material features"""
    
    def __init__(self, dataset_path, target_name, material_features=None, 
                 normalize_features=True, train_mode=True, scaler=None):
        self.dataset_path = dataset_path
        self.target_name = target_name
        self.material_features = material_features or []
        self.normalize_features = normalize_features
        self.train_mode = train_mode
        self.scaler = scaler
        
        if self.train_mode and self.normalize_features:
            self.feature_scaler = StandardScaler()
        elif not self.train_mode and scaler is not None:
            self.feature_scaler = scaler
        
        self.data = self.load_dataset()
        
    def smiles_to_embedding(self, smiles):
        """Convert SMILES to polyBERT embedding"""
        encoded_input = tokenizer([smiles], padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = polyBERT(**encoded_input)
        embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        return embedding.squeeze(0).cpu()
    
    def load_dataset(self):
        """Load and preprocess dataset"""
        df = pd.read_excel(self.dataset_path)
        data = []
        material_features_data = []
        
        print(f"Dataset shape: {df.shape}")
        print(f"Available columns: {df.columns.tolist()}")
        
        # Check available material feature columns
        available_features = []
        for feature in self.material_features:
            if feature in df.columns:
                available_features.append(feature)
            else:
                print(f"Warning: Feature '{feature}' not found in dataset")
        
        self.material_features = available_features
        print(f"Using material features: {self.material_features}")
        
        # Load data
        for idx, row in df.iterrows():
            try:
                smiles = row['SMILES']
                # Handle SMILES if it (unexpectedly) becomes a Series due to duplicate columns
                if isinstance(smiles, pd.Series):
                    smiles = smiles.dropna().iloc[0] if smiles.dropna().size > 0 else None
                if smiles is None or pd.isna(smiles):
                    print(f"Warning: Invalid SMILES at row {idx}, skipping.")
                    continue

                # Extract target value robustly (may be a Series if duplicate column names exist)
                target_raw = row[self.target_name]
                if isinstance(target_raw, pd.Series):
                    if len(target_raw) == 1:
                        target_raw = target_raw.iloc[0]
                    else:
                        target_raw = target_raw.dropna().iloc[0] if target_raw.dropna().size > 0 else np.nan
                try:
                    target_value = float(target_raw)
                except (TypeError, ValueError):
                    print(f"Warning: Invalid target at row {idx}: {target_raw}, skipping.")
                    continue
                if pd.isna(target_value):
                    print(f"Warning: NaN target at row {idx}, skipping.")
                    continue
                
                # Get polyBERT embedding
                polybert_embedding = self.smiles_to_embedding(smiles)
                
                # Get material features
                material_feat = []
                if self.material_features:
                    for feature in self.material_features:
                        feat_val = row[feature]
                        # Handle cases where a pandas Series (e.g., from duplicate columns) is returned
                        if isinstance(feat_val, pd.Series):
                            if len(feat_val) == 1:
                                feat_val = feat_val.iloc[0]
                            else:
                                # If multiple values exist, take the first non-null; fallback to NaN
                                feat_val = feat_val.dropna().iloc[0] if feat_val.dropna().size > 0 else np.nan
                        try:
                            feat_val_converted = float(feat_val)
                        except (TypeError, ValueError):
                            feat_val_converted = np.nan
                        material_feat.append(feat_val_converted)
                    material_features_data.append(material_feat)
                
                data.append({
                    'polybert_embedding': polybert_embedding,
                    'material_features': material_feat,
                    self.target_name: target_value,
                    'smiles': smiles,
                    'index': idx
                })
                
            except Exception as e:
                print(f"Warning: Failed to process row {idx}: {e}")
                continue
        
        # Normalize material features
        if self.material_features and material_features_data:
            material_features_array = np.array(material_features_data)
            
            if self.train_mode and self.normalize_features:
                material_features_normalized = self.feature_scaler.fit_transform(material_features_array)
            elif not self.train_mode and hasattr(self, 'feature_scaler'):
                material_features_normalized = self.feature_scaler.transform(material_features_array)
            else:
                material_features_normalized = material_features_array
            
            for i, item in enumerate(data):
                item['material_features'] = material_features_normalized[i].tolist()
        
        print(f"Successfully loaded {len(data)} samples")
        return data
    
    def get_scaler(self):
        """Return the feature scaler"""
        return getattr(self, 'feature_scaler', None)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        polybert_embedding = item['polybert_embedding']
        material_features = torch.tensor(item['material_features'], dtype=torch.float32) if item['material_features'] else torch.tensor([], dtype=torch.float32)
        target = torch.tensor(item[self.target_name], dtype=torch.float32)
        
        return polybert_embedding, material_features, target, item['smiles'], item['index']


class FeatureFusionModule(nn.Module):
    """Feature fusion module using attention mechanism and residual connections"""
    
    def __init__(self, polybert_dim, material_dim, fusion_dim=128):
        super().__init__()
        self.polybert_dim = polybert_dim
        self.material_dim = material_dim
        self.fusion_dim = fusion_dim
        
        # Material feature expansion network
        if material_dim > 0:
            # 修复：确保第一层输入维度正确
            self.material_expander = nn.Sequential(
                nn.Linear(material_dim, 32),  # 这里material_dim应该是实际特征数量
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, fusion_dim)
            )
        
        # polyBERT feature compression network
        self.polybert_compressor = nn.Sequential(
            nn.Linear(polybert_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, fusion_dim)
        )
        
        # Cross-attention mechanism
        if material_dim > 0:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=fusion_dim, 
                num_heads=4, 
                dropout=0.1, 
                batch_first=True
            )
            
            # Feature fusion layer
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            # Residual projection
            self.residual_proj = nn.Linear(fusion_dim * 2, fusion_dim)
        
    def forward(self, polybert_features, material_features=None):
        batch_size = polybert_features.shape[0]
        
        # Compress polyBERT features
        compressed_polybert = self.polybert_compressor(polybert_features)
        
        if self.material_dim > 0 and material_features is not None and material_features.numel() > 0:
            
            # 确保材料特征有正确的维度
            if material_features.dim() == 1:
                material_features = material_features.unsqueeze(0)
            
            # 确保批次维度正确
            if material_features.shape[0] != batch_size:
                material_features = material_features.repeat(batch_size, 1)
            
            # 确保特征维度正确
            if material_features.shape[1] != self.material_dim:
                raise ValueError(f"Material features dimension mismatch: got {material_features.shape[1]}, expected {self.material_dim}")
            
            # Expand material features
            expanded_material = self.material_expander(material_features)
            
            # Add sequence dimension for attention computation
            poly_seq = compressed_polybert.unsqueeze(1)
            mat_seq = expanded_material.unsqueeze(1)
            
            # Cross-attention
            poly_attended, _ = self.cross_attention(poly_seq, mat_seq, mat_seq)
            poly_attended = poly_attended.squeeze(1)
            
            mat_attended, _ = self.cross_attention(mat_seq, poly_seq, poly_seq)
            mat_attended = mat_attended.squeeze(1)
            
            # Feature fusion
            combined = torch.cat([poly_attended, mat_attended], dim=1)
            fused_features = self.fusion_layer(combined)
            
            # Residual connection
            residual = self.residual_proj(combined)
            final_features = fused_features + residual
            
            return final_features
        else:
            return compressed_polybert

class LightHybridRegressor(nn.Module):
    """Lightweight hybrid regression model"""
    
    def __init__(self, polybert_dim, material_dim=0):
        super().__init__()
        self.fusion_module = FeatureFusionModule(polybert_dim, material_dim, fusion_dim=64)
        
        self.predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )
    
    def forward(self, polybert_features, material_features=None):
        fused_features = self.fusion_module(polybert_features, material_features)
        return self.predictor(fused_features)

class ResidualHybridRegressor(nn.Module):
    """Residual connection hybrid regression model"""
    
    def __init__(self, polybert_dim, material_dim=0):
        super().__init__()
        self.fusion_module = FeatureFusionModule(polybert_dim, material_dim, fusion_dim=128)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._make_res_block(128) for _ in range(2)
        ])
        
        self.predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def _make_res_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
    
    def forward(self, polybert_features, material_features=None):
        x = self.fusion_module(polybert_features, material_features)
        
        # Residual blocks
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x) + residual
            x = torch.relu(x)
        
        return self.predictor(x)

class AttentionHybridRegressor(nn.Module):
    """Attention mechanism hybrid regression model"""
    
    def __init__(self, polybert_dim, material_dim=0):
        super().__init__()
        self.fusion_module = FeatureFusionModule(polybert_dim, material_dim, fusion_dim=128)
        
        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=128, 
            num_heads=8, 
            dropout=0.1, 
            batch_first=True
        )
        
        # Feature importance attention
        self.feature_attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Softmax(dim=1)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, polybert_features, material_features=None):
        x = self.fusion_module(polybert_features, material_features)
        
        # Self-attention
        x_seq = x.unsqueeze(1)
        attended_x, _ = self.self_attention(x_seq, x_seq, x_seq)
        attended_x = attended_x.squeeze(1)
        
        # Feature importance weights
        attention_weights = self.feature_attention(attended_x)
        weighted_features = attended_x * attention_weights
        
        return self.predictor(weighted_features)

class EnsembleHybridRegressor(nn.Module):
    """Ensemble hybrid regression model"""
    
    def __init__(self, polybert_dim, material_dim=0, n_models=3):
        super().__init__()
        self.fusion_module = FeatureFusionModule(polybert_dim, material_dim, fusion_dim=96)
        
        # Multiple prediction branches
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(96, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            ) for _ in range(n_models)
        ])
        
        # Ensemble weight learning
        self.ensemble_weights = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, n_models),
            nn.Softmax(dim=1)
        )
    
    def forward(self, polybert_features, material_features=None):
        fused_features = self.fusion_module(polybert_features, material_features)
        
        # Multiple predictor outputs
        predictions = [predictor(fused_features) for predictor in self.predictors]
        predictions = torch.cat(predictions, dim=1)
        
        # Learn ensemble weights
        weights = self.ensemble_weights(fused_features)
        
        # Weighted average
        final_pred = torch.sum(predictions * weights, dim=1, keepdim=True)
        
        return final_pred

class CNN1DHybridRegressor(nn.Module):
    """1D CNN hybrid regression model"""
    
    def __init__(self, polybert_dim, material_dim=0):
        super().__init__()
        self.fusion_module = FeatureFusionModule(polybert_dim, material_dim, fusion_dim=128)
        
        # 1D CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, polybert_features, material_features=None):
        fused_features = self.fusion_module(polybert_features, material_features)
        
        # Add channel dimension for 1D convolution
        x = fused_features.unsqueeze(1)
        
        # 1D convolution
        conv_out = self.conv_layers(x)
        conv_out = conv_out.squeeze(-1)
        
        return self.fc_layers(conv_out)

def create_model(model_type, polybert_dim, material_dim):
    """Create model based on type"""
    models = {
        'light': LightHybridRegressor,
        'residual': ResidualHybridRegressor,
        'attention': AttentionHybridRegressor,
        'ensemble': EnsembleHybridRegressor,
        'cnn1d': CNN1DHybridRegressor
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](polybert_dim, material_dim)


def train_test_split(dataset, test_size):
    """Split dataset into train/test only (no validation set)"""
    total_size = len(dataset)
    indices = torch.randperm(total_size).tolist()
    
    # Ensure we have enough samples
    if test_size >= total_size:
        raise ValueError(f"Test size ({test_size}) >= Total size ({total_size})")
    
    train_size = total_size - test_size
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, test_dataset

def train_model(model, train_loader, test_loader, config):
    """Train the model and evaluate on test set each epoch"""
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], 
                                weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma']
    )
    
    # Training history
    history = {
        'train_loss': [],
        'test_loss': [],
        'learning_rate': []
    }
    
    print(f"Starting training for {config['epochs']} epochs...")
    print("-" * 80)
    print(f"{'Epoch':>5} {'Train Loss':>12} {'Test Loss':>12} {'LR':>12} {'Time':>8}")
    print("-" * 80)
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        
        for batch_idx, (polybert_emb, material_feat, targets, _, _) in enumerate(train_loader):
            polybert_emb = polybert_emb.to(device)
            material_feat = material_feat.to(device) if material_feat.numel() > 0 else None
            targets = targets.to(device).unsqueeze(1)
            
            # Forward pass
            outputs = model(polybert_emb, material_feat)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            num_train_batches += 1
        
        avg_train_loss = train_loss / num_train_batches
        
        # Test evaluation phase
        model.eval()
        test_loss = 0.0
        num_test_batches = 0
        
        with torch.no_grad():
            for polybert_emb, material_feat, targets, _, _ in test_loader:
                polybert_emb = polybert_emb.to(device)
                material_feat = material_feat.to(device) if material_feat.numel() > 0 else None
                targets = targets.to(device).unsqueeze(1)
                
                outputs = model(polybert_emb, material_feat)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                num_test_batches += 1
        
        avg_test_loss = test_loss / num_test_batches
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['learning_rate'].append(current_lr)
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"{epoch+1:5d} {avg_train_loss:12.4f} {avg_test_loss:12.4f} {current_lr:12.2e} {epoch_time:8.1f}s")
    
    print("-" * 80)
    print(f"Training completed!")
    
    return model, history

def evaluate_model(model, test_loader, config):
    """Evaluate the model"""
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    all_smiles = []
    all_indices = []
    
    criterion = nn.MSELoss()
    test_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for polybert_emb, material_feat, targets, smiles, indices in test_loader:
            polybert_emb = polybert_emb.to(device)
            material_feat = material_feat.to(device) if material_feat.numel() > 0 else None
            targets = targets.to(device).unsqueeze(1)
            
            outputs = model(polybert_emb, material_feat)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            num_batches += 1
            
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            all_smiles.extend(smiles)
            all_indices.extend(indices.numpy())
    
    avg_test_loss = test_loss / num_batches
    
    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    r2 = r2_score(all_targets, all_predictions)
    
    # Calculate relative errors
    relative_errors = np.abs((np.array(all_targets) - np.array(all_predictions)) / np.array(all_targets)) * 100
    mean_relative_error = np.mean(relative_errors)
    
    metrics = {
        'test_loss': avg_test_loss,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mean_relative_error': mean_relative_error
    }
    
    results = {
        'predictions': all_predictions,
        'targets': all_targets,
        'smiles': all_smiles,
        'indices': all_indices,
        'metrics': metrics
    }
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Test Loss (MSE): {avg_test_loss:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Relative Error: {mean_relative_error:.2f}%")
    
    return results

def plot_training_history(history, save_path):
    """Plot training history with test loss"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Loss curves
    plt.subplot(1, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, history['test_loss'], label='Test Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Learning rate
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['learning_rate'], color='green', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Loss curves (log scale)
    plt.subplot(1, 3, 3)
    plt.semilogy(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.semilogy(epochs, history['test_loss'], label='Test Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Curves (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_training_history.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional detailed loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2, marker='o', markersize=3)
    plt.plot(epochs, history['test_loss'], label='Test Loss', color='red', linewidth=2, marker='s', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress: Train vs Test Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text annotations for best values
    min_train_loss = min(history['train_loss'])
    min_test_loss = min(history['test_loss'])
    min_train_epoch = history['train_loss'].index(min_train_loss) + 1
    min_test_epoch = history['test_loss'].index(min_test_loss) + 1
    
    plt.annotate(f'Min Train Loss: {min_train_loss:.4f}\nEpoch: {min_train_epoch}', 
                xy=(min_train_epoch, min_train_loss), xytext=(0.7, 0.9),
                textcoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    plt.annotate(f'Min Test Loss: {min_test_loss:.4f}\nEpoch: {min_test_epoch}', 
                xy=(min_test_epoch, min_test_loss), xytext=(0.7, 0.8),
                textcoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_detailed_loss.png", dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model_final(model, test_loader, config):
    """Final comprehensive evaluation of the model"""
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    all_smiles = []
    all_indices = []
    
    criterion = nn.MSELoss()
    test_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for polybert_emb, material_feat, targets, smiles, indices in test_loader:
            polybert_emb = polybert_emb.to(device)
            material_feat = material_feat.to(device) if material_feat.numel() > 0 else None
            targets = targets.to(device).unsqueeze(1)
            
            outputs = model(polybert_emb, material_feat)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            num_batches += 1
            
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            all_smiles.extend(smiles)
            all_indices.extend(indices.numpy())
    
    avg_test_loss = test_loss / num_batches
    
    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    r2 = r2_score(all_targets, all_predictions)
    
    # Calculate relative errors
    relative_errors = np.abs((np.array(all_targets) - np.array(all_predictions)) / np.array(all_targets)) * 100
    mean_relative_error = np.mean(relative_errors)
    
    metrics = {
        'test_loss': avg_test_loss,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mean_relative_error': mean_relative_error
    }
    
    results = {
        'predictions': all_predictions,
        'targets': all_targets,
        'smiles': all_smiles,
        'indices': all_indices,
        'metrics': metrics
    }
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    print(f"Test Loss (MSE): {avg_test_loss:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Relative Error: {mean_relative_error:.2f}%")
    
    return results

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def save_model_and_results(model, config, history, results, model_info, save_dir):
    """Save model and results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_info,
        'training_config': config,
        'polybert_dim': model_info['polybert_dim'],
        'material_dim': model_info['material_dim'],
        'model_type': model_info['model_type']
    }, model_path)
    
    # Save scaler if available
    if 'scaler' in model_info and model_info['scaler'] is not None:
        import pickle
        scaler_path = os.path.join(save_dir, 'feature_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(model_info['scaler'], f)
    
    # Save training history - convert numpy types
    history_converted = convert_numpy_types(history)
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_converted, f, indent=2)
    
    # Prepare results for saving - convert all numpy types
    results_to_save = {
        'metrics': convert_numpy_types(results['metrics']),
        'model_info': convert_numpy_types({
            k: v for k, v in model_info.items() 
            if k != 'scaler'  # Skip scaler as it's not JSON serializable
        }),
        'training_config': convert_numpy_types(config),
        'predictions': convert_numpy_types(results['predictions']),
        'targets': convert_numpy_types(results['targets']),
        'smiles': results['smiles'],  # Already strings
        'indices': convert_numpy_types(results['indices'])
    }
    
    results_path = os.path.join(save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Save detailed results CSV
    results_df = pd.DataFrame({
        'SMILES': results['smiles'],
        'True_Value': convert_numpy_types(results['targets']),
        'Predicted_Value': convert_numpy_types(results['predictions']),
        'Absolute_Error': [abs(float(t) - float(p)) for t, p in zip(results['targets'], results['predictions'])],
        'Relative_Error_%': [abs(float(t) - float(p)) / float(t) * 100 for t, p in zip(results['targets'], results['predictions'])],
        'Data_Index': convert_numpy_types(results['indices'])
    })
    
    csv_path = os.path.join(save_dir, 'detailed_results.csv')
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n📁 Model and results saved to: {save_dir}")
    print(f"   • Model: {model_path}")
    print(f"   • Results: {results_path}")
    print(f"   • Detailed CSV: {csv_path}")
    print(f"   • Training History: {history_path}")
    if 'scaler' in model_info and model_info['scaler'] is not None:
        print(f"   • Feature Scaler: {scaler_path}")

def plot_predictions(results, save_path, target_name):
    """Plot prediction results"""
    predictions = np.array(results['predictions'])
    targets = np.array(results['targets'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Predictions vs True values
    ax1 = axes[0, 0]
    ax1.scatter(targets, predictions, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate R²
    r2 = results['metrics']['r2']
    ax1.set_xlabel(f'True {target_name}')
    ax1.set_ylabel(f'Predicted {target_name}')
    ax1.set_title(f'Predictions vs True Values (R² = {r2:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals plot
    ax2 = axes[0, 1]
    residuals = predictions - targets
    ax2.scatter(targets, residuals, alpha=0.6, s=50)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel(f'True {target_name}')
    ax2.set_ylabel('Residuals (Predicted - True)')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Error Distribution (Mean: {np.mean(residuals):.3f})')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Relative error plot
    ax4 = axes[1, 1]
    relative_errors = np.abs(residuals / targets) * 100
    ax4.scatter(targets, relative_errors, alpha=0.6, s=50)
    ax4.axhline(y=np.mean(relative_errors), color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(relative_errors):.1f}%')
    ax4.set_xlabel(f'True {target_name}')
    ax4.set_ylabel('Relative Error (%)')
    ax4.set_title('Relative Error vs True Values')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional detailed prediction plot
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with color mapping based on error
    scatter = plt.scatter(targets, predictions, c=np.abs(residuals), 
                         cmap='viridis', alpha=0.7, s=60)
    
    # Perfect prediction line
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             label='Perfect Prediction', alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Absolute Error', rotation=270, labelpad=20)
    
    # Statistics text box
    mae = results['metrics']['mae']
    rmse = results['metrics']['rmse']
    stats_text = f'R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top', fontsize=12)
    
    plt.xlabel(f'True {target_name}')
    plt.ylabel(f'Predicted {target_name}')
    plt.title(f'Model Predictions vs True Values\nColor indicates absolute error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_detailed.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Error statistics summary
    print(f"\n📊 Prediction Analysis Summary:")
    print(f"   • Total samples: {len(predictions)}")
    print(f"   • Mean absolute error: {mae:.4f}")
    print(f"   • Root mean square error: {rmse:.4f}")
    print(f"   • R² coefficient: {r2:.4f}")
    print(f"   • Mean relative error: {np.mean(relative_errors):.2f}%")
    print(f"   • Median relative error: {np.median(relative_errors):.2f}%")
    print(f"   • Max absolute error: {np.max(np.abs(residuals)):.4f}")
    print(f"   • Min absolute error: {np.min(np.abs(residuals)):.4f}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="polyBERT 高分子性质回归训练脚本 (CLI 版本)"
    )
    parser.add_argument("--data_path", type=str, required=True, help="数据集路径 (Excel)")
    parser.add_argument("--target_col", type=str, default="Conductivity", help="目标列名")
    parser.add_argument("--features", type=str, default="WaterContent,SwellingRate,Degreeofpolymerization,ElongationatBreak,TensileStrength",
                        help="材料特征列(逗号分隔)")
    parser.add_argument("--model_type", type=str, default="light",
                        choices=["light","residual","attention","ensemble","cnn1d"],
                        help="模型类型")
    parser.add_argument("--save_dir", type=str, default="", help="保存目录(默认: trained_models/<model>_<time>)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr_step_size", type=int, default=30)
    parser.add_argument("--lr_gamma", type=float, default=0.7)
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="测试集大小: <1 视为比例, >=1 视为绝对数量")
    parser.add_argument("--no_normalize_features", action="store_true", help="关闭特征标准化")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = args.save_dir.strip() or f"trained_models/{args.model_type}_{timestamp}"

    material_features = [f.strip() for f in args.features.split(",") if f.strip()]
    print("="*60)
    print("polyBERT Feature Fusion Model Training (CLI)")
    print("="*60)
    print(f"数据集        : {args.data_path}")
    print(f"目标列        : {args.target_col}")
    print(f"材料特征      : {material_features}")
    print(f"模型类型      : {args.model_type}")
    print(f"保存目录      : {save_dir}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch Size    : {args.batch_size}")
    print(f"LR / Decay    : {args.lr} / {args.weight_decay}")
    print(f"LR Step/Gamma : {args.lr_step_size} / {args.lr_gamma}")
    print(f"Test Size Arg : {args.test_size}")
    print(f"特征标准化    : {not args.no_normalize_features}")
    print(f"随机种子      : {args.seed}")
    print("-"*60)

    try:
        print("\n获取 polyBERT 维度...")
        polybert_dim = get_polybert_dim()

        print("\n加载数据集...")
        full_dataset = SMILESDataset(
            dataset_path=args.data_path,
            target_name=args.target_col,
            material_features=material_features,
            normalize_features=not args.no_normalize_features,
            train_mode=True
        )
        material_dim = len(full_dataset.material_features)
        total_samples = len(full_dataset)
        if total_samples < 3:
            raise ValueError("样本量过小 (<3)")

        # 解析 test_size
        if args.test_size < 1:
            test_size = max(1, int(total_samples * args.test_size))
        else:
            test_size = int(args.test_size)
        if test_size >= total_samples:
            test_size = total_samples // 5 if total_samples >= 5 else 1
            print(f"自动调整 test_size={test_size}")

        train_size = total_samples - test_size
        print(f"\n数据划分: Train={train_size}  Test={test_size}  (Total={total_samples})")

        scaler = full_dataset.get_scaler()
        print("\n拆分数据集...")
        train_dataset, test_dataset = train_test_split(full_dataset, test_size)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        print(f"\n创建模型: {args.model_type}")
        model = create_model(args.model_type, polybert_dim, material_dim)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"参数总量: {total_params:,}")

        config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'lr_step_size': args.lr_step_size,
            'lr_gamma': args.lr_gamma
        }

        print("\n开始训练...")
        trained_model, history = train_model(model, train_loader, test_loader, config)

        print("\n最终评估...")
        results = evaluate_model_final(trained_model, test_loader, config)

        os.makedirs(save_dir, exist_ok=True)
        print("\n绘制曲线与结果图...")
        plot_training_history(history, os.path.join(save_dir, "training"))
        plot_predictions(results, os.path.join(save_dir, "predictions"), args.target_col)

        model_info = {
            'model_type': args.model_type,
            'polybert_dim': polybert_dim,
            'material_dim': material_dim,
            'material_features': full_dataset.material_features,
            'target_name': args.target_col,
            'dataset_path': args.data_path,
            'scaler': scaler,
            'timestamp': timestamp,
            'dataset_split': {
                'total_samples': total_samples,
                'train_size': train_size,
                'test_size': test_size
            }
        }
        save_model_and_results(trained_model, config, history, results, model_info, save_dir)

        print("\n训练总结:")
        print(f"最佳训练损失 : {min(history['train_loss']):.4f}")
        print(f"最佳测试损失 : {min(history['test_loss']):.4f}")
        print(f"最终测试 MSE : {results['metrics']['test_loss']:.4f}")
        print(f"MAE / RMSE   : {results['metrics']['mae']:.4f} / {results['metrics']['rmse']:.4f}")
        print(f"R2           : {results['metrics']['r2']:.4f}")
        print(f"结果目录     : {save_dir}")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()