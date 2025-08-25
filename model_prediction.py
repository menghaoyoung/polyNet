import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pickle
import json
import os
import argparse
from typing import List, Dict, Any
import warnings
from sklearn.preprocessing import StandardScaler
import torch.serialization

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

def smiles_to_embedding(smiles):
    """Convert SMILES to polyBERT embedding"""
    encoded_input = tokenizer([smiles], padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = polyBERT(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    return embedding.squeeze(0).cpu()

# ========== Model Architecture Classes (copied from training script) ==========

class FeatureFusionModule(nn.Module):
    """Feature fusion module using attention mechanism and residual connections"""
    
    def __init__(self, polybert_dim, material_dim, fusion_dim=128):
        super().__init__()
        self.polybert_dim = polybert_dim
        self.material_dim = material_dim
        self.fusion_dim = fusion_dim
        
        # Material feature expansion network
        if material_dim > 0:
            self.material_expander = nn.Sequential(
                nn.Linear(material_dim, 32),
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

# ========== Model Loading and Prediction Functions ==========

class ModelPredictor:
    """Class for loading trained models and making predictions"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.model_info = None
        self.scaler = None
        self.material_features = None
        
        self.load_model()
    
    def load_model(self):
        print(f"Loading model from: {self.model_path}")
        # 允许旧 checkpoint 中的 StandardScaler
        torch.serialization.add_safe_globals([StandardScaler])
        checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)

        # 兼容两种保存格式
        if 'model_config' in checkpoint and 'model_state_dict' in checkpoint:
            self.model_info = checkpoint['model_config']
            state_dict = checkpoint['model_state_dict']
        elif 'model_type' in checkpoint:
            # 可能直接把 config 扩展开保存
            self.model_info = checkpoint
            state_dict = checkpoint['state_dict']
        else:
            raise ValueError("不支持的 checkpoint 格式")

        model_type = self.model_info['model_type']
        polybert_dim = self.model_info['polybert_dim']
        material_dim = self.model_info['material_dim']
        self.material_features = self.model_info.get('material_features', [])
        print(f"Model type: {model_type}")
        print(f"polyBERT dimension: {polybert_dim}")
        print(f"Material dimension: {material_dim}")
        print(f"Material features: {self.material_features}")

        self.model = create_model(model_type, polybert_dim, material_dim).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # 优先独立 scaler 文件
        model_dir = os.path.dirname(self.model_path)
        scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Loaded feature scaler from: {scaler_path}")
        elif 'scaler' in checkpoint:
            # 旧版本打包在 checkpoint 中
            self.scaler = checkpoint['scaler']
            print("Loaded feature scaler from checkpoint (旧格式)")
        else:
            print("Warning: No feature scaler found. Using raw feature values.")

        print("✅ Model loaded successfully!")

    def predict_single(self, smiles: str, material_properties: Dict[str, float] = None):#type: ignore
        """Predict for a single SMILES string"""
        return self.predict_batch([smiles], [material_properties] if material_properties else None)[0]#type: ignore
    
    def predict_batch(self, smiles_list: List[str], material_properties_list: List[Dict[str, float]] = None):   #type: ignore
        predictions = []
        need_features = bool(self.material_features)
        with torch.no_grad():
            for i, smiles in enumerate(smiles_list):
                poly_vec = smiles_to_embedding(smiles).unsqueeze(0).to(device)

                mat_tensor = None
                if need_features:
                    # 取该样本特征字典
                    props = {}
                    if material_properties_list and i < len(material_properties_list) and material_properties_list[i]:
                        props = material_properties_list[i]
                    # 按训练顺序构建
                    values = []
                    for feat in self.material_features: #type: ignore
                        v = props.get(feat, 0.0)
                        try:
                            v = float(v)
                        except:
                            v = 0.0
                        values.append(v)
                    arr = np.array(values, dtype=np.float32).reshape(1, -1)

                    # 归一化
                    if self.scaler:
                        arr = self.scaler.transform(arr)
                    mat_tensor = torch.tensor(arr, dtype=torch.float32).to(device)
            
                # Make prediction
                output = self.model(poly_vec, mat_tensor)#type: ignore
                prediction = output.cpu().numpy().flatten()[0]
                predictions.append(prediction)
        
        return predictions
    
    def predict_from_csv(self, csv_path: str, smiles_column: str = 'SMILES', 
                        output_path: str = None):#type: ignore
        """Predict from CSV file"""
        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Extract SMILES
        smiles_list = df[smiles_column].tolist()
        
        # Extract material properties if available
        material_properties_list = []
        if self.material_features:
            for _, row in df.iterrows():
                properties = {}
                for feature in self.material_features:
                    if feature in df.columns:
                        properties[feature] = row[feature] if not pd.isna(row[feature]) else 0.0#type: ignore
                    else:
                        properties[feature] = 0.0
                material_properties_list.append(properties)
        else:
            material_properties_list = [None] * len(smiles_list)
        
        # Make predictions
        print(f"Making predictions for {len(smiles_list)} samples...")
        predictions = self.predict_batch(smiles_list, material_properties_list)#type: ignore
        
        # Add predictions to dataframe
        df['Predicted_Conductivity'] = predictions
        
        # Save results
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")
        
        return df

def interactive_prediction():
    """Interactive prediction interface"""
    print("="*60)
    print("🔮 polyBERT Model Prediction Interface")
    print("="*60)
    
    # Get model path
    model_path = input("Enter model file path (.pth): ").strip()
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Load model
    try:
        predictor = ModelPredictor(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    while True:
        print("\n" + "="*50)
        print("Prediction Options:")
        print("1. Single SMILES prediction")
        print("2. Batch prediction from CSV")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            # Single prediction
            smiles = input("\nEnter SMILES string: ").strip()
            if not smiles:
                print("❌ Please enter a valid SMILES string")
                continue
            
            # Get material properties if needed
            material_properties = {}
            if predictor.material_features:
                print(f"\nEnter material properties (press Enter to use default 0.0):")
                for feature in predictor.material_features:
                    value_input = input(f"  {feature}: ").strip()
                    if value_input:
                        try:
                            material_properties[feature] = float(value_input)
                        except ValueError:
                            print(f"Warning: Invalid value for {feature}, using 0.0")
                            material_properties[feature] = 0.0
                    else:
                        material_properties[feature] = 0.0
            
            # Make prediction
            try:
                prediction = predictor.predict_single(smiles, material_properties)
                print(f"\n🎯 Prediction Result:")
                print(f"   SMILES: {smiles}")
                if material_properties:
                    print(f"   Material Properties: {material_properties}")
                print(f"   Predicted Conductivity: {prediction:.4f}")
            except Exception as e:
                print(f"❌ Prediction error: {e}")
        
        elif choice == '2':
            # Batch prediction from CSV
            csv_path = input("\nEnter CSV file path: ").strip()
            if not os.path.exists(csv_path):
                print(f"❌ CSV file not found: {csv_path}")
                continue
            
            smiles_column = input("Enter SMILES column name (default: SMILES): ").strip()
            if not smiles_column:
                smiles_column = 'SMILES'
            
            output_path = input("Enter output CSV path (optional): ").strip()
            if not output_path:
                output_path = csv_path.replace('.csv', '_predictions.csv')
            
            try:
                results_df = predictor.predict_from_csv(csv_path, smiles_column, output_path)
                print(f"\n🎯 Batch Prediction Results:")
                print(f"   Processed samples: {len(results_df)}")
                print(f"   Mean prediction: {results_df['Predicted_Conductivity'].mean():.4f}")
                print(f"   Min prediction: {results_df['Predicted_Conductivity'].min():.4f}")
                print(f"   Max prediction: {results_df['Predicted_Conductivity'].max():.4f}")
                print(f"   Results saved to: {output_path}")
            except Exception as e:
                print(f"❌ Batch prediction error: {e}")
        
        elif choice == '3':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option. Please select 1-3.")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='polyBERT Model Prediction')
    parser.add_argument('--model', '-m', type=str, required=False,
                        help='Path to trained model (.pth file)')
    parser.add_argument('--smiles', '-s', type=str, required=False,
                        help='Single SMILES string to predict')
    parser.add_argument('--csv', '-c', type=str, required=False,
                        help='CSV file path for batch prediction')
    parser.add_argument('--smiles_col', type=str, default='SMILES',
                        help='SMILES column name in CSV (default: SMILES)')
    parser.add_argument('--output', '-o', type=str, required=False,
                        help='Output CSV file path')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive or (not args.model):
        interactive_prediction()
        return
    
    # Command line prediction
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    try:
        predictor = ModelPredictor(args.model)
        
        if args.smiles:
            # Single prediction
            prediction = predictor.predict_single(args.smiles)
            print(f"\n🎯 Prediction Result:")
            print(f"   SMILES: {args.smiles}")
            print(f"   Predicted Conductivity: {prediction:.4f}")
        
        elif args.csv:
            # Batch prediction
            if not os.path.exists(args.csv):
                print(f"❌ CSV file not found: {args.csv}")
                return
            
            output_path = args.output or args.csv.replace('.csv', '_predictions.csv')
            results_df = predictor.predict_from_csv(args.csv, args.smiles_col, output_path)
            
            print(f"\n🎯 Batch Prediction Results:")
            print(f"   Processed samples: {len(results_df)}")
            print(f"   Mean prediction: {results_df['Predicted_Conductivity'].mean():.4f}")
            print(f"   Results saved to: {output_path}")
        
        else:
            print("❌ Please provide either --smiles or --csv for prediction")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()