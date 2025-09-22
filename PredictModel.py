from __future__ import annotations

import os
import json
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

from dataset import mean_pooling
from FiLM_model import create_film_model


def _device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_tokenizer_model(device: torch.device):
    tok = AutoTokenizer.from_pretrained('kuelumbus/polyBERT')
    mdl = AutoModel.from_pretrained('kuelumbus/polyBERT').to(device)
    mdl.eval()
    return tok, mdl


def _embed_smiles_batch(smiles_list: List[str], tokenizer, polyBERT, device: torch.device, batch_size: int = 16) -> torch.Tensor:
    embs: List[torch.Tensor] = []
    polyBERT.eval()
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
            out = polyBERT(**enc)
            emb = mean_pooling(out, enc['attention_mask'])  # (B, D)
            embs.append(emb.detach().cpu())
    return torch.cat(embs, dim=0) if embs else torch.empty(0)


@dataclass
class LoadedArtifacts:
    model_path: str
    model_type: str
    poly_dim: int
    material_dim: int
    material_features: List[str]
    scaler: Optional[StandardScaler]
    log_target: bool


class Predictor:
    """Notebook-friendly predictor.

    Usage (in notebook):
        from PredictModel import Predictor
        pred = Predictor.load('trained_models/film_60C/model.pth')
        pred.predict_single('CCO', {'WaterContent': 0.5, ...})
        pred.predict_dataframe(df, smiles_col='SMILES')
    """

    def __init__(self, artifacts: LoadedArtifacts, model: nn.Module, tokenizer, polyBERT, device: torch.device):
        self.artifacts = artifacts
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.polyBERT = polyBERT
        self.device = device

    # -----------------
    # Construction
    # -----------------
    @staticmethod
    def load(model_path_or_dir: str, *, device: Optional[torch.device] = None,
             film_overrides: Optional[Dict[str, Any]] = None) -> 'Predictor':
        device = device or _device()
        # resolve model file path
        model_path = model_path_or_dir
        if os.path.isdir(model_path_or_dir):
            model_path = os.path.join(model_path_or_dir, 'model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        chkpt = torch.load(model_path, map_location='cpu')
        model_info = chkpt.get('model_info', {})
    model_type = 'film'
    poly_dim = int(model_info.get('poly_dim', 768))
    material_dim = int(model_info.get('material_dim', 0))

        # try to read material features and log_target from results.json
        model_dir = os.path.dirname(model_path)
        material_features: List[str] = []
        log_target = False
        results_json = os.path.join(model_dir, 'results.json')
        if os.path.exists(results_json):
            try:
                with open(results_json, 'r') as f:
                    res = json.load(f)
                mi = res.get('model_info', {})
                material_features = list(mi.get('material_features', []))
                tr_cfg = res.get('training_config', {})
                log_target = bool(tr_cfg.get('log_target', False))
            except Exception:
                pass

        # load scaler if present
        scaler = None
        scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

        # build model
        # 只保留film模型，且只用dense的film head
        film_cfg = dict(
            use_full_poly=True,
            d_model=128,
            cond_hidden=(32, 64),
            head_hidden=(64, 32),
            dropout=0.1,
            film_mode='dense',
            film_rank=16,
            film_groups=None,
            film_scale=0.1,
            add_poly_shortcut=True,
        )
        if film_overrides:
            film_cfg.update(film_overrides)
        model = create_film_model(poly_dim, material_dim, **film_cfg)

        # load weights
        state_dict = chkpt.get('model_state_dict', chkpt)
        model.load_state_dict(state_dict, strict=True)

        # tokenizer + polyBERT
        tokenizer, polyBERT = _load_tokenizer_model(device)

        artifacts = LoadedArtifacts(
            model_path=model_path,
            model_type=model_type,
            poly_dim=poly_dim,
            material_dim=material_dim,
            material_features=material_features,
            scaler=scaler,
            log_target=log_target,
        )
        return Predictor(artifacts, model, tokenizer, polyBERT, device)

    # -----------------
    # Prediction APIs
    # -----------------
    def _prep_features(self, mat_props_list: Optional[List[Optional[Dict[str, float]]]], n: int) -> Optional[torch.Tensor]:
        if self.artifacts.material_dim <= 0:
            return None
        feats = []
        names = self.artifacts.material_features or []
        for i in range(n):
            props = (mat_props_list[i] if mat_props_list and i < len(mat_props_list) else None) or {}
            row = []
            for k in names:
                v = props.get(k, 0.0)
                try:
                    row.append(float(v))
                except Exception:
                    row.append(0.0)
            feats.append(row)
        arr = np.asarray(feats, dtype=np.float32)
        if self.artifacts.scaler is not None and arr.size > 0:
            arr = self.artifacts.scaler.transform(arr)
        return torch.tensor(arr, dtype=torch.float32)

    def predict_single(self, smiles: str, material_properties: Optional[Dict[str, float]] = None) -> float:
        preds = self.predict_batch([smiles], [material_properties] if material_properties is not None else None)
        return float(preds[0])

    def predict_batch(self, smiles_list: List[str], material_properties_list: Optional[List[Optional[Dict[str, float]]]] = None, *, batch_size: int = 16) -> List[float]:
        self.model.eval()
        # Embed SMILES in batches
        poly_emb = _embed_smiles_batch(smiles_list, self.tokenizer, self.polyBERT, self.device, batch_size=batch_size)  # (N, D)
        if poly_emb.numel() == 0:
            return []
        mat_tensor = self._prep_features(material_properties_list, poly_emb.shape[0])
        if mat_tensor is not None:
            mat_tensor = mat_tensor.to(self.device)

        preds: List[float] = []
        with torch.no_grad():
            for i in range(0, poly_emb.shape[0], batch_size):
                p = poly_emb[i:i + batch_size].to(self.device)
                m = mat_tensor[i:i + batch_size] if mat_tensor is not None else None
                out = self.model(p, m)
                if self.artifacts.log_target:
                    out = torch.expm1(out)
                preds.extend(out.detach().cpu().numpy().reshape(-1).tolist())
        return [float(x) for x in preds]

    def predict_dataframe(self, df: pd.DataFrame, smiles_col: str = 'SMILES', *, batch_size: int = 16, output_col: str = 'Predicted_Conductivity') -> pd.DataFrame:
        if smiles_col not in df.columns:
            raise KeyError(f"Column '{smiles_col}' not found in dataframe")
        smiles = df[smiles_col].astype(str).tolist()

        mat_list: Optional[List[Dict[str, float]]] = None
        if self.artifacts.material_features:
            mat_list = []
            for _, row in df.iterrows():
                props = {}
                for k in self.artifacts.material_features:
                    props[k] = float(row[k]) if k in df.columns and pd.notna(row[k]) else 0.0
                mat_list.append(props)

        preds = self.predict_batch(smiles, mat_list, batch_size=batch_size)
        out_df = df.copy()
        out_df[output_col] = preds
        return out_df


# -------------
# CLI (optional)
# -------------
def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Predict using trained model (FiLM or hybrid)')
    p.add_argument('--model', type=str, required=True, help='Path to model file or its directory')
    p.add_argument('--csv', type=str, help='CSV to predict (must include SMILES and optional features)')
    p.add_argument('--smiles_col', type=str, default='SMILES')
    p.add_argument('--out', type=str, default='')
    return p.parse_args()


def main():
    args = _parse_args()
    pred = Predictor.load(args.model)
    if args.csv:
        df = pd.read_csv(args.csv)
        out_df = pred.predict_dataframe(df, smiles_col=args.smiles_col)
        out_path = args.out or args.csv.replace('.csv', '_pred.csv')
        out_df.to_csv(out_path, index=False)
        print(f"Saved predictions to {out_path}")
    else:
        print('Loaded model. Use from notebook or provide --csv for batch prediction.')


if __name__ == '__main__':
    main()

