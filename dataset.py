"""
Reusable dataset module for polyBERT + tabular feature fusion.

Provides a SMILESDataset class that:
- Reads CSV/Excel with columns: 'SMILES', target column, and optional material features
- Encodes SMILES via a provided HuggingFace tokenizer + polyBERT model
- Handles feature selection, scaling (StandardScaler) without data leakage
- Returns tensors suitable for PyTorch training loops
- Supports optional on-disk caching of SMILES embeddings to speed up repeated runs

Example
-------
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from dataset import SMILESDataset, fit_scaler_on_indices

tokenizer = AutoTokenizer.from_pretrained('kuelumbus/polyBERT')
polyBERT = AutoModel.from_pretrained('kuelumbus/polyBERT').to(device)
polyBERT.eval()

ds = SMILESDataset(
    dataset_path='datasets/60C_dataset.xlsx',
    target_name='Conductivity',
    tokenizer=tokenizer,
    polyBERT=polyBERT,
    device=device,
    material_features=[
        'WaterContent','SwellingRate','Degreeofpolymerization',
        'ElongationatBreak','TensileStrength'
    ],
    scaler=None,
    cache_path='datasets/polybert_emb_cache.pkl',  # optional
)

# Fit scaler on train indices only, then set back to dataset to avoid leakage
train_idx = list(range(int(0.8 * len(ds))))
scaler = fit_scaler_on_indices(ds, train_idx)
ds.set_scaler(scaler)

loader = DataLoader(ds, batch_size=2, shuffle=True)
"""

from __future__ import annotations

import os
import math
import pickle
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool token embeddings with attention mask.

    Args:
        model_output: output from HF model (tuple with last hidden state at [0])
        attention_mask: (batch, seq_len)

    Returns:
        Tensor of shape (batch, hidden_size)
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )


def get_polybert_dim(tokenizer, polyBERT, device: torch.device) -> int:
    """Probe polyBERT embedding dimension using a dummy SMILES."""
    test_smiles = "CCO"
    encoded = tokenizer([test_smiles], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = polyBERT(**encoded)
    emb = mean_pooling(output, encoded["attention_mask"])  # (1, D)
    return int(emb.shape[1])


class SMILESDataset(Dataset):
    """Dataset that fuses polyBERT embeddings with tabular material features.

    Each item returns:
        poly_emb: torch.FloatTensor [D]  (CPU tensor)
        mat_feat: torch.FloatTensor [M]  (empty tensor if no features)
        target  : torch.FloatTensor []   (scalar)
        smiles  : str
        index   : int (original row index in the file)

    Notes
    -----
    - Embedding is computed once during initialization (no_grad, eval mode) and kept on CPU
    - If `scaler` is provided, material features are transformed at __getitem__ time
    - Optional on-disk cache (pickle dict) maps SMILES -> np.ndarray embedding
    """

    def __init__(
        self,
        dataset_path: str,
        target_name: str,
        tokenizer,
        polyBERT,
        device: torch.device,
        material_features: Optional[List[str]] = None,
        scaler: Optional[StandardScaler] = None,
        cache_path: Optional[str] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.target_name = target_name
        self.device = device
        self.tokenizer = tokenizer
        self.polyBERT = polyBERT
        self.material_features = list(material_features or [])
        self.scaler = scaler
        self.cache_path = cache_path

        # lazy-loaded/updated cache dict: {smiles: np.ndarray(D,)}
        self._cache: Dict[str, np.ndarray] = {}
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict):
                    # ensure values are numpy arrays
                    self._cache = {k: (np.asarray(v) if not isinstance(v, np.ndarray) else v) for k, v in obj.items()}
            except Exception:
                # corrupt cache shouldn't break training
                self._cache = {}

        self.data: List[Dict] = self._load()

        # Persist any newly created cache entries
        if self.cache_path:
            try:
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self._cache, f)
            except Exception:
                pass

    # -----------------
    # File IO utilities
    # -----------------
    def _read_df(self, path: str) -> pd.DataFrame:
        if path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        return pd.read_csv(path)

    # -----------------
    # Embedding helpers
    # -----------------
    def _smiles_to_embedding(self, smiles: str) -> torch.Tensor:
        # Check cache first
        if smiles in self._cache:
            arr = self._cache[smiles]
            return torch.from_numpy(arr).float()

        encoded = self.tokenizer([smiles], padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.polyBERT(**encoded)
        emb = mean_pooling(output, encoded["attention_mask"]).squeeze(0).detach().cpu()

        # update cache (numpy)
        self._cache[smiles] = emb.numpy()
        return emb

    # -----------------
    # Loading & access
    # -----------------
    def _load(self) -> List[Dict]:
        df = self._read_df(self.dataset_path)

        # keep only available feature columns
        if self.material_features:
            available = []
            for f in self.material_features:
                if f in df.columns:
                    available.append(f)
                else:
                    print(f"Warning: Feature '{f}' not in dataset columns; skipping.")
            self.material_features = available

        data: List[Dict] = []
        for idx, row in df.iterrows():
            smiles = row["SMILES"] if "SMILES" in df.columns else None
            if pd.isna(smiles) or smiles is None or str(smiles).strip() == "":
                continue

            try:
                poly = self._smiles_to_embedding(str(smiles))  # Tensor [D] on CPU
            except Exception as e:
                print(f"Warning: failed to embed SMILES at row {idx}: {e}")
                continue

            # target
            try:
                target_val = float(row[self.target_name])
            except Exception:
                continue
            if pd.isna(target_val):
                continue

            # material features (raw, before scaling)
            feats: List[float] = []
            for f in self.material_features:
                v = row[f] if f in row else np.nan
                try:
                    v = float(v)
                except Exception:
                    v = np.nan
                # convert NaN/None to 0.0 (mask-based enhancement can be added later if needed)
                feats.append(0.0 if (v is None or (isinstance(v, float) and math.isnan(v))) else v)

            data.append(
                {
                    "polybert_embedding": poly,  # Tensor [D]
                    "material_features": feats,  # list length M
                    "target": float(target_val),
                    "smiles": str(smiles),
                    "index": int(idx),
                }
            )

        print(f"Loaded {len(data)} samples from {self.dataset_path}")
        return data

    # -----------------
    # Public API
    # -----------------
    def set_scaler(self, scaler: Optional[StandardScaler]) -> None:
        """Set a StandardScaler for material features (fitted on training split)."""
        self.scaler = scaler

    @property
    def feature_names(self) -> List[str]:
        return list(self.material_features)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int]:
        item = self.data[idx]
        poly: torch.Tensor = item["polybert_embedding"]  # [D]
        feats: List[float] = item["material_features"]

        if self.scaler is not None and len(self.material_features) > 0:
            arr = np.asarray(feats, dtype=np.float32).reshape(1, -1)
            feats = self.scaler.transform(arr)[0].tolist()

        mat = (
            torch.tensor(feats, dtype=torch.float32)
            if len(self.material_features) > 0
            else torch.tensor([], dtype=torch.float32)
        )
        tgt = torch.tensor(item["target"], dtype=torch.float32)
        return poly, mat, tgt, item["smiles"], item["index"]


def fit_scaler_on_indices(dataset: SMILESDataset, indices: List[int]) -> Optional[StandardScaler]:
    """Fit a StandardScaler on a subset of dataset indices (to prevent leakage).

    Returns None if there are no material features.
    """
    if not dataset.feature_names:
        return None

    X: List[List[float]] = []
    for i in indices:
        feats = dataset.data[i]["material_features"]
        if feats:
            X.append(feats)
    if not X:
        return None

    X_np = np.asarray(X, dtype=np.float32)
    scaler = StandardScaler().fit(X_np)
    return scaler

