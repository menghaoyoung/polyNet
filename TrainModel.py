import os
import json
import math
import time
import pickle
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel

from dataset import SMILESDataset, fit_scaler_on_indices, get_polybert_dim
from FiLM_model import create_film_model


# ------------------------------
# Helpers
# ------------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_training_history(history: Dict, save_prefix: str):
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    epochs = list(range(1, len(history['train_loss']) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs, history['test_loss'], label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss.png", dpi=200)
    plt.close()


def plot_predictions(results: Dict, save_prefix: str, target_name: str):
    preds = np.array(results['predictions'])
    targs = np.array(results['targets'])
    min_v = float(min(targs.min(), preds.min()))
    max_v = float(max(targs.max(), preds.max()))
    plt.figure(figsize=(6, 6))
    plt.scatter(targs, preds, alpha=0.7)
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', label='Perfect')
    plt.xlabel(f'True {target_name}')
    plt.ylabel(f'Predicted {target_name}')
    plt.title('Predictions vs True')
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_scatter.png", dpi=200)
    plt.close()


def attribute_on_loader(model, loader, device, use_grad_x_input=False):
    model.eval()
    w2_list, w5_list, idx_all = [], [], []
    for poly, mat, tgt, _, idx in loader:
        poly = poly.to(device).requires_grad_(True)
        mat = mat.to(device) if mat.numel() > 0 else None
        if mat is not None:
            mat = mat.requires_grad_(True)
        out = model(poly, mat)
        out.sum().backward()
        g_poly = poly.grad.detach().abs().mean(dim=1)  # (B,)
        if mat is not None and mat.numel() > 0:
            g_mat = mat.grad.detach()
            if use_grad_x_input:
                g_mat = g_mat * mat.detach()
            g_mat = g_mat.abs()
            g_mat_s = g_mat.mean(dim=1)
            tot = g_poly + g_mat_s + 1e-12
            w2 = torch.stack([g_poly / tot, g_mat_s / tot], dim=1).cpu().numpy()
            w2_list.append(w2)
            g5 = g_mat.detach().cpu().numpy()
            g5 = g5 / (g5.sum(axis=1, keepdims=True) + 1e-12)
            w5_list.append(g5)
        else:
            w2 = torch.stack([torch.ones_like(g_poly), torch.zeros_like(g_poly)], dim=1).cpu().numpy()
            w2_list.append(w2)
        idx_all.extend(idx.numpy().tolist())
        model.zero_grad(set_to_none=True)
    w2_all = np.concatenate(w2_list, axis=0) if w2_list else None
    w5_all = np.concatenate(w5_list, axis=0) if w5_list else None
    return w2_all, w5_all, idx_all


def save_attribution_results(save_dir: str, sample_w2, sample_w5, idx_all, feature_names: Optional[List[str]] = None):
    os.makedirs(save_dir, exist_ok=True)
    if sample_w2 is not None:
        import pandas as pd
        df2 = pd.DataFrame(sample_w2, columns=['w_polyBERT', 'w_material'])
        df2['index'] = idx_all
        df2.to_csv(os.path.join(save_dir, 'attribution_w2.csv'), index=False)
    if sample_w5 is not None and feature_names is not None:
        import pandas as pd
        df5 = pd.DataFrame(sample_w5, columns=feature_names)
        df5['index'] = idx_all[:len(df5)]
        df5.to_csv(os.path.join(save_dir, 'attribution_w5.csv'), index=False)
    summary = {}
    if sample_w2 is not None:
        w2m = sample_w2.mean(axis=0)
        summary['w2_dataset_mean'] = {'w_polyBERT': float(w2m[0]), 'w_material': float(w2m[1])}
    if sample_w5 is not None and feature_names is not None:
        w5m = sample_w5.mean(axis=0)
        summary['w5_dataset_mean'] = {feature_names[i]: float(w5m[i]) for i in range(len(feature_names))}
    with open(os.path.join(save_dir, 'attribution_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def plot_combined_heatmap(save_dir: str):
    jp = os.path.join(save_dir, 'attribution_summary.json')
    if not os.path.exists(jp):
        print('No attribution_summary.json, skip heatmap.')
        return
    with open(jp, 'r') as f:
        summary = json.load(f)
    w2 = summary.get('w2_dataset_mean')
    w5 = summary.get('w5_dataset_mean')
    if w2 is None:
        return
    import pandas as pd
    if w5 is None:
        df = pd.DataFrame([{'material': w2['w_material'], 'polyBERT': w2['w_polyBERT']}])
        plt.figure(figsize=(4, 2))
        sns.heatmap(df, annot=True, cmap='viridis')
        plt.title('Attribution (w2)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'attribution_heatmap_w2.png'), dpi=200)
        plt.close()


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj


def save_model_and_results(model, config, history, results, model_info, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_info': {
            'poly_dim': model_info['poly_dim'],
            'material_dim': model_info['material_dim'],
            'model_type': model_info['model_type']
        }
    }, model_path)
    if model_info.get('scaler') is not None:
        with open(os.path.join(save_dir, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(model_info['scaler'], f)
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(convert_numpy_types(history), f, indent=2)
    results_to_save = {
        'metrics': convert_numpy_types(results['metrics']),
        'model_info': convert_numpy_types({k: v for k, v in model_info.items() if k != 'scaler'}),
        'training_config': convert_numpy_types(config),
        'predictions': convert_numpy_types(results['predictions']),
        'targets': convert_numpy_types(results['targets']),
        'smiles': results['smiles'],
        'indices': convert_numpy_types(results['indices'])
    }
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results_to_save, f, indent=2)
    import pandas as pd
    df = pd.DataFrame({'SMILES': results['smiles'], 'True_Value': results['targets'], 'Predicted_Value': results['predictions']})
    df['Absolute_Error'] = (df['True_Value'] - df['Predicted_Value']).abs()
    df['Relative_Error_%'] = (df['Absolute_Error'] / (pd.Series(results['targets']).replace(0, np.nan).abs())) * 100
    df.to_csv(os.path.join(save_dir, 'detailed_results.csv'), index=False)
    print(f"\nModel and results saved to: {save_dir}")
    print(f"   - Model: {model_path}")


# ------------------------------
# Train / Eval
# ------------------------------
def train_model(model, train_loader, test_loader, config, device):
    model = model.to(device)

    loss_name = str(config.get('loss', 'mse')).lower()
    if loss_name == 'smoothl1':
        criterion = nn.SmoothL1Loss(beta=1.0)
    elif loss_name == 'l1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    if str(config.get('optimizer', 'adam')).lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    sched_name = str(config.get('lr_scheduler', 'plateau')).lower()
    if sched_name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=float(config.get('plateau_factor', 0.5)),
            patience=int(config.get('plateau_patience', 10)), min_lr=float(config.get('plateau_min_lr', 1e-6))
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.get('lr_step_size', 30), gamma=config.get('lr_gamma', 0.7))

    history = {'train_loss': [], 'test_loss': [], 'learning_rate': []}
    print(f"Starting training for {config['epochs']} epochs...")
    print("-" * 80)
    print(f"{'Epoch':>5} {'Train Loss':>12} {'Test Loss':>12} {'LR':>12} {'Time':>8}")
    print("-" * 80)

    for epoch in range(config['epochs']):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        ntr = 0
        for poly, mat, tgt, _, _ in train_loader:
            poly = poly.to(device)
            mat = mat.to(device) if mat.numel() > 0 else None
            tgt_raw = tgt.to(device).unsqueeze(1)
            tgt_proc = torch.log1p(tgt_raw) if config.get('log_target', False) else tgt_raw

            pred = model(poly, mat)
            loss = criterion(pred, tgt_proc)

            optimizer.zero_grad()
            loss.backward()
            clip_val = float(config.get('clip_grad_norm', 0.0) or 0.0)
            if clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
            optimizer.step()

            tr_loss += loss.item()
            ntr += 1

        avg_tr = tr_loss / max(1, ntr)

        # 验证集（若无测试集，则用训练损失代替便于调度器工作）
        if test_loader is not None:
            model.eval()
            te_loss = 0.0
            nte = 0
            with torch.no_grad():
                for poly, mat, tgt, _, _ in test_loader:
                    poly = poly.to(device)
                    mat = mat.to(device) if mat.numel() > 0 else None
                    tgt_raw = tgt.to(device).unsqueeze(1)
                    tgt_proc = torch.log1p(tgt_raw) if config.get('log_target', False) else tgt_raw
                    pred = model(poly, mat)
                    loss = criterion(pred, tgt_proc)
                    te_loss += loss.item()
                    nte += 1
            avg_te = te_loss / max(1, nte)
        else:
            avg_te = avg_tr  # 无测试集时，用训练损失替代

        if sched_name == 'plateau':
            scheduler.step(avg_te)
        else:
            scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(avg_tr)
        history['test_loss'].append(avg_te)
        history['learning_rate'].append(lr)

        print(f"{epoch+1:5d} {avg_tr:12.4f} {avg_te:12.4f} {lr:12.2e} {time.time()-t0:8.1f}s")

    print("-" * 80)
    return model, history


def evaluate_model_final(model, test_loader, config, device):
    model.eval()
    preds, targs, smiles, indices = [], [], [], []
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    nb = 0
    with torch.no_grad():
        for poly, mat, tgt, s, idx in test_loader:
            poly = poly.to(device)
            mat = mat.to(device) if mat.numel() > 0 else None
            tgt_raw = tgt.to(device).unsqueeze(1)
            out = model(poly, mat)
            if config.get('log_target', False):
                out_orig = torch.expm1(out)
                loss = mse_loss(out_orig, tgt_raw)
                preds.extend(out_orig.cpu().numpy().flatten().tolist())
                targs.extend(tgt_raw.cpu().numpy().flatten().tolist())
            else:
                loss = mse_loss(out, tgt_raw)
                preds.extend(out.cpu().numpy().flatten().tolist())
                targs.extend(tgt_raw.cpu().numpy().flatten().tolist())
            smiles.extend(list(s))
            indices.extend(idx.numpy().tolist())
            total_loss += loss.item()
            nb += 1

    avg_mse = total_loss / max(1, nb)
    mae = mean_absolute_error(targs, preds)
    rmse = math.sqrt(mean_squared_error(targs, preds))
    r2 = r2_score(targs, preds)
    rel = (np.abs((np.array(targs) - np.array(preds)) / np.array(targs)) * 100).mean()

    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)
    print(f"Test Loss (MSE): {avg_mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Relative Error: {rel:.2f}%")

    return {
        'metrics': {'test_loss': avg_mse, 'mae': mae, 'rmse': rmse, 'r2': r2, 'mean_relative_error': rel},
        'predictions': preds, 'targets': targs, 'smiles': smiles, 'indices': indices
    }


# ------------------------------
# Model factory
# ------------------------------
def build_model(poly_dim: int, material_dim: int, device: torch.device):
    film_config = dict(
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
    model = create_film_model(poly_dim, material_dim, **film_config)
    return model.to(device)
# ------------------------------
# CLI
# ------------------------------
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Unified trainer for FiLM (and other) models')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--target_col', type=str, default='Conductivity')
    parser.add_argument('--features', type=str, default='WaterContent,SwellingRate,Degreeofpolymerization,ElongationatBreak,TensileStrength')
    parser.add_argument('--save_dir', type=str, default='')

    # training details (from CLI)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--loss', type=str, default='smoothl1', choices=['mse', 'smoothl1', 'l1'])
    parser.add_argument('--lr_scheduler', type=str, default='plateau', choices=['plateau', 'step'])
    parser.add_argument('--plateau_patience', type=int, default=10)
    parser.add_argument('--plateau_factor', type=float, default=0.5)
    parser.add_argument('--plateau_min_lr', type=float, default=1e-6)
    parser.add_argument('--lr_step_size', type=int, default=30)
    parser.add_argument('--lr_gamma', type=float, default=0.7)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--test_size', type=float, default=2, help='0 for no test set; <1 ratio; >=1 absolute count')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_normalize_features', action='store_true')
    parser.add_argument('--log_target', action='store_true')
    parser.add_argument('--attr_gradxinput', action='store_true')

    # model select
    # 只保留film模型，无需选择

    # embedding cache
    parser.add_argument('--cache_path', type=str, default='')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print('\nLoading polyBERT...')
    tokenizer = AutoTokenizer.from_pretrained('kuelumbus/polyBERT')
    polyBERT = AutoModel.from_pretrained('kuelumbus/polyBERT').to(device)
    polyBERT.eval()

    material_features = [f.strip() for f in args.features.split(',') if f.strip()]
    print('=' * 60)
    print('Unified Training (FiLM/Hybrid)')
    print('=' * 60)
    print(f'Dataset       : {args.data_path}')
    print(f'Target        : {args.target_col}')
    print(f'Features      : {material_features}')
    print(f'Model Type    : film')
    print(f'Epochs        : {args.epochs}')
    print(f'Batch Size    : {args.batch_size}')
    print(f'LR/Decay      : {args.lr} / {args.weight_decay}')
    print(f'LR Scheduler  : {args.lr_scheduler}')
    print(f'Loss/Optim    : {args.loss} / {args.optimizer}')
    print(f'Grad Clip     : {args.clip_grad_norm}')
    print(f'Log Target    : {args.log_target}')
    print(f'Test Size Arg : {args.test_size}')
    print(f'Normalize     : {not args.no_normalize_features}')
    print(f'Seed          : {args.seed}')
    print('-' * 60)

    print('\nProbing polyBERT dim...')
    poly_dim = get_polybert_dim(tokenizer, polyBERT, device)
    print(f'polyBERT dim  : {poly_dim}')

    print('\nLoading dataset...')
    ds = SMILESDataset(
        dataset_path=args.data_path,
        target_name=args.target_col,
        tokenizer=tokenizer,
        polyBERT=polyBERT,
        device=device,
        material_features=material_features,
        scaler=None,
        cache_path=(args.cache_path if args.cache_path else None),
    )
    total = len(ds)
    mat_dim = len(ds.feature_names)
    if total < 3:
        raise ValueError('Too few samples (<3)')

    # 支持 test_size=0 表示不划分测试集
    if args.test_size == 0:
        test_size = 0
        train_size = total
    elif args.test_size < 1:
        test_size = max(1, int(total * args.test_size))
        train_size = total - test_size
    else:
        test_size = int(args.test_size)
        if test_size >= total:
            test_size = total // 5 if total >= 5 else 1
            print(f'Auto adjust test_size={test_size}')
        train_size = total - test_size

    print(f'Split sizes   : Train={train_size}  Test={test_size}  (Total={total})')

    all_idx = torch.randperm(total).tolist()
    train_idx = all_idx[:train_size]
    test_idx = all_idx[train_size:]

    scaler = None
    if mat_dim > 0 and not args.no_normalize_features:
        scaler = fit_scaler_on_indices(ds, train_idx)
    ds.set_scaler(scaler)

    from torch.utils.data import Subset, DataLoader
    train_ds = Subset(ds, train_idx)
    test_ds = Subset(ds, test_idx) if test_size > 0 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False) if test_ds is not None else None

    print(f"\nCreate model: film")
    model = build_model(poly_dim, mat_dim, device)
    print(f"Total params  : {sum(p.numel() for p in model.parameters()):,}")

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'lr_step_size': args.lr_step_size,
        'lr_gamma': args.lr_gamma,
        'lr_scheduler': args.lr_scheduler,
        'plateau_patience': args.plateau_patience,
        'plateau_factor': args.plateau_factor,
        'plateau_min_lr': args.plateau_min_lr,
        'optimizer': args.optimizer,
        'loss': args.loss,
        'clip_grad_norm': args.clip_grad_norm,
        'log_target': args.log_target,
    }

    print('\nStart training...')
    model, history = train_model(model, train_loader, test_loader, config, device)

    # 若无测试集，跳过最终评估与相关可视化
    if test_loader is not None:
        print('\nFinal evaluation...')
        results = evaluate_model_final(model, test_loader, config, device)

        save_dir = args.save_dir.strip()
        if not save_dir:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = os.path.join('trained_models', f"film_{ts}")
        os.makedirs(save_dir, exist_ok=True)

        print('\nPlotting...')
        plot_training_history(history, os.path.join(save_dir, 'training'))
        plot_predictions(results, os.path.join(save_dir, 'predictions'), args.target_col)

        print('\nRunning gradient attribution...')
        sample_w2, sample_w5, idx_all = attribute_on_loader(model, test_loader, device, use_grad_x_input=args.attr_gradxinput)
        save_attribution_results(save_dir, sample_w2=sample_w2, sample_w5=sample_w5, idx_all=idx_all,
                                 feature_names=ds.feature_names if sample_w5 is not None else None)
        plot_combined_heatmap(save_dir)

        model_info = {
            'model_type': 'film',
            'poly_dim': poly_dim,
            'material_dim': mat_dim,
            'material_features': ds.feature_names,
            'target_name': args.target_col,
            'dataset_path': args.data_path,
            'scaler': scaler,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'dataset_split': {'total_samples': total, 'train_size': train_size, 'test_size': test_size},
        }
        save_model_and_results(model, config, history, results, model_info, save_dir)

        print('\nSummary:')
        print(f"Best Train Loss : {min(history['train_loss']):.4f}")
        print(f"Best Test Loss  : {min(history['test_loss']):.4f}")
        print(f"Final Test MSE  : {results['metrics']['test_loss']:.4f}")
        print(f"MAE / RMSE      : {results['metrics']['mae']:.4f} / {results['metrics']['rmse']:.4f}")
        print(f"R2              : {results['metrics']['r2']:.4f}")
        print(f"Saved to        : {save_dir}")
    else:
        # 仍保存模型与训练曲线（无测试集）
        save_dir = args.save_dir.strip()
        if not save_dir:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = os.path.join('trained_models', f"film_{ts}")
        os.makedirs(save_dir, exist_ok=True)

        plot_training_history(history, os.path.join(save_dir, 'training'))

        empty_results = {
            'metrics': {},
            'predictions': [],
            'targets': [],
            'smiles': [],
            'indices': []
        }
        model_info = {
            'model_type': 'film',
            'poly_dim': poly_dim,
            'material_dim': mat_dim,
            'material_features': ds.feature_names,
            'target_name': args.target_col,
            'dataset_path': args.data_path,
            'scaler': scaler,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'dataset_split': {'total_samples': total, 'train_size': train_size, 'test_size': test_size},
        }
        save_model_and_results(model, config, history, empty_results, model_info, save_dir)

        print('\nSummary:')
        print(f"Best Train Loss : {min(history['train_loss']):.4f}")
        print("No test set: skipped evaluation/attribution.")


if __name__ == '__main__':
    main()

