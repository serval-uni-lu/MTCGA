import numpy as np
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from .config import Config
from .models import load_model, compute_pna_deg


def predict_for_ids(model, graph, ids: list[int]) -> np.ndarray:
    """Get predictions for specific node IDs."""
    model.eval()
    graph = graph.to(next(model.parameters()).device)
    
    with torch.no_grad():
        x, edge_index = graph.x, graph.edge_index
        
        if hasattr(model, 'convs') and getattr(model.convs[0], 'edge_dim', None) is not None and graph.edge_attr is not None:
            logits = model(x, edge_index, graph.edge_attr)
        else:
            logits = model(x, edge_index)
        
        probs = F.softmax(logits[ids], dim=-1)[:, 1]
        return probs.cpu().numpy()


def evaluate_model(model_name: str, data: Data, config: Config, split: str = 'test') -> Dict[str, float]:
    split_path = Path(f"models/{model_name}/{split}.npy")
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    indices = np.load(split_path)
    mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    mask[indices] = True

    edge_dim = data.edge_attr.shape[1] if data.edge_attr is not None else 0

    # Compute pna_deg if needed
    pna_deg = compute_pna_deg(data.edge_index, data.x.shape[0]) if model_name == 'PNA' else None

    model = load_model(model_name, data.x.shape[1], edge_dim, config, pna_deg=pna_deg)
    
    device = config.get_device()
    model = model.to(device)
    data = data.to(device)
    mask = mask.to(device)
    
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'convs') and getattr(model.convs[0], 'edge_dim', None) is not None and data.edge_attr is not None:
            out = model(data.x, data.edge_index, data.edge_attr)
        else:
            out = model(data.x, data.edge_index)
        
        loss = F.nll_loss(out[mask], data.y[mask])
        pred = out[mask].max(1)[1]
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_prob = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
        
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
            'confusion_matrix': cm.tolist(),
            'true_positives': int(cm[1, 1]) if cm.shape[0] > 1 else 0,
            'false_positives': int(cm[0, 1]) if cm.shape[0] > 1 else 0,
            'true_negatives': int(cm[0, 0]),
            'false_negatives': int(cm[1, 0]) if cm.shape[0] > 1 else 0
        }
        
        return metrics


def evaluate_all_models(data: Data, config: Config, split: str = 'test') -> Dict[str, Dict[str, float]]:
    models_dir = Path("models")
    if not models_dir.exists():
        raise FileNotFoundError("Models directory not found")
    
    results = {}
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir() and (model_dir / "model.pth").exists():
            model_name = model_dir.name
            try:
                metrics = evaluate_model(model_name, data, config, split)
                results[model_name] = metrics
                print(f"{model_name} - {split} F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
    
    return results