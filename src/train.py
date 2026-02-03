import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .config import Config
from .models import get_gnn_constructors, GAT, GATv2, GraphTransformer, compute_pna_deg


class GNNTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.get_device()

    def train_epoch(self, model: Module, data: Data, optimizer: torch.optim.Optimizer,
                   train_mask: torch.Tensor) -> float:
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr) if isinstance(model, (GAT, GATv2, GraphTransformer)) else model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def evaluate(self, model: Module, data: Data, mask: torch.Tensor) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr) if isinstance(model, (GAT, GATv2, GraphTransformer)) else model(data.x, data.edge_index)
            loss = F.nll_loss(out[mask], data.y[mask])
            pred = out[mask].max(1)[1]
            y_true, y_pred = data.y[mask].cpu().numpy(), pred.cpu().numpy()
            y_prob = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
            
            return {
                'loss': loss.item(),
                'accuracy': accuracy_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
            }
    
    def create_masks(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_nodes = data.x.shape[0]
        indices = np.arange(num_nodes)
        
        train_idx, test_idx = train_test_split(
            indices, test_size=self.config.test_size, 
            stratify=data.y.cpu(), random_state=self.config.seed
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=self.config.val_size/(1-self.config.test_size), 
            stratify=data.y[train_idx].cpu(), random_state=self.config.seed
        )
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        return train_mask.to(self.device), val_mask.to(self.device), test_mask.to(self.device), train_idx, val_idx, test_idx
    
    def train_model(self, model_name: str, data: Data) -> Dict[str, Any]:
        edge_dim = data.edge_attr.shape[1] if data.edge_attr is not None else 0

        # Compute PNA deg if needed
        pna_deg = compute_pna_deg(data.edge_index, data.x.shape[0]) if model_name == 'PNA' else None

        constructors = get_gnn_constructors(data.x.shape[1], edge_dim, self.config, pna_deg=pna_deg)

        if model_name not in constructors:
            raise ValueError(f"Model {model_name} not available")
        
        model = constructors[model_name]()
        model, data = model.to(self.device), data.to(self.device)

        train_mask, val_mask, test_mask, train_idx, val_idx, test_idx = self.create_masks(data)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        best_val_f1, best_model_state, patience_counter = -1, model.state_dict().copy(), 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(model, data, optimizer, train_mask)
            val_metrics = self.evaluate(model, data, val_mask)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['f1'])
            
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                break
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        model.load_state_dict(best_model_state)
        test_metrics = self.evaluate(model, data, test_mask)
        
        model_dir = Path(f"models/{model_name}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(model.state_dict(), model_dir / "model.pth")
        np.save(model_dir / "train.npy", train_idx)
        np.save(model_dir / "val.npy", val_idx)
        np.save(model_dir / "test.npy", test_idx)
        
        print(f"{model_name} - Test F1: {test_metrics['f1']:.4f}, Test AUC: {test_metrics['auc']:.4f}")

        return {
            'model': model,
            'history': history,
            'test_metrics': test_metrics,
            'splits': {'train': train_idx, 'val': val_idx, 'test': test_idx}
        }

    def train_model_with_splits(
        self,
        model_name: str,
        data: Data,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        model_dir_suffix: str = ''
    ) -> Dict[str, Any]:
        """
        Train a model with pre-defined splits.

        Args:
            model_name: Name of the model architecture
            data: PyG Data object
            train_idx: Training node indices
            val_idx: Validation node indices
            test_idx: Test node indices
            model_dir_suffix: Suffix for model directory (e.g., '_adv')

        Returns:
            Dictionary with model, history, and metrics
        """
        edge_dim = data.edge_attr.shape[1] if data.edge_attr is not None else 0

        # Compute PNA deg if needed
        pna_deg = compute_pna_deg(data.edge_index, data.x.shape[0]) if model_name == 'PNA' else None

        constructors = get_gnn_constructors(data.x.shape[1], edge_dim, self.config, pna_deg=pna_deg)

        if model_name not in constructors:
            raise ValueError(f"Model {model_name} not available")

        model = constructors[model_name]()
        model, data = model.to(self.device), data.to(self.device)

        # Create masks from provided indices
        num_nodes = data.x.shape[0]
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        test_mask = test_mask.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        best_val_f1, best_model_state, patience_counter = -1, model.state_dict().copy(), 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(model, data, optimizer, train_mask)
            val_metrics = self.evaluate(model, data, val_mask)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['f1'])

            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                break

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val F1: {val_metrics['f1']:.4f}")

        model.load_state_dict(best_model_state)
        test_metrics = self.evaluate(model, data, test_mask)

        # Save model with optional suffix
        save_name = f"{model_name}{model_dir_suffix}"
        model_dir = Path(f"models/{save_name}")
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), model_dir / "model.pth")
        np.save(model_dir / "train.npy", train_idx)
        np.save(model_dir / "val.npy", val_idx)
        np.save(model_dir / "test.npy", test_idx)

        print(f"{save_name} - Test F1: {test_metrics['f1']:.4f}, Test AUC: {test_metrics['auc']:.4f}")

        return {
            'model': model,
            'history': history,
            'test_metrics': test_metrics,
            'splits': {'train': train_idx, 'val': val_idx, 'test': test_idx}
        }