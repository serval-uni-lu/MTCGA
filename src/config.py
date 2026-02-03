import yaml
import torch
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Config:
    dataset_name: str = 'ethfraud30k'
    num_classes: int = 2
    lr: float = 5e-3
    epochs: int = 250
    dropout: float = 0.2
    weight_decay: float = 5e-4
    early_stopping_patience: int = 30
    device: str = "auto"
    seed: int = 42
    
    layers: int = 2
    hidden_units: int = 256
    use_batch_norm: bool = False
    
    val_size: float = 0.1
    test_size: float = 0.2
    
    gat_heads: int = 4
    gat_dropout: float = 0.2
    gatv2_heads: int = 4
    gatv2_dropout: float = 0.2
    cheb_k: list[int] = field(default_factory=lambda: [2, 3])

    scheduler_patience: int = 10
    scheduler_factor: float = 0.7

    p_evasion_threshold: float = 0.5
    cost_penalty_weight: float = 0.01
    max_budget_prop: float = 0.4
    max_sybils: int = 5
    num_optim_steps: int = 100
    max_transformations: int = 10
    maxiter: int = 100
    vol_tol: float = 1e-10
    len_tol: float = 1e-6
    
    def __post_init__(self) -> None:
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive")
        if self.hidden_units <= 0:
            raise ValueError("Hidden units must be positive")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("Dropout must be between 0 and 1")
        if self.layers < 1:
            raise ValueError("Number of layers must be at least 1")
        if self.gat_heads < 1:
            raise ValueError("Number of GAT heads must be at least 1")
        if self.val_size < 0 or self.val_size > 1:
            raise ValueError("Validation size must be between 0 and 1")
        if self.test_size < 0 or self.test_size > 1:
            raise ValueError("Test size must be between 0 and 1")
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        config = cls()
        
        if Path(path).exists():
            with open(path, 'r') as f:
                yaml_data = yaml.safe_load(f) or {}
            
            for key, value in yaml_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    def get_device(self) -> torch.device:
        if self.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device)
        
        return device