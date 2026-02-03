import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .dataloader import DataPreprocessor, load_txs, get_balance_for_id
from .config import Config
from .train import GNNTrainer
from .models import load_model, compute_pna_deg, GAT, GATv2, GraphTransformer
from .predict import predict_for_ids
from .attack import AttackConfig, MultiTargetConstrainedGraphAttack

AVAILABLE_MODELS = ['GCN', 'GAT', 'GATv2', 'SAGE', 'Chebyshev', 'GraphTransformer', 'PNA']


def load_attack_results(results_dir: Path, model_name: str) -> List[Dict]:
    """Load attack results JSON for a model."""
    attack_file = results_dir / f"{model_name}_attack.json"
    if not attack_file.exists():
        raise FileNotFoundError(f"Attack results not found: {attack_file}")

    with open(attack_file, 'r') as f:
        data = json.load(f)

    return data.get('attacks', [])


def get_successful_attacks(attacks: List[Dict]) -> List[Dict]:
    """Filter to only successful attacks with transactions."""
    return [a for a in attacks if a.get('success', False) and a.get('transactions')]

def remap_sybil_ids(attacks: List[Dict], base_graph_size: int) -> Tuple[List[Dict], int]:
    """
    Remap sybil IDs to unique global IDs across all attacks.

    Args:
        attacks: List of successful attack results with transactions
        base_graph_size: Original graph size (sybils start from this index)

    Returns:
        Tuple of (remapped_attacks, total_sybils_created)
    """
    next_sybil_id = base_graph_size
    remapped_attacks = []

    for attack in attacks:
        if not attack.get('success') or not attack.get('transactions'):
            continue

        # Find sybil IDs actually used in transactions (not just created)
        local_sybils_used = set()
        for tx in attack['transactions']:
            if tx['from_id'] >= base_graph_size:
                local_sybils_used.add(tx['from_id'])
            if tx['to_id'] >= base_graph_size:
                local_sybils_used.add(tx['to_id'])

        # Build local to global sybil mapping only for used sybils
        local_to_global = {}
        for local_sybil in sorted(local_sybils_used):
            local_to_global[local_sybil] = next_sybil_id
            next_sybil_id += 1

        # Remap transaction IDs
        remapped_txs = []
        for tx in attack['transactions']:
            new_tx = tx.copy()
            from_id = tx['from_id']
            to_id = tx['to_id']

            if from_id >= base_graph_size:
                new_tx['from_id'] = local_to_global[from_id]
            if to_id >= base_graph_size:
                new_tx['to_id'] = local_to_global[to_id]

            remapped_txs.append(new_tx)

        remapped_attack = attack.copy()
        remapped_attack['transactions'] = remapped_txs
        remapped_attack['sybil_mapping'] = local_to_global
        remapped_attacks.append(remapped_attack)

    total_sybils = next_sybil_id - base_graph_size
    return remapped_attacks, total_sybils


def attacks_to_txs_df(attacks: List[Dict]) -> pd.DataFrame:
    """
    Convert attack transactions to DataFrame compatible with dataloader.

    Args:
        attacks: List of attack results (should be remapped first)

    Returns:
        DataFrame with transaction columns matching original txs format
    """
    rows = []
    for attack in attacks:
        for tx in attack.get('transactions', []):
            rows.append({
                'from_id': tx['from_id'],
                'to_id': tx['to_id'],
                'value': tx['value'],
                'gas': tx['gas'],
                'gas_price': tx['gas_price'],
                'input': tx.get('input', '0x'),
                'receipt_gas_used': tx.get('receipt_gas_used', tx['gas'] * 0.95),
                'from_address': f"adv_{tx['from_id']}",
                'to_address': f"adv_{tx['to_id']}",
                'from_scam': True,
                'to_scam': True,
                'from_category': 'adversarial',
                'to_category': 'adversarial',
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_augmented_txs(
    original_txs: pd.DataFrame,
    attacks: List[Dict],
    base_graph_size: int
) -> Tuple[pd.DataFrame, int]:
    """
    Build augmented transaction DataFrame from original + adversarial transactions.

    Args:
        original_txs: Original transaction DataFrame
        attacks: List of successful attack results
        base_graph_size: Original graph size for sybil remapping

    Returns:
        Tuple of (augmented_txs DataFrame, num_sybils_created)
    """
    remapped_attacks, num_sybils = remap_sybil_ids(attacks, base_graph_size)
    adv_txs = attacks_to_txs_df(remapped_attacks)

    if adv_txs.empty:
        return original_txs.copy(), 0

    augmented_txs = pd.concat([original_txs, adv_txs], ignore_index=True)
    return augmented_txs, num_sybils


def build_augmented_data(
    original_txs: pd.DataFrame,
    attacks: List[Dict],
    base_graph_size: int
) -> DataPreprocessor:
    """
    Build augmented DataPreprocessor from original + adversarial transactions.

    Args:
        original_txs: Original transaction DataFrame
        attacks: List of successful attack results
        base_graph_size: Original graph size for sybil remapping

    Returns:
        New DataPreprocessor with augmented graph
    """
    augmented_txs, num_sybils = build_augmented_txs(
        original_txs, attacks, base_graph_size
    )
    return DataPreprocessor(augmented_txs)


def split_test_indices(
    test_indices: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split test indices into two stratified halves.

    Args:
        test_indices: Array of test node indices
        labels: Array of node labels for stratification
        seed: Random seed for reproducibility

    Returns:
        Tuple of (test_aug_indices, test_final_indices)
    """
    from sklearn.model_selection import train_test_split

    test_labels = labels[test_indices]

    test_aug, test_final = train_test_split(
        test_indices,
        test_size=0.5,
        stratify=test_labels
    )
    return test_aug, test_final


def filter_attacks_by_nodes(
    attacks: List[Dict],
    node_ids: np.ndarray
) -> List[Dict]:
    """
    Filter attacks to only include those for specified node IDs.

    Args:
        attacks: List of attack results
        node_ids: Array of node IDs to include

    Returns:
        Filtered list of attacks
    """
    node_set = set(node_ids)
    return [a for a in attacks if a.get('node_id') in node_set]


# ============================================================================
# Adversarial Training Workflow
# ============================================================================

def _load_splits(model_name: str) -> dict:
    """Load train/val/test splits from model directory."""
    model_dir = Path(f"models/{model_name}")
    return {
        'train': np.load(model_dir / "train.npy"),
        'val': np.load(model_dir / "val.npy"),
        'test': np.load(model_dir / "test.npy"),
    }


def _evaluate_on_test_final(model, data, test_final_idx, device):
    """Evaluate model performance on test_final set."""
    model.eval()
    data_gpu = data.graph.to(device)

    with torch.no_grad():
        if isinstance(model, (GAT, GATv2, GraphTransformer)):
            out = model(data_gpu.x, data_gpu.edge_index, data_gpu.edge_attr)
        else:
            out = model(data_gpu.x, data_gpu.edge_index)
        probs = torch.softmax(out, dim=1)[:, 1]

    test_probs = probs[test_final_idx].cpu().numpy()
    test_labels = data.graph.y[test_final_idx].cpu().numpy()
    test_preds = (test_probs >= 0.5).astype(int)

    return {
        'accuracy': accuracy_score(test_labels, test_preds),
        'f1': f1_score(test_labels, test_preds),
        'precision': precision_score(test_labels, test_preds, zero_division=0),
        'recall': recall_score(test_labels, test_preds, zero_division=0),
        'auc': roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.0,
    }


def _attack_test_final(model, model_name, data, txs, test_final_idx, attack_config, results_dir, suffix=''):
    """Attack nodes in test_final set."""
    probs = predict_for_ids(model, data.graph, test_final_idx)
    scam_mask = probs >= 0.5
    scam_indices = test_final_idx[scam_mask]

    attack_results = []
    for node_id in scam_indices:
        balance = get_balance_for_id(txs, int(node_id))
        if balance <= 0:
            continue

        try:
            attacker = MultiTargetConstrainedGraphAttack(
                evading_ids=int(node_id),
                model=model,
                datapreprocessor=data,
                config=attack_config
            )
            result = attacker.run()

            attack_results.append({
                'node_id': int(node_id),
                'success': bool(result.success),
                'initial_prob': float(result.initial_prob),
                'final_prob': float(result.final_prob),
                'steps_taken': int(result.steps_taken),
                'sybils_created': int(result.sybils_created),
                'budget_spent_prop': float(result.budget_spent_prop),
                'num_transactions': len(result.transactions),
            })
        except Exception as e:
            attack_results.append({
                'node_id': int(node_id),
                'success': False,
                'error': str(e),
            })

    save_name = f"{model_name}{suffix}_test_final.json"
    with open(results_dir / save_name, 'w') as f:
        json.dump({'model': f"{model_name}{suffix}", 'attacks': attack_results}, f, indent=2)

    return attack_results


def run_adversarial_training(
    models: Optional[List[str]] = None,
    dataset: str = 'ethfraud',
    config_path: Optional[str] = 'config.yaml',
    attack_results_dir: str = 'results/full_test_attacks'
) -> pd.DataFrame:
    """
    Run adversarial training experiment.

    Args:
        models: List of models to train (None for all)
        dataset: Dataset name
        config_path: Path to config file
        attack_results_dir: Directory containing attack results

    Returns:
        DataFrame with comparison results
    """
    config = Config.from_yaml(config_path) if config_path else Config()
    torch.manual_seed(config.seed)

    print("Loading dataset...")
    txs = load_txs(dataset)
    data = DataPreprocessor(txs)
    base_graph_size = data.graph.x.shape[0]

    attack_config = AttackConfig()
    trainer = GNNTrainer(config)

    results_dir = Path('results/adv_training')
    results_dir.mkdir(parents=True, exist_ok=True)

    attack_dir = Path(attack_results_dir)
    if not attack_dir.exists():
        raise FileNotFoundError(f"Attack results not found at {attack_results_dir}. Run attack command first.")

    if models is None or 'all' in models:
        models = AVAILABLE_MODELS

    all_results = []

    for model_name in models:
        if model_name not in AVAILABLE_MODELS:
            print(f"Model {model_name} not available. Available: {AVAILABLE_MODELS}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {model_name}...")
        print('='*60)

        model_dir = Path(f"models/{model_name}")
        if not model_dir.exists():
            print(f"  Model not found, skipping")
            continue

        # Load original splits
        splits = _load_splits(model_name)
        train_idx = splits['train']
        val_idx = splits['val']
        test_idx = splits['test']

        # Split test into test_aug and test_final
        labels = data.graph.y.cpu().numpy()
        test_aug_idx, test_final_idx = split_test_indices(test_idx, labels)

        print(f"  Original splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        print(f"  Test split: test_aug={len(test_aug_idx)}, test_final={len(test_final_idx)}")

        # Extended training set
        extended_train_idx = np.concatenate([train_idx, test_aug_idx])
        print(f"  Extended train: {len(extended_train_idx)} nodes")

        # Load and filter attack results
        attacks = load_attack_results(attack_dir, model_name)
        test_aug_attacks = filter_attacks_by_nodes(attacks, test_aug_idx)
        successful_attacks = get_successful_attacks(test_aug_attacks)

        print(f"  Attacks on test_aug: {len(test_aug_attacks)} total, {len(successful_attacks)} successful")

        # Build augmented data
        augmented_txs, num_sybils = build_augmented_txs(txs, successful_attacks, base_graph_size)
        print(f"  Augmented graph: {len(augmented_txs)} transactions, {num_sybils} new sybils")

        augmented_data = DataPreprocessor(augmented_txs)

        # Train clean model
        print(f"\n  Training clean model {model_name}_clean...")
        trainer.train_model_with_splits(
            model_name, data.graph,
            extended_train_idx, val_idx, test_final_idx,
            model_dir_suffix='_clean'
        )

        # Train adversarial model
        print(f"\n  Training adversarial model {model_name}_adv...")
        trainer.train_model_with_splits(
            model_name, augmented_data.graph,
            extended_train_idx, val_idx, test_final_idx,
            model_dir_suffix='_adv'
        )

        # Load trained models
        edge_dim = data.graph.edge_attr.shape[1] if data.graph.edge_attr is not None else 0
        pna_deg = compute_pna_deg(data.graph.edge_index, data.graph.x.shape[0]) if model_name == 'PNA' else None

        clean_model = load_model(model_name, data.graph.x.shape[1], edge_dim, config, pna_deg=pna_deg)
        clean_model.load_state_dict(torch.load(f"models/{model_name}_clean/model.pth", weights_only=True))
        clean_model = clean_model.to(config.get_device())
        clean_model.eval()

        aug_edge_dim = augmented_data.graph.edge_attr.shape[1] if augmented_data.graph.edge_attr is not None else 0
        aug_pna_deg = compute_pna_deg(augmented_data.graph.edge_index, augmented_data.graph.x.shape[0]) if model_name == 'PNA' else None

        adv_model = load_model(model_name, augmented_data.graph.x.shape[1], aug_edge_dim, config, pna_deg=aug_pna_deg)
        adv_model.load_state_dict(torch.load(f"models/{model_name}_adv/model.pth", weights_only=True))
        adv_model = adv_model.to(config.get_device())
        adv_model.eval()

        # Evaluate on test_final
        print(f"\n  Evaluating on test_final ({len(test_final_idx)} nodes)...")
        clean_metrics = _evaluate_on_test_final(clean_model, data, test_final_idx, config.get_device())
        adv_metrics = _evaluate_on_test_final(adv_model, data, test_final_idx, config.get_device())

        print(f"    Clean model - F1: {clean_metrics['f1']:.4f}, AUC: {clean_metrics['auc']:.4f}")
        print(f"    Adv model   - F1: {adv_metrics['f1']:.4f}, AUC: {adv_metrics['auc']:.4f}")

        # Attack test_final
        print(f"\n  Attacking test_final with clean model...")
        clean_attacks = _attack_test_final(clean_model, model_name, data, txs, test_final_idx, attack_config, results_dir, '_clean')
        clean_success = [a for a in clean_attacks if a.get('success', False)]
        print(f"    {len(clean_success)}/{len(clean_attacks)} attacks successful")

        print(f"\n  Attacking test_final with adversarial model...")
        adv_attacks = _attack_test_final(adv_model, model_name, data, txs, test_final_idx, attack_config, results_dir, '_adv')
        adv_success = [a for a in adv_attacks if a.get('success', False)]
        print(f"    {len(adv_success)}/{len(adv_attacks)} attacks successful")

        # Compute robustness improvement
        clean_attack_rate = len(clean_success) / len(clean_attacks) if clean_attacks else 0
        adv_attack_rate = len(adv_success) / len(adv_attacks) if adv_attacks else 0
        robustness_improvement = (clean_attack_rate - adv_attack_rate) / clean_attack_rate if clean_attack_rate > 0 else 0

        result = {
            'model': model_name,
            'clean_f1': clean_metrics['f1'],
            'clean_auc': clean_metrics['auc'],
            'clean_attack_success_rate': clean_attack_rate,
            'adv_f1': adv_metrics['f1'],
            'adv_auc': adv_metrics['auc'],
            'adv_attack_success_rate': adv_attack_rate,
            'robustness_improvement': robustness_improvement,
            'num_augmented_attacks': len(successful_attacks),
            'num_sybils_added': num_sybils,
        }
        all_results.append(result)

        print(f"\n  Summary for {model_name}:")
        print(f"    Clean:  F1={clean_metrics['f1']:.3f}, Attack success={clean_attack_rate:.1%}")
        print(f"    Adv:    F1={adv_metrics['f1']:.3f}, Attack success={adv_attack_rate:.1%}")
        print(f"    Robustness improvement: {robustness_improvement:.1%}")

    # Save report
    df = pd.DataFrame(all_results)
    df.to_csv(results_dir / 'comparison_report.csv', index=False)

    print(f"\n{'='*60}")
    print("Adversarial Training Comparison Report")
    print('='*60)
    if not df.empty:
        print(df[['model', 'clean_f1', 'adv_f1', 'clean_attack_success_rate',
                  'adv_attack_success_rate', 'robustness_improvement']].to_string(index=False))
    print(f"\nFull report saved to {results_dir / 'comparison_report.csv'}")

    return df
