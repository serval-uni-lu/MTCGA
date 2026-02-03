import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .config import Config
from .dataloader import load_txs, DataPreprocessor
from .models import load_model, compute_pna_deg, GAT, GATv2, GraphTransformer
from .adversarial import (
    load_attack_results,
    get_successful_attacks,
    build_augmented_txs,
    AVAILABLE_MODELS,
)

EVASION_THRESHOLD = 0.5


def _get_predictions(model, graph, node_ids: List[int], device) -> np.ndarray:
    """Get fraud probabilities for specified node IDs."""
    model.eval()
    graph_gpu = graph.to(device)

    with torch.no_grad():
        x, edge_index = graph_gpu.x, graph_gpu.edge_index
        edge_attr = graph_gpu.edge_attr if hasattr(graph_gpu, 'edge_attr') else None

        if isinstance(model, (GAT, GATv2, GraphTransformer)) and edge_attr is not None:
            logits = model(x, edge_index, edge_attr)
        else:
            logits = model(x, edge_index)

        probs = F.softmax(logits, dim=1)[:, 1]
        valid_ids = [nid for nid in node_ids if nid < probs.shape[0]]

        return probs[valid_ids].cpu().numpy()


def _load_all_models(data: DataPreprocessor, config: Config, device) -> Dict[str, torch.nn.Module]:
    """Load all trained models."""
    models = {}
    edge_dim = data.graph.edge_attr.shape[1] if data.graph.edge_attr is not None else 0

    for model_name in AVAILABLE_MODELS:
        model_dir = Path(f"models/{model_name}")
        if not model_dir.exists():
            print(f"  Model {model_name} not found, skipping")
            continue

        try:
            pna_deg = compute_pna_deg(data.graph.edge_index, data.graph.x.shape[0]) if model_name == 'PNA' else None
            model = load_model(model_name, data.graph.x.shape[1], edge_dim, config, pna_deg=pna_deg)
            model = model.to(device)
            model.eval()
            models[model_name] = model
        except Exception as e:
            print(f"  Error loading {model_name}: {e}")

    return models


def _compute_baseline_evasion(
    models: Dict[str, torch.nn.Module],
    data: DataPreprocessor,
    attack_results: Dict[str, List[Dict]],
    device
) -> Dict[str, Dict[str, float]]:
    """Compute baseline evasion rates on clean graph."""
    baseline = {}

    for source_model in attack_results:
        if source_model not in models:
            continue

        attacks = attack_results[source_model]
        successful = get_successful_attacks(attacks)
        if not successful:
            continue

        attacked_nodes = [a['node_id'] for a in successful]
        baseline[source_model] = {}

        for target_model, model in models.items():
            probs = _get_predictions(model, data.graph, attacked_nodes, device)
            evasion_rate = (probs < EVASION_THRESHOLD).mean()
            baseline[source_model][target_model] = float(evasion_rate)

    return baseline


def _compute_transferability_matrix(
    models: Dict[str, torch.nn.Module],
    original_txs: pd.DataFrame,
    base_graph_size: int,
    attack_results: Dict[str, List[Dict]],
    config: Config,
    device
) -> Tuple[pd.DataFrame, Dict]:
    """Compute attack transferability matrix."""
    transfer_data = {src: {} for src in AVAILABLE_MODELS}
    detailed_results = {}

    for source_model in AVAILABLE_MODELS:
        if source_model not in attack_results:
            print(f"\n  No attack results for {source_model}, skipping")
            continue

        attacks = attack_results[source_model]
        successful = get_successful_attacks(attacks)

        if not successful:
            print(f"\n  No successful attacks for {source_model}, skipping")
            continue

        print(f"\n  Source: {source_model} ({len(successful)} successful attacks)")
        attacked_nodes = [a['node_id'] for a in successful]

        augmented_txs, num_sybils = build_augmented_txs(
            original_txs, successful, base_graph_size
        )
        print(f"    Augmented graph: {len(augmented_txs)} txs, {num_sybils} sybils")

        augmented_data = DataPreprocessor(augmented_txs)
        edge_dim = augmented_data.graph.edge_attr.shape[1] if augmented_data.graph.edge_attr is not None else 0

        detailed_results[source_model] = {
            'attacked_nodes': attacked_nodes,
            'num_sybils': num_sybils,
            'target_results': {}
        }

        for target_model in AVAILABLE_MODELS:
            if target_model not in models:
                continue

            try:
                pna_deg = compute_pna_deg(
                    augmented_data.graph.edge_index,
                    augmented_data.graph.x.shape[0]
                ) if target_model == 'PNA' else None

                target_model_instance = load_model(
                    target_model,
                    augmented_data.graph.x.shape[1],
                    edge_dim,
                    config,
                    pna_deg=pna_deg
                )
                target_model_instance = target_model_instance.to(device)
                target_model_instance.eval()

                probs = _get_predictions(
                    target_model_instance,
                    augmented_data.graph,
                    attacked_nodes,
                    device
                )

                evasion_rate = (probs < EVASION_THRESHOLD).mean()
                transfer_data[source_model][target_model] = float(evasion_rate)

                detailed_results[source_model]['target_results'][target_model] = {
                    'evasion_rate': float(evasion_rate),
                    'probs': probs.tolist(),
                    'evaded': (probs < EVASION_THRESHOLD).tolist()
                }

                marker = "*" if source_model == target_model else ""
                print(f"    -> {target_model}: {evasion_rate:.1%} evasion {marker}")

            except Exception as e:
                print(f"    -> {target_model}: Error - {e}")
                transfer_data[source_model][target_model] = np.nan

    transfer_matrix = pd.DataFrame(transfer_data).T
    transfer_matrix = transfer_matrix.reindex(index=AVAILABLE_MODELS, columns=AVAILABLE_MODELS)

    return transfer_matrix, detailed_results


def _compute_correlation_matrix(transfer_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation between attack patterns across models."""
    normalized = transfer_matrix.copy()
    for model in transfer_matrix.index:
        if model in transfer_matrix.columns:
            diag_val = transfer_matrix.loc[model, model]
            if pd.notna(diag_val) and diag_val > 0:
                normalized.loc[model] = normalized.loc[model] / diag_val
    return normalized


def run_transferability_analysis(
    models: Optional[List[str]] = None,
    dataset: str = 'ethfraud',
    config_path: Optional[str] = 'config.yaml',
    attack_results_dir: str = 'results/full_test_attacks'
) -> pd.DataFrame:
    """
    Analyze attack transferability across models.

    Args:
        models: List of models to analyze (None for all)
        dataset: Dataset name
        config_path: Path to config file
        attack_results_dir: Directory containing attack results

    Returns:
        Transfer matrix DataFrame
    """
    config = Config.from_yaml(config_path) if config_path else Config()
    device = config.get_device()
    torch.manual_seed(config.seed)

    print("=" * 60)
    print("Attack Transferability Analysis")
    print("=" * 60)

    print("\nLoading dataset...")
    txs = load_txs(dataset)
    data = DataPreprocessor(txs)
    base_graph_size = data.graph.x.shape[0]
    print(f"  Base graph: {base_graph_size} nodes, {data.graph.edge_index.shape[1]} edges")

    print("\nLoading models...")
    loaded_models = _load_all_models(data, config, device)
    print(f"  Loaded {len(loaded_models)} models: {list(loaded_models.keys())}")

    print("\nLoading attack results...")
    attack_dir = Path(attack_results_dir)
    attack_results = {}

    model_list = models if models and 'all' not in models else AVAILABLE_MODELS

    for model_name in model_list:
        try:
            attacks = load_attack_results(attack_dir, model_name)
            attack_results[model_name] = attacks
            successful = get_successful_attacks(attacks)
            print(f"  {model_name}: {len(attacks)} attacks, {len(successful)} successful")
        except FileNotFoundError:
            print(f"  {model_name}: No attack results found")

    # Compute baseline
    print("\n" + "=" * 60)
    print("Computing baseline evasion rates (clean graph)...")
    baseline = _compute_baseline_evasion(loaded_models, data, attack_results, device)

    # Compute transferability
    print("\n" + "=" * 60)
    print("Computing transferability matrix...")
    transfer_matrix, detailed_results = _compute_transferability_matrix(
        loaded_models, txs, base_graph_size, attack_results, config, device
    )

    # Compute matrices
    baseline_matrix = pd.DataFrame(baseline).T
    baseline_matrix = baseline_matrix.reindex(index=AVAILABLE_MODELS, columns=AVAILABLE_MODELS)
    improvement_matrix = transfer_matrix - baseline_matrix
    correlation_matrix = _compute_correlation_matrix(transfer_matrix)

    # Save results
    results_dir = Path('results/transferability')
    results_dir.mkdir(parents=True, exist_ok=True)

    transfer_matrix.to_csv(results_dir / 'transfer_matrix.csv')
    baseline_matrix.to_csv(results_dir / 'baseline_matrix.csv')
    improvement_matrix.to_csv(results_dir / 'improvement_matrix.csv')
    correlation_matrix.to_csv(results_dir / 'correlation_matrix.csv')

    with open(results_dir / 'detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)

    # Print results
    print("\n" + "=" * 60)
    print("TRANSFER MATRIX (Adversarial Graph)")
    print("=" * 60)
    print("Rows = Source model (attacks), Columns = Target model (detector)")
    print(transfer_matrix.round(3).to_string())

    print("\n" + "=" * 60)
    print("IMPROVEMENT OVER BASELINE")
    print("=" * 60)
    print(improvement_matrix.round(3).to_string())

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    diag = pd.Series({m: transfer_matrix.loc[m, m] for m in AVAILABLE_MODELS
                     if m in transfer_matrix.index and m in transfer_matrix.columns})
    print(f"\nSelf-attack success rates (diagonal):")
    for model, rate in diag.items():
        if pd.notna(rate):
            print(f"  {model}: {rate:.1%}")

    off_diag_rates = []
    for src in AVAILABLE_MODELS:
        for tgt in AVAILABLE_MODELS:
            if src != tgt and src in transfer_matrix.index and tgt in transfer_matrix.columns:
                val = transfer_matrix.loc[src, tgt]
                if pd.notna(val):
                    off_diag_rates.append(val)

    if off_diag_rates:
        print(f"\nOff-diagonal transfer statistics:")
        print(f"  Mean: {np.mean(off_diag_rates):.1%}")
        print(f"  Std:  {np.std(off_diag_rates):.1%}")

    print(f"\nResults saved to {results_dir}")

    return transfer_matrix
