import torch
import numpy as np
import pandas as pd
from scipy.optimize import direct
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional

from .account import Account
from .transaction import Transaction
from .dataloader import (
    DataPreprocessor,
    get_txs_for_ids,
    get_balance_for_id,
    to_account_features,
)


@dataclass
class AttackConfig:
    """Configuration for multi-target constrained graph attack."""

    # transaction constraints
    value_min: float = 1e14  # 0.0001 ETH
    value_max: float = 1e21
    gas: int = 21000
    gas_price_min: float = 1e9
    gas_price_max: float = 1e12

    # budget constraints
    max_budget_prop: float = 0.4  # max proportion of available funds (alpha)

    # structural constraints
    max_transformations: int = -1  # -1 means auto
    max_sybils: int = 5

    # optimization parameters
    num_optim_steps: int = 10
    maxiter: int = 10
    vol_tol: float = 1e-10
    len_tol: float = 1e-6

    # attack parameters
    p_evasion_threshold: float = 0.5

    # penalty coefficients (μ_g, μ_w in paper)
    gas_penalty_coef: float = 0.1
    value_penalty_coef: float = 0.01


@dataclass
class MTCGAResult:
    """Results from multi-target constrained graph attack."""

    success: bool = False
    steps_taken: int = 0
    sybils_created: int = 0
    transactions: List[Dict] = field(default_factory=list)
    probabilities: List[Dict] = field(default_factory=list)
    early_stopped: bool = False
    total_budget: float = 0.0
    budget_spent: float = 0.0
    budget_spent_prop: float = 0.0
    budget_exhausted: bool = False
    initial_prob: float = 0.0
    final_prob: float = 0.0


class MultiTargetConstrainedGraphAttack:
    """Multi-Target Constrained Graph Attack implementation."""

    def __init__(
        self,
        evading_ids: int | List[int],
        model: torch.nn.Module,
        datapreprocessor: DataPreprocessor,
        config: AttackConfig,
    ):
        """
        Initialize the attack.

        Args:
            evading_ids: Original fraudulent linked account IDs
            model: Target GNN model
            datapreprocessor: Data preprocessor object
            config: Attack configuration
        """
        self.controlled_ids = [evading_ids] if isinstance(evading_ids, int) else evading_ids
        self.model = model
        self.data = datapreprocessor
        self.config = config
        self.device = next(model.parameters()).device
        self.model.eval()

        # initialize state
        self.results = MTCGAResult()
        self.controlled_accounts = {}
        self.budget_spent = 0.0

    def run(self) -> MTCGAResult:
        """
        Execute the multi-target constrained graph attack.

        Returns:
            AttackResult with success status and metrics
        """
        # initialize attack components
        self._initialize_attack()

        # main attack loop
        for step in range(self.config.max_transformations):
            if self._execute_attack_step(step):
                break

        # final evaluation
        self._finalize_results()

        return self.results

    def _initialize_attack(self):
        """Initialize graph components and budgets."""
        # set up transaction history and graph
        self.txs = get_txs_for_ids(self.data.txs, self.controlled_ids)
        self.graph = self.data.graph.clone().to(self.device)
        
        # get initial probabilities
        _, initial_probs = self._compute_gradients()
        self.results.initial_prob = float(initial_probs.max())

        # initialize controlled accounts
        for id in self.controlled_ids:
            balance = get_balance_for_id(self.txs, id)
            self.controlled_accounts[id] = Account(id, balance=balance)

        # set up budget
        self.total_budget = self._calculate_total_balance()
        self.available_budget = self.total_budget * self.config.max_budget_prop
        self.results.total_budget = self.total_budget

        # create initial sybil if needed
        if len(self.controlled_ids) == 1:
            self._create_sybil()

        # initialize features
        self.node_features = self._get_node_features(self.txs, self.controlled_ids)

        # auto-configure max transformations
        if self.config.max_transformations < 0:
            self.config.max_transformations = max(10, len(self.txs) // 4)

    def _execute_attack_step(self, step: int) -> bool:
        """
        Execute a single attack step.

        Args:
            step: Current step number

        Returns:
            True if attack should terminate early
        """
        # compute gradients and probabilities
        gradients, probs = self._compute_gradients()

        # record state
        self._record_probability_state(step, probs)

        # check success condition
        max_prob = float(probs.max())
        if max_prob < self.config.p_evasion_threshold:
            self._mark_success(step, max_prob)
            return True

        # find and apply transformation
        self._apply_transformation(step, gradients, probs)

        self.results.steps_taken = step + 1
        return False

    def _apply_transformation(self, step: int, gradients: torch.Tensor, probs: torch.Tensor):
        """Apply optimal transformation or create sybil."""
        # identify highest risk node
        high_risk_idx = probs.argmax()
        high_risk_id = self.controlled_ids[high_risk_idx]

        # find optimal transaction
        optimal_tx = self._find_optimal_tx(high_risk_id, gradients, probs, step)

        if optimal_tx is None:
            # create sybil if no beneficial transformation found
            if self.results.sybils_created < self.config.max_sybils:
                self._create_sybil()
        else:
            # apply the transformation
            self._apply_tx(optimal_tx)
            self.results.transactions.append(optimal_tx.to_dict())

    def _finalize_results(self):
        """Compute final metrics and update results."""
        _, final_probs = self._compute_gradients()

        self.results.budget_spent = self.budget_spent
        self.results.budget_spent_prop = self.budget_spent / self.total_budget
        self.results.budget_exhausted = self.budget_spent >= self.available_budget * 0.95

        max_final_prob = float(final_probs.max())
        self.results.final_prob = max_final_prob
        self.results.success = max_final_prob < self.config.p_evasion_threshold

    def _mark_success(self, step: int, max_prob: float):
        """Mark attack as successful."""
        self.results.success = True
        self.results.early_stopped = True
        self.results.steps_taken = step
        self.results.budget_spent = self.budget_spent
        self.results.budget_spent_prop = self.budget_spent / self.total_budget
        self.results.final_prob = max_prob

    def _record_probability_state(self, step: int, probs: torch.Tensor):
        """Record probability state for analysis."""
        self.results.probabilities.append({
            'step': step + 1,
            'probs': {int(id): float(p) for id, p in zip(self.controlled_ids, probs)},
        })

    def _create_sybil(self) -> int:
        """
        Create a new sybil account.

        Returns:
            ID of the newly created sybil account
        """
        sybil_id = self.graph.add_node()
        self.controlled_ids.append(sybil_id)
        self.controlled_accounts[sybil_id] = Account(sybil_id)
        self.node_features = self._get_node_features(self.txs, self.controlled_ids)
        self.results.sybils_created += 1
        return sybil_id

    def _get_node_features(self, txs: pd.DataFrame, ids: List[int]) -> torch.Tensor:
        """
        Extract node features for specified accounts.

        Note: Caching disabled due to mutable txs DataFrame.
        Consider using immutable keys if caching is critical.

        Args:
            txs: Transaction dataframe
            ids: List of node IDs

        Returns:
            Normalized node feature tensor
        """
        features_df = to_account_features(txs)
        features_df = features_df[features_df['node_id'].isin(ids)]
        feature_values = features_df[self.data.feature_names].values

        # pad for newly created sybils
        num_missing = len(ids) - feature_values.shape[0]
        if num_missing > 0:
            feature_values = np.pad(feature_values, ((0, num_missing), (0, 0)), mode='constant')

        normalized = self.data.normalize_node_features(feature_values)
        return torch.tensor(normalized, dtype=torch.float32, device=self.device)

    def _compute_gradients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients and fraud probabilities for controlled nodes.

        Returns:
            Tuple of (gradients, probabilities) for controlled nodes
        """
        result = self.graph.compute_gradients(
            model=self.model,
            node_ids=self.controlled_ids
        )

        gradients = result['gradients']
        probs = result['probabilities'][:, 1]

        return gradients, probs

    def _compute_tx_impact(self, tx: Transaction, gradients: torch.Tensor, probs: torch.Tensor) -> float:
        """
        Compute transaction impact using gradient approximation.

        Args:
            tx: Proposed transaction
            gradients: Node gradients
            probs: Current probabilities

        Returns:
            Weighted impact score (negative is beneficial)
        """
        delta_features = self._get_delta_features(tx)
        logit_changes = (delta_features * gradients).sum(dim=1)
        impact = float(probs @ logit_changes)
        return impact

    def _get_delta_features(self, tx: Transaction) -> torch.Tensor:
        """
        Compute feature change from transaction.

        Args:
            tx: Transaction to evaluate

        Returns:
            Feature delta tensor
        """
        augmented_txs = pd.concat([self.txs, tx.to_df()], ignore_index=True)
        features_after = self._get_node_features(augmented_txs, self.controlled_ids)
        return features_after - self.node_features

    def _find_optimal_tx(
        self,
        high_risk_id: Optional[int],
        gradients: torch.Tensor,
        probs: torch.Tensor,
        step: int,
    ) -> Optional[Transaction]:
        """
        Find optimal transaction using DIRECT optimization.

        Args:
            high_risk_id: Highest risk node
            gradients: Current gradients
            probs: Current probabilities
            step: Current step

        Returns:
            Optimal transaction or None
        """
        # calculate budget constraints
        remaining_budget = self.available_budget - self.budget_spent
        required_reserve = self._get_minimal_reserve(step)
        max_usable = max(0, remaining_budget - required_reserve)

        # check minimum feasibility
        min_cost = self.config.value_min + self.config.gas * self.config.gas_price_min
        if max_usable < min_cost:
            return None

        # optimize over candidate pairs
        best_tx = None
        best_score = float('inf')

        # generate candidate pairs
        pairs = self._generate_candidate_pairs(high_risk_id)

        for sender_id, receiver_id in pairs:
            tx, score = self._optimize_transaction_pair(
                sender_id, receiver_id, max_usable,
                gradients, probs
            )

            if tx is not None and score < best_score:
                best_score = score
                best_tx = tx

        # return if beneficial
        return best_tx if best_score < 0 else None

    def _generate_candidate_pairs(self, high_risk_id: int) -> List[Tuple[int, int]]:
        """Generate candidate sender-receiver pairs."""
        pairs = []
        for id in self.controlled_ids:
            if id != high_risk_id:
                pairs.append((high_risk_id, id))
                pairs.append((id, high_risk_id))
        return pairs

    def _optimize_transaction_pair(
        self,
        sender_id: int,
        receiver_id: int,
        max_usable: float,
        gradients: torch.Tensor,
        probs: torch.Tensor
    ) -> Tuple[Optional[Transaction], float]:
        """
        Optimize transaction parameters for a sender-receiver pair.

        Returns:
            Tuple of (transaction, score)
        """
        # set bounds
        max_value = max_usable - self.config.gas * self.config.gas_price_min
        max_value = min(max_value, self.config.value_max)

        if max_value < self.config.value_min:
            return None, float('inf')

        bounds = [
            (np.log10(self.config.value_min), np.log10(max_value)),
            (np.log10(self.config.gas_price_min),
             np.log10(min(self.config.gas_price_max, max_usable / self.config.gas)))
        ]

        def objective(x):
            """Objective: impact + efficiency-based penalty."""
            log_value, log_gas_price = x
            value = 10 ** log_value
            gas_price = 10 ** log_gas_price

            total_cost = value + self.config.gas * gas_price
            if total_cost > max_usable:
                return 1.0

            tx = Transaction(
                from_id=sender_id,
                to_id=receiver_id,
                value=value,
                gas=self.config.gas,
                gas_price=gas_price
            )

            impact = self._compute_tx_impact(tx, gradients, probs)

            # Efficiency-based penalty: cost per unit impact
            # penalty = [μ_g (g/g_min - 1) + μ_w (v/v_min - 1)] / (|J| + ε)
            gas_ratio = gas_price / self.config.gas_price_min - 1.0
            value_ratio = value / self.config.value_min - 1.0
            efficiency_penalty = (
                self.config.gas_penalty_coef * gas_ratio +
                self.config.value_penalty_coef * value_ratio
            ) / (abs(impact) + 1e-6)

            return impact + efficiency_penalty

        try:
            # run optimization
            result = direct(
                objective,
                bounds,
                maxfun=self.config.num_optim_steps,
                maxiter=self.config.maxiter,
                vol_tol=self.config.vol_tol,
                len_tol=self.config.len_tol,
            )

            # extract optimal transaction
            if result.fun < 0:
                log_value, log_gas_price = result.x
                value = 10 ** log_value
                gas_price = 10 ** log_gas_price

                # verify feasibility
                if value + self.config.gas * gas_price <= max_usable:
                    tx = Transaction(
                        from_id=sender_id,
                        to_id=receiver_id,
                        value=value,
                        gas=self.config.gas,
                        gas_price=gas_price
                    )
                    return tx, result.fun
        except Exception:
            pass

        return None, float('inf')

    def _apply_tx(self, tx: Transaction):
        """
        Apply transaction to graph and update state.

        Args:
            tx: Transaction to apply
        """
        # update transaction history
        self.txs = pd.concat([self.txs, tx.to_df()], ignore_index=True)

        # update features
        self.node_features = self._get_node_features(self.txs, self.controlled_ids)
        self.graph.update_node_features(self.controlled_ids, self.node_features)

        # add edge
        edge_features = tx.to_edge_features(device=self.device)
        self.graph.add_edge(tx.from_id, tx.to_id, edge_features)

        # update balances
        self.controlled_accounts[tx.from_id].balance -= tx.total_cost
        self.controlled_accounts[tx.to_id].balance += tx.value

        # track spending
        self.budget_spent += tx.gas_cost

    def _calculate_total_balance(self) -> float:
        """Calculate total balance across controlled accounts."""
        return sum(account.balance for account in self.controlled_accounts.values())

    def _get_minimal_reserve(self, step: int) -> float:
        """
        Calculate minimum reserve for remaining steps.

        Args:
            step: Current step

        Returns:
            Minimum reserve amount
        """
        remaining_steps = max(0, self.config.max_transformations - step - 1)
        min_gas_per_tx = self.config.gas * self.config.gas_price_min
        return remaining_steps * min_gas_per_tx