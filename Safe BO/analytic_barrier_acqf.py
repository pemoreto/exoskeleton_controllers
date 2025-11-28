import numpy as np
import math
import sys
from typing import Dict, Optional, Tuple, Union
import torch
from botorch.models.model import Model
from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
from torch import Tensor
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition.analytic import (
    LogConstrainedExpectedImprovement,
    ConstrainedExpectedImprovement,
    _preprocess_constraint_bounds,
    _scaled_improvement,
    _ei_helper,
)


class ConstrainedExpectedImprovementSafeUpdate(AnalyticAcquisitionFunction):
    r"""Constrained Expected Improvement (barrier-constrained).

    Computes the analytic expected improvement for a Normal posterior
    distribution, weighted by a probability of feasibility. The objective and
    constraints are assumed to be independent and have Gaussian posterior
    distributions. Only supports non-batch mode (i.e. `q=1`). The model should be
    multi-outcome, with the index of the objective and constraints passed to
    the constructor.

    `Constrained_EI(x) = EI(x) * Product_i P(y_i \in [lower_i, upper_i])`,
    where `y_i ~ constraint_i(x)` and `lower_i`, `upper_i` are the lower and
    upper bounds for the i-th constraint, respectively.

    Example:
        # example where the 0th output has a non-negativity constraint and
        # 1st output is the objective
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> constraints = {0: (0.0, None)}
        >>> cEI = ConstrainedExpectedImprovement(model, 0.2, 1, constraints)
        >>> cei = cEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        objective_index: int,
        constraints: Dict[int, Tuple[Optional[float], Optional[float]]],
        beta: float = 1.0,
        tau: float = 0.01,
        maximize: bool = True,
    ) -> None:
        r"""Analytic Constrained Expected Improvement.

        Args:
            model: A fitted multi-output model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best feasible function value observed so far (assumed noiseless).
            objective_index: The index of the objective.
            constraints: A dictionary of the form `{i: [lower, upper]}`, where
                `i` is the output index, and `lower` and `upper` are lower and upper
                bounds on that output (resp. interpreted as -Inf / Inf if None)
            maximize: If True, consider the problem a maximization problem.
        """
        # Use AcquisitionFunction constructor to avoid check for posterior transform.
        super(AnalyticAcquisitionFunction, self).__init__(model=model)
        self.posterior_transform = None
        self.beta = beta
        self.tau = tau
        self.maximize = maximize
        self.objective_index = objective_index
        self.constraints = constraints
        self.register_buffer("best_f", torch.as_tensor(best_f))
        _preprocess_constraint_bounds(self, constraints=constraints)
        self.register_forward_pre_hook(convert_to_target_pre_hook)

    def _compute_barrier(
        self,
        acqf: Union[LogConstrainedExpectedImprovement, ConstrainedExpectedImprovement],
        means: Tensor,
        sigmas: Tensor,
        beta: float,
        tau: float,
    ) -> Tensor:
        r"""Compute log-barrier of the feasibility probability for each batch of X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
            means: A `(b) x m`-dim Tensor of means.
            sigmas: A `(b) x m`-dim Tensor of standard deviations.
        Returns:
            A `b`-dim tensor of log-barrier feasibility probabilities
        """
        acqf.to(device=means.device)
        barrier_val = torch.zeros_like(means[..., 0])

        # print("means.shape", means.shape)
        # print("sigmas.shape", sigmas.shape)
        # print(means[..., 1])
        # print(sigmas[..., 1])
        # sys.exit()

        constraints = list(acqf.constraints.keys())
        for c in constraints:
            ucb = means[..., c] + beta * sigmas[..., c]
            # print("UCB:", ucb)
            barrier_val += tau * torch.log(-ucb)
            # print("Barrier val:", barrier_val)

        barrier_val = torch.nan_to_num(barrier_val, nan=-1e30)

        # print("barrier_val.shape", barrier_val.shape)
        # sys.exit()
        # print("barrier_val", barrier_val)

        return barrier_val

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate Constrained Expected Improvement on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Expected Improvement values at the given
            design points `X`.
        """
        means, sigmas = self._mean_and_sigma(X)  # (b) x 1 + (m = num constraints)
        ind = self.objective_index
        mean_obj, sigma_obj = means[..., ind], sigmas[..., ind]
        u = _scaled_improvement(mean_obj, sigma_obj, self.best_f, self.maximize)
        ei = sigma_obj * _ei_helper(u)
        barrier_term = self._compute_barrier(self, means=means, sigmas=sigmas, beta=self.beta, tau=self.tau)
        return ei.add(barrier_term)
