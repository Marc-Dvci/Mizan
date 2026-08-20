"""Workstream C, second half: which observation to buy next.

Linear data-worth analysis through the Schur complement. Given a Jacobian of every
observation, existing and candidate, with respect to every parameter, the posterior
covariance under any subset of observations follows in closed form, and so does the
variance of any linear forecast under that subset. Ranking candidate instruments by
the variance they remove from the forecast is the question a water authority actually
faces: not whether to meter, but which meter to install first.

The Schur complement is implemented here directly and cross-checked against pyEMU's
implementation on the same inputs by `scripts/04_voi.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DataWorth:
    jac: np.ndarray            # (nobs, npar) sensitivity of every observation
    sigma_obs: np.ndarray      # (nobs,) observation standard deviation
    sigma_par: np.ndarray      # (npar,) prior parameter standard deviation
    names: list[str]           # observation names

    def posterior_cov(self, use: np.ndarray) -> np.ndarray:
        """Posterior parameter covariance given the observation subset `use`."""
        J = self.jac[use]
        w = 1.0 / self.sigma_obs[use] ** 2
        prior_inv = np.diag(1.0 / self.sigma_par ** 2)
        return np.linalg.inv(prior_inv + J.T @ (w[:, None] * J))

    def forecast_sd(self, use: np.ndarray, f: np.ndarray) -> float:
        """Posterior standard deviation of the linear forecast `f`."""
        return float(np.sqrt(max(f @ self.posterior_cov(use) @ f, 0.0)))

    def worth(self, base: np.ndarray, candidates: dict[str, np.ndarray],
              f: np.ndarray) -> dict[str, float]:
        """Forecast standard deviation removed by adding each candidate to `base`."""
        s0 = self.forecast_sd(base, f)
        return {k: s0 - self.forecast_sd(base | m, f) for k, m in candidates.items()}

    def greedy(self, base: np.ndarray, candidates: dict[str, np.ndarray],
               f: np.ndarray, n: int) -> tuple[list[str], list[float]]:
        """Forward selection of `n` instruments, and the forecast standard deviation
        after each one is added."""
        use = base.copy()
        pool = dict(candidates)
        order, curve = [], [self.forecast_sd(use, f)]
        for _ in range(min(n, len(pool))):
            best, best_sd = None, np.inf
            for k, m in pool.items():
                sd = self.forecast_sd(use | m, f)
                if sd < best_sd:
                    best, best_sd = k, sd
            if best is None:
                break
            use = use | pool.pop(best)
            order.append(best)
            curve.append(best_sd)
        return order, curve
