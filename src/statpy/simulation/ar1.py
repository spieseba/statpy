"""AR(1) chains with known autocorrelations (Wolff, hep-lat/0306017, App. C.2)."""

import numpy as np


def generate_chains(tau: float | np.ndarray, num_configurations: int, rng: np.random.Generator) -> np.ndarray:
    """Independent AR(1) chains with unit variance and integrated autocorrelation time ``tau`` (Wolff eqs. 68-70).

    Returns shape ``(num_configurations,) + tau.shape``, one chain per entry of ``tau``, each starting in
    equilibrium. Exact: rho(t) = a^t with a = (2 tau - 1) / (2 tau + 1).
    """
    tau = np.asarray(tau)
    if np.any(tau < 0.5):
        raise ValueError(f"generate_chains: tau must be >= 1/2, got {tau.min()}")
    a = (2 * tau - 1) / (2 * tau + 1)
    eta = rng.standard_normal((num_configurations,) + a.shape)
    nu = np.empty_like(eta)
    nu[0] = eta[0]
    for i in range(1, num_configurations):
        nu[i] = np.sqrt(1 - a**2) * eta[i] + a * nu[i - 1]
    return nu
