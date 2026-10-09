import numpy as np


def fluctuations(values, weights):
    """Weighted mean of ``values`` over axis 0 and its fluctuations per configuration.

    ``delta_i = weights_i (values_i - mean) / mean(weights)``: the fluctuations
    of ``weights * values`` and ``weights`` projected with the gradient of
    their ratio of means. Uniform weights give ``values_i - mean``.
    """
    mean = np.average(values, axis=0, weights=weights)
    w_bcast = weights.reshape((-1,) + (1,) * (values.ndim - 1))
    delta = w_bcast * (values - mean) / np.mean(weights)
    return mean, delta
