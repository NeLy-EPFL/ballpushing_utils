"""Energy two-sample statistic for fly-grouped UMAP embeddings.

Original author: Tommy Lam (@tkclam). Used by
``ballpushing_utils.umap.analysis`` to compare a genotype's embedding
distribution against its split-specific control.
"""

import numpy as np


def energy_test_fly(
    X: np.ndarray,
    Y: np.ndarray,
    fly_ids_x: np.ndarray,
    fly_ids_y: np.ndarray,
    n_samples: int,
    random_state: int = 0,
):
    from scipy.spatial.distance import cdist

    n_flies_x = fly_ids_x.max() + 1
    n_flies_y = fly_ids_y.max() + 1
    n_flies = n_flies_x + n_flies_y
    fly_ids = np.concatenate((fly_ids_x, fly_ids_y + n_flies_x))
    combined_embeddings = np.concatenate((X, Y))
    dist_matrix = cdist(combined_embeddings, combined_embeddings)

    def get_energy_statistic(x_indices, y_indices):
        a = dist_matrix[np.ix_(x_indices, x_indices)]
        b = dist_matrix[np.ix_(y_indices, y_indices)]
        c = dist_matrix[np.ix_(x_indices, y_indices)]
        return 2 * np.mean(c) - np.mean(a) - np.mean(b)

    E_observed = get_energy_statistic(np.arange(len(X)), np.arange(len(X), len(combined_embeddings)))
    rng = np.random.default_rng(random_state)
    exceed_count = 0
    fly_order = np.arange(n_flies)

    for _ in range(n_samples):
        rng.shuffle(fly_order)
        x_indices = np.where(np.isin(fly_ids, fly_order[:n_flies_x]))[0]
        y_indices = np.where(np.isin(fly_ids, fly_order[n_flies_x:]))[0]
        if get_energy_statistic(x_indices, y_indices) >= E_observed:
            exceed_count += 1

    p = (exceed_count + 1) / (n_samples + 1)
    return E_observed, p
