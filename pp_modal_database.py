"""Modal-analysis kernels for uniformly sampled PeleC snapshot matrices.

POD, SPOD, and DMD are descriptive decompositions of the supplied data; none
of them is an LST eigenvalue solver.  This module keeps that distinction
explicit and provides one validated data contract for probe and future field
snapshots.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SnapshotMatrix:
    """Uniform time-by-space data with coordinates and optional quadrature."""

    time: np.ndarray
    values: np.ndarray
    coordinates: np.ndarray
    variable: str = "disturbance"
    weights: np.ndarray | None = None

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        self.coordinates = np.asarray(self.coordinates, dtype=float)
        if self.time.ndim != 1 or self.time.size < 3:
            raise ValueError("time must be one-dimensional with at least 3 samples")
        if self.values.ndim != 2 or self.values.shape[0] != self.time.size:
            raise ValueError("values must have shape (n_time, n_space)")
        if self.coordinates.ndim == 1:
            self.coordinates = self.coordinates[:, None]
        if self.coordinates.ndim != 2 or self.coordinates.shape[0] != self.values.shape[1]:
            raise ValueError("coordinates must have one row per spatial degree of freedom")
        if not np.all(np.isfinite(self.time)) or not np.all(np.isfinite(self.values)):
            raise ValueError("time and values must contain only finite values")
        difference = np.diff(self.time)
        if np.any(difference <= 0):
            raise ValueError("time must be strictly increasing")
        dt = float(np.median(difference))
        if not np.allclose(difference, dt, rtol=1.0e-7, atol=max(dt * 1.0e-10, 1.0e-15)):
            raise ValueError("SnapshotMatrix requires a uniform time grid")
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=float)
            if self.weights.shape != (self.values.shape[1],):
                raise ValueError("weights must have shape (n_space,)")
            if np.any(~np.isfinite(self.weights)) or np.any(self.weights <= 0):
                raise ValueError("weights must be positive and finite")

    @property
    def dt(self):
        return float(np.median(np.diff(self.time)))

    @property
    def sample_rate(self):
        return 1.0 / self.dt

    def centered_values(self):
        """Return temporal-mean-subtracted values without mutating the input."""
        return self.values - np.mean(self.values, axis=0, keepdims=True)


def _truncated_svd(matrix, n_modes):
    """Return descending truncated SVD, using an iterative solver when useful."""
    matrix = np.asarray(matrix)
    limit = min(matrix.shape)
    n_modes = int(n_modes)
    if not 1 <= n_modes <= limit:
        raise ValueError(f"n_modes must lie in [1, {limit}]")
    if n_modes < limit and limit > 64:
        from scipy.sparse.linalg import svds
        u, singular, vh = svds(matrix, k=n_modes, which="LM")
        order = np.argsort(singular)[::-1]
        return u[:, order], singular[order], vh[order, :]
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    return u[:, :n_modes], singular[:n_modes], vh[:n_modes, :]


def compute_pod(dataset, n_modes=10):
    """Compute weighted snapshot POD and energy fractions."""
    if not isinstance(dataset, SnapshotMatrix):
        raise TypeError("dataset must be a SnapshotMatrix")
    centered = dataset.centered_values()
    weights = (
        np.ones(centered.shape[1], dtype=float)
        if dataset.weights is None else dataset.weights
    )
    sqrt_weights = np.sqrt(weights)
    weighted = centered * sqrt_weights[None, :]
    temporal, singular, spatial_weighted = _truncated_svd(weighted, n_modes)
    modes = spatial_weighted.T / sqrt_weights[:, None]
    total_energy = float(np.sum(weighted ** 2))
    eigenvalues = singular ** 2
    return {
        "modes": modes,
        "temporal_coefficients": temporal * singular[None, :],
        "singular_values": singular,
        "eigenvalues": eigenvalues,
        "energy_fraction": eigenvalues / max(total_energy, 1.0e-30),
        "mean": np.mean(dataset.values, axis=0),
        "coordinates": dataset.coordinates,
        "variable": dataset.variable,
    }


def compute_spod(dataset, nperseg=1024, noverlap=None, n_modes=3,
                 frequency_indices=None, frequency_stride=1):
    """Compute Welch-block SPOD using the method of snapshots at each bin."""
    if not isinstance(dataset, SnapshotMatrix):
        raise TypeError("dataset must be a SnapshotMatrix")
    nperseg = int(nperseg)
    if not 8 <= nperseg <= dataset.values.shape[0]:
        raise ValueError("nperseg must be between 8 and n_time")
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)
    if not 0 <= noverlap < nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")
    step = nperseg - noverlap
    starts = np.arange(0, dataset.values.shape[0] - nperseg + 1, step)
    if starts.size < 2:
        raise ValueError("SPOD requires at least two complete Welch blocks")
    all_frequency = np.fft.rfftfreq(nperseg, dataset.dt)
    if frequency_indices is None:
        selected = np.arange(0, all_frequency.size, max(1, int(frequency_stride)))
    else:
        selected = np.unique(np.asarray(frequency_indices, dtype=int))
        if selected.size == 0 or selected[0] < 0 or selected[-1] >= all_frequency.size:
            raise ValueError("frequency_indices are outside the rFFT grid")

    window = np.hanning(nperseg)
    scale = np.sqrt(dataset.dt / max(np.sum(window ** 2), 1.0e-30))
    blocks = np.empty(
        (starts.size, selected.size, dataset.values.shape[1]), dtype=complex
    )
    for block, start in enumerate(starts):
        segment = dataset.values[start:start + nperseg, :]
        segment = segment - np.mean(segment, axis=0, keepdims=True)
        spectrum = np.fft.rfft(segment * window[:, None], axis=0) * scale
        blocks[block, :, :] = spectrum[selected, :]

    weights = (
        np.ones(dataset.values.shape[1])
        if dataset.weights is None else dataset.weights
    )
    sqrt_weights = np.sqrt(weights)
    mode_count = min(int(n_modes), starts.size, dataset.values.shape[1])
    eigenvalues = np.zeros((selected.size, mode_count), dtype=float)
    modes = np.zeros(
        (selected.size, mode_count, dataset.values.shape[1]), dtype=complex
    )
    for row in range(selected.size):
        realization_matrix = (
            blocks[:, row, :].T * sqrt_weights[:, None] / np.sqrt(starts.size)
        )
        spatial, singular, _ = _truncated_svd(realization_matrix, mode_count)
        eigenvalues[row, :] = singular ** 2
        modes[row, :, :] = (spatial / sqrt_weights[:, None]).T
    return {
        "frequency_hz": all_frequency[selected],
        "frequency_indices": selected,
        "eigenvalues": eigenvalues,
        "modes": modes,
        "n_blocks": starts.size,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "coordinates": dataset.coordinates,
        "variable": dataset.variable,
    }


def compute_dmd(dataset, n_modes=10):
    """Compute exact DMD of mean-subtracted sequential snapshots."""
    if not isinstance(dataset, SnapshotMatrix):
        raise TypeError("dataset must be a SnapshotMatrix")
    snapshots = dataset.centered_values().T
    x = snapshots[:, :-1]
    y = snapshots[:, 1:]
    u, singular, vh = _truncated_svd(x, n_modes)
    inverse_singular = np.diag(1.0 / np.maximum(singular, 1.0e-30))
    reduced_operator = u.conj().T @ y @ vh.conj().T @ inverse_singular
    eigenvalues, eigenvectors = np.linalg.eig(reduced_operator)
    modes = y @ vh.conj().T @ inverse_singular @ eigenvectors
    continuous = np.log(eigenvalues.astype(complex)) / dataset.dt
    amplitudes, _, _, _ = np.linalg.lstsq(modes, snapshots[:, 0], rcond=None)
    order = np.argsort(np.abs(amplitudes))[::-1]
    return {
        "modes": modes[:, order],
        "eigenvalues": eigenvalues[order],
        "continuous_eigenvalues": continuous[order],
        "frequency_hz": np.imag(continuous[order]) / (2.0 * np.pi),
        "growth_rate_per_s": np.real(continuous[order]),
        "amplitudes": amplitudes[order],
        "coordinates": dataset.coordinates,
        "variable": dataset.variable,
    }
