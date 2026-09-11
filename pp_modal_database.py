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


def trapezoidal_spatial_weights(coordinates):
    """Return one-dimensional quadrature weights for an ordered probe line."""
    coordinate = np.asarray(coordinates, dtype=float)
    if coordinate.ndim == 2:
        coordinate = coordinate[:, 0]
    coordinate = coordinate.ravel()
    if coordinate.size < 2 or not np.all(np.isfinite(coordinate)):
        raise ValueError("at least two finite spatial coordinates are required")
    order = np.argsort(coordinate)
    sorted_coordinate = coordinate[order]
    spacing = np.diff(sorted_coordinate)
    if np.any(spacing <= 0.0):
        raise ValueError("spatial coordinates must be distinct")
    sorted_weights = np.empty(coordinate.size)
    sorted_weights[0] = 0.5 * spacing[0]
    sorted_weights[-1] = 0.5 * spacing[-1]
    if coordinate.size > 2:
        sorted_weights[1:-1] = 0.5 * (
            sorted_coordinate[2:] - sorted_coordinate[:-2]
        )
    weights = np.empty_like(sorted_weights)
    weights[order] = sorted_weights
    return weights


def scalar_compressible_energy_weights(
        coordinates, variable, rho_base, temperature_base,
        gamma=1.4, gas_constant=287.05):
    """Build a diagonal scalar contribution to a compressible-energy norm.

    A single probe variable cannot represent the complete Chu energy because
    thermodynamic cross terms and all velocity components are unavailable.
    These weights therefore describe only the declared scalar contribution.
    """
    quadrature = trapezoidal_spatial_weights(coordinates)
    rho = np.broadcast_to(
        np.asarray(rho_base, dtype=float), quadrature.shape
    )
    temperature = np.broadcast_to(
        np.asarray(temperature_base, dtype=float), quadrature.shape
    )
    if np.any(rho <= 0.0) or np.any(temperature <= 0.0):
        raise ValueError("base density and temperature must be positive")
    pressure = rho * float(gas_constant) * temperature
    name = str(variable).lower()
    if "pressure" in name or name in ("p", "column_3"):
        physical = 1.0 / (float(gamma) * pressure)
        norm = "pressure contribution p'^2/(gamma*p_bar)"
    elif "velocity" in name or name in ("u", "column_2") or name.startswith("u "):
        physical = rho
        norm = "kinetic contribution rho_bar*u'^2"
    elif "density" in name or name in ("rho", "column_1"):
        physical = float(gamma) * pressure / rho ** 2
        norm = "isentropic density contribution gamma*p_bar*rho'^2/rho_bar^2"
    elif "temperature" in name or name in ("t", "column_4"):
        cv = float(gas_constant) / (float(gamma) - 1.0)
        physical = rho * cv / temperature
        norm = "diagonal thermal contribution rho_bar*Cv*T'^2/T_bar"
    else:
        raise ValueError(
            f"no declared scalar compressible-energy weight for {variable!r}"
        )
    weights = quadrature * physical
    weights /= np.mean(weights)
    return {
        "weights": weights,
        "quadrature_weights": quadrature,
        "physical_factor": physical,
        "norm_definition": norm,
        "scope": "diagonal scalar contribution; not the complete Chu norm",
    }


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


def prepare_spod_blocks(dataset, nperseg=1024, noverlap=None, n_modes=3,
                        frequency_indices=None, frequency_stride=1):
    """Prepare the read-only Welch blocks used by the independent SPOD bins."""
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
    blocks = np.empty((starts.size, selected.size, dataset.values.shape[1]), dtype=complex)
    for block, start in enumerate(starts):
        segment = dataset.values[start:start + nperseg, :]
        segment = segment - np.mean(segment, axis=0, keepdims=True)
        spectrum = np.fft.rfft(segment * window[:, None], axis=0) * scale
        blocks[block, :, :] = spectrum[selected, :]
    weights = np.ones(dataset.values.shape[1]) if dataset.weights is None else dataset.weights
    mode_count = min(int(n_modes), starts.size, dataset.values.shape[1])
    return {
        "blocks": blocks,
        "frequency_hz": all_frequency[selected],
        "frequency_indices": selected,
        "weights": np.asarray(weights),
        "mode_count": mode_count,
        "n_blocks": int(starts.size),
        "nperseg": nperseg,
        "noverlap": noverlap,
        "coordinates": dataset.coordinates,
        "variable": dataset.variable,
    }


def compute_spod_frequency_batch(blocks, weights, mode_count, start_index, stop_index):
    """Compute independent SPOD frequency rows for one contiguous batch."""
    blocks = np.asarray(blocks)
    weights = np.asarray(weights, dtype=float)
    mode_count = int(mode_count)
    sqrt_weights = np.sqrt(weights)
    eigenvalues = np.zeros((int(stop_index) - int(start_index), mode_count), dtype=float)
    modes = np.zeros((int(stop_index) - int(start_index), mode_count, blocks.shape[2]), dtype=complex)
    for row, frequency_index in enumerate(range(int(start_index), int(stop_index))):
        realization_matrix = blocks[:, frequency_index, :].T * sqrt_weights[:, None] / np.sqrt(blocks.shape[0])
        spatial, singular, _ = _truncated_svd(realization_matrix, mode_count)
        eigenvalues[row, :] = singular ** 2
        modes[row, :, :] = (spatial / sqrt_weights[:, None]).T
    return {"start_index": int(start_index), "eigenvalues": eigenvalues, "modes": modes}


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
        "retained_singular_values": singular,
        "retained_condition_number": float(
            singular[0] / max(singular[-1], 1.0e-30)
        ),
        "coordinates": dataset.coordinates,
        "variable": dataset.variable,
    }


def compute_modal_sensitivity(
        dataset, pod_modes=2, dmd_ranks=(2, 4, 8),
        window_fractions=((0.0, 0.5), (0.5, 1.0))):
    """Quantify POD subspace and DMD rank/window sensitivity."""
    if not isinstance(dataset, SnapshotMatrix):
        raise TypeError("dataset must be a SnapshotMatrix")
    if int(pod_modes) < 1:
        raise ValueError("pod_modes must be positive")
    windows = []
    for start_fraction, stop_fraction in window_fractions:
        if not 0.0 <= start_fraction < stop_fraction <= 1.0:
            raise ValueError("window fractions must satisfy 0 <= start < stop <= 1")
        first = int(np.floor(start_fraction * dataset.time.size))
        last = int(np.ceil(stop_fraction * dataset.time.size))
        if last - first < 8:
            raise ValueError("each modal sensitivity window needs >=8 samples")
        windows.append(SnapshotMatrix(
            dataset.time[first:last],
            dataset.values[first:last],
            dataset.coordinates,
            variable=dataset.variable,
            weights=dataset.weights,
        ))
    if len(windows) < 2:
        raise ValueError(
            "modal sensitivity requires at least two comparison windows"
        )
    pod_modes = min(
        int(pod_modes), dataset.values.shape[1],
        min(window.time.size for window in windows),
    )
    pod_results = [
        compute_pod(window, n_modes=pod_modes) for window in windows
    ]
    weights = (
        np.ones(dataset.values.shape[1])
        if dataset.weights is None else dataset.weights
    )
    reference_modes = pod_results[0]["modes"]
    pod_similarity = []
    for result in pod_results[1:]:
        overlap = (
            reference_modes.conj().T
            @ (weights[:, None] * result["modes"])
        )
        singular = np.linalg.svd(overlap, compute_uv=False)
        pod_similarity.append(float(np.min(np.clip(singular, 0.0, 1.0))))

    accepted_ranks = sorted(set(
        min(
            int(rank), dataset.values.shape[1],
            min(window.time.size - 1 for window in windows),
        )
        for rank in dmd_ranks if int(rank) >= 1
    ))
    if not accepted_ranks:
        raise ValueError("at least one positive DMD sensitivity rank is required")
    dmd_frequency = np.full((len(windows), len(accepted_ranks)), np.nan)
    dmd_growth = np.full_like(dmd_frequency, np.nan)
    dmd_condition = np.full_like(dmd_frequency, np.nan)
    for window_index, window in enumerate(windows):
        for rank_index, rank in enumerate(accepted_ranks):
            result = compute_dmd(window, n_modes=rank)
            dominant = int(np.argmax(np.abs(result["amplitudes"])))
            dmd_frequency[window_index, rank_index] = abs(
                result["frequency_hz"][dominant]
            )
            dmd_growth[window_index, rank_index] = result[
                "growth_rate_per_s"
            ][dominant]
            dmd_condition[window_index, rank_index] = result[
                "retained_condition_number"
            ]
    return {
        "window_fractions": np.asarray(window_fractions, dtype=float),
        "pod_subspace_min_cosine_to_first_window": np.asarray(
            pod_similarity, dtype=float
        ),
        "dmd_ranks": np.asarray(accepted_ranks, dtype=int),
        "dmd_dominant_frequency_hz": dmd_frequency,
        "dmd_dominant_growth_rate_per_s": dmd_growth,
        "dmd_retained_condition_number": dmd_condition,
        "interpretation": (
            "Descriptive window/rank sensitivity; no LST eigenmode attribution"
        ),
    }
