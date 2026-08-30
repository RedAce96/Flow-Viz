"""Single authoritative catalog of user-facing scientific recipes."""

from __future__ import annotations

from dataclasses import dataclass

from .protocol import RegisteredWorkflow, assert_workflow_contract


@dataclass(frozen=True)
class RecipeDefinition:
    name: str
    question: str
    summary: str
    supported_dimensions: tuple[int, ...]
    supported_geometries: tuple[str, ...]
    required_inputs: tuple[str, ...]
    required_fields: tuple[str, ...]
    assumptions: tuple[str, ...]
    outputs: tuple[str, ...]
    limitations: tuple[str, ...]
    workflow: str
    dependencies: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    executor_module: str = "pelecpost.analysis.executors"


GEOMETRIES = ("flat_plate", "wedge", "polyline", "volume_fraction")


def _recipe(
    name: str,
    question: str,
    summary: str,
    *,
    inputs: tuple[str, ...],
    fields: tuple[str, ...],
    assumptions: tuple[str, ...],
    outputs: tuple[str, ...],
    limitations: tuple[str, ...],
    geometries: tuple[str, ...] = GEOMETRIES,
    workflow: str | None = None,
    dependencies: tuple[str, ...] = (),
    conflicts: tuple[str, ...] = (),
) -> RecipeDefinition:
    return RecipeDefinition(
        name=name,
        question=question,
        summary=summary,
        supported_dimensions=(2,),
        supported_geometries=geometries,
        required_inputs=inputs,
        required_fields=fields,
        assumptions=assumptions,
        outputs=outputs,
        limitations=limitations,
        workflow=workflow or name,
        dependencies=dependencies,
        conflicts=conflicts,
    )


RECIPES: dict[str, RecipeDefinition] = {
    item.name: item
    for item in (
        _recipe(
            "flow_overview",
            "What does the resolved two-dimensional flow field look like?",
            "Contours, derived fields, line samples, and optional streamlines.",
            inputs=("plotfiles",),
            fields=(),
            assumptions=("Requested fields are represented by the plotfile data.",),
            outputs=("field.contours", "field.lines", "field.streamlines"),
            limitations=("Images are descriptive and do not establish causality.",),
        ),
        _recipe(
            "boundary_layer_reference",
            "How does a flat-plate boundary layer compare with a laminar reference?",
            "Boundary-layer integral quantities and compressible similarity profiles.",
            inputs=("plotfiles",),
            fields=("density", "x_velocity", "temperature"),
            geometries=("flat_plate",),
            assumptions=("Laminar flow.", "Zero streamwise pressure gradient."),
            outputs=(
                "boundary_layer.profiles", "boundary_layer.thickness", "boundary_layer.gip",
            ),
            limitations=("The similarity curve is a reference, not an LST or PSE result.",),
        ),
        _recipe(
            "surface_diagnostics",
            "Is the reconstructed wall geometry and near-wall sampling trustworthy?",
            "Surface coordinates, normals, wall samples, and fit-quality diagnostics.",
            inputs=("plotfiles",),
            fields=("pressure", "temperature", "x_velocity", "y_velocity"),
            assumptions=("A unique fluid-facing surface normal can be established.",),
            outputs=("surface.curve", "surface.samples", "surface.quality", "surface.figure"),
            limitations=("Resolution and normal-fit quality constrain wall quantities.",),
            dependencies=("geometry.surface",),
        ),
        _recipe(
            "aerodynamic_forces",
            "What pressure, viscous, thermal, force, and moment loads act on the body?",
            "Wall reconstruction, component integration, baselines, and sensitivity.",
            inputs=("plotfiles",),
            fields=("pressure", "temperature", "x_velocity", "y_velocity"),
            assumptions=("Newtonian stress and configured transport properties apply.",),
            outputs=(
                "forces.history", "forces.components", "forces.sensitivity",
                "forces.control_volume", "forces.probe_linkage",
            ),
            limitations=("General EB output is validated_2d_eb_v1, not externally certified.",),
            dependencies=("geometry.surface",),
        ),
        _recipe(
            "probe_spectrum",
            "What stationary frequency content is present in the probe measurements?",
            "One-sided FFT/Welch spectra, coherence, and confidence summaries.",
            inputs=("probes",),
            fields=(),
            assumptions=("The selected record is approximately stationary.",),
            outputs=(
                "spectral.psd", "spectral.coherence", "spectral.confidence",
                "spectral.figure",
            ),
            limitations=("Finite records and windowing limit frequency discrimination.",),
        ),
        _recipe(
            "single_pulse_response",
            "What response follows a known finite laser pulse?",
            "Quiescent-baseline and finite-record source/response deconvolution.",
            inputs=("probes",),
            fields=(),
            assumptions=("The configured pulse model represents the source history.",),
            outputs=("pulse.source_spectrum", "pulse.transfer", "pulse.validity"),
            limitations=("Small source amplitudes are masked rather than inverted.",),
        ),
        _recipe(
            "directional_wave",
            "What direction, wavelength, phase speed, and amplification are measured?",
            "Spatial FFT, coherence-gated complex wavenumber, amplification, and k-omega.",
            inputs=("probes",),
            fields=(),
            assumptions=("Probe coordinates form a suitable approximately uniform aperture.",),
            outputs=(
                "wave.spatial_spectrum", "wave.wavenumber", "wave.komega",
                "wave.komega_sensitivity", "wave.komega.figure",
            ),
            limitations=("Measurements alone do not constitute LST/PSE or causal evidence.",),
        ),
        _recipe(
            "transient_wavepacket",
            "How does a transient packet arrive and propagate?",
            "STFT, filtered envelopes, arrival times, group velocity, and uncertainty.",
            inputs=("probes",),
            fields=(),
            assumptions=("A localized packet exists in the selected time-frequency band.",),
            outputs=("transient.stft", "transient.envelope", "transient.group_velocity"),
            limitations=("Arrival detection depends on bandwidth and signal-to-noise ratio.",),
        ),
        _recipe(
            "nonlinear_coupling",
            "Are statistically significant quadratic frequency interactions measured?",
            "Surrogate and FDR-controlled bicoherence and triad screening.",
            inputs=("probes",),
            fields=(),
            assumptions=("Enough independent segments exist for surrogate inference.",),
            outputs=("nonlinear.bicoherence", "nonlinear.triads"),
            limitations=("Bicoherence is association, not proof of causal energy transfer.",),
        ),
        _recipe(
            "modal_screening",
            "Which coherent low-rank structures are visible in the measurements?",
            "Weighted POD, SPOD, DMD, and rank/window conditioning sensitivity.",
            inputs=("probes",),
            fields=(),
            assumptions=("The probe measure and weights are meaningful for the question.",),
            outputs=("modal.pod", "modal.spod", "modal.dmd", "modal.sensitivity"),
            limitations=("Probe modes are measurement modes, not full-domain eigenmodes.",),
        ),
        _recipe(
            "case_comparison",
            "How do compatible products differ between two registered runs?",
            "Schema- and provenance-checked comparison of existing artifacts.",
            inputs=("comparison_archives",),
            fields=(),
            assumptions=("Compared artifacts use compatible coordinates and preprocessing.",),
            outputs=("comparison.metrics", "comparison.figures"),
            limitations=("Incompatible artifacts are rejected rather than interpolated silently.",),
        ),
    )
}


INTERNAL_WORKFLOWS = frozenset({
    "input.plotfiles",
    "input.probes",
    "input.comparison_archives",
    "geometry.surface",
})

WORKFLOWS = {name: RegisteredWorkflow(definition) for name, definition in RECIPES.items()}
for _workflow in WORKFLOWS.values():
    assert_workflow_contract(_workflow)


def recipe_for(name: str) -> RecipeDefinition:
    try:
        return RECIPES[name]
    except KeyError as exc:
        available = ", ".join(sorted(RECIPES))
        raise KeyError(f"Unknown recipe {name!r}; available recipes: {available}") from exc


def workflow_for(name: str) -> RegisteredWorkflow:
    try:
        return WORKFLOWS[name]
    except KeyError as exc:
        available = ", ".join(sorted(WORKFLOWS))
        raise KeyError(f"Unknown workflow {name!r}; available workflows: {available}") from exc
