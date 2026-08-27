from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pocketlab.analyzers.registry import analyze_sensor_recording
from pocketlab.public_replay_dataset import (
    PublicDataClass,
    PublicReplayDatasetManifest,
    PublicReplayRecording,
    get_public_replay_dataset,
    read_public_replay_recording,
    verify_public_source_files,
)
from pocketlab.sensor_models import SensorAnalysis, SensorRecordingUpload

PRIVACY_DUAL_DATASET_ID = "light-privacy-dual-20231127-v1"
PRIVACY_DUAL_RECORDINGS = frozenset(
    {"privacy-dual-occluder", "privacy-dual-touch"}
)
PHYPhOX_SNR_DATASET_ID = "light-phyphox-snr-20260611-v1"
PHYPhOX_SNR_RECORDING_ID = "light-hand-wave-session"
BRIGHTER_TIME_DATASET_ID = "light-brighter-time-20220701-v1"

ClaimKind = Literal[
    "descriptive_difference",
    "temporal_pattern",
    "naturalistic_context",
    "causal_effect",
    "distance_law",
    "behavior_identification",
    "absolute_calibration",
]
ClaimSupportStatus = Literal[
    "supported_with_limitations",
    "unsupported",
]
ProcessingLevel = Literal[
    "raw_sensor_series",
    "author_derived_series",
    "aggregated_summary_series",
]
DeviceClass = Literal["android_tablet", "android_phone", "mixed_android_phones"]

_CADENCE_FIELDS = frozenset(
    {"sampling_rate_hz", "sampling_jitter_ratio", "max_sampling_gap_ratio"}
)


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


class PublicLightSourceIntegrity(_FrozenStrictModel):
    file_id: str = Field(min_length=1, max_length=80)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verified: Literal[True] = True


class PublicLightProvenance(_FrozenStrictModel):
    dataset_id: str = Field(min_length=1, max_length=100)
    recording_id: str | None = Field(default=None, max_length=80)
    recording_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    data_class: PublicDataClass
    processing_level: ProcessingLevel
    device_class: DeviceClass
    device_alias: str | None = Field(default=None, max_length=120)
    acquisition_app: str = Field(min_length=2, max_length=120)
    source_title: str = Field(min_length=3, max_length=240)
    source_url: str = Field(min_length=10, max_length=500)
    doi: str | None = Field(default=None, max_length=160)
    license_spdx: str = Field(min_length=2, max_length=40)
    source_files: tuple[PublicLightSourceIntegrity, ...] = Field(min_length=1)
    invalidated_analysis_fields: tuple[str, ...]
    invalidated_metric_keys: tuple[str, ...]
    processing_disclosures: tuple[str, ...]
    claim_boundary: tuple[str, ...] = Field(min_length=1)
    privacy_categories: tuple[str, ...]
    deployment_scope: Literal["local_only", "local_and_deployed"]
    allowed_operations: tuple[str, ...] = Field(min_length=1)
    requires_user_acknowledgement: bool
    local_replay_only: bool
    gate_c_eligible: Literal[False] = False
    agent_ready: Literal[False] = False


class PublicLightMetric(_FrozenStrictModel):
    key: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=80)
    value: float
    unit: str = Field(max_length=24)


class PublicLightTraceResult(_FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    result_kind: Literal["public_light_trace"] = "public_light_trace"
    provenance: PublicLightProvenance
    sample_count: int = Field(ge=2)
    confidence: Literal["low", "medium", "high"]
    warnings: tuple[str, ...]
    metrics: tuple[PublicLightMetric, ...]
    zero_fraction: float = Field(ge=0.0, le=1.0)
    unique_levels: int = Field(ge=1)
    transition_count: int = Field(ge=0)
    temporal_axis_valid: bool
    duration_s: float | None = Field(default=None, gt=0.0)
    sampling_rate_hz: float | None = Field(default=None, gt=0.0)
    transition_rate_hz: float | None = Field(default=None, ge=0.0)
    transition_count_interpretation: Literal[
        "time_ordered_raw_signal", "order_only_processed_series"
    ]


class PublicLightConditionSummary(_FrozenStrictModel):
    recording_id: str = Field(min_length=1, max_length=80)
    condition_id: str = Field(min_length=1, max_length=80)
    acquisition_id: str = Field(min_length=1, max_length=100)
    repeat_index: int = Field(ge=1)
    sample_count: int = Field(ge=2)
    median_illuminance_lx: float = Field(ge=0.0)
    illuminance_iqr_lx: float = Field(ge=0.0)
    coefficient_of_variation_ratio: float = Field(ge=0.0)
    zero_fraction: float = Field(ge=0.0, le=1.0)


class PublicLightConditionComparison(_FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    result_kind: Literal["registered_light_condition_comparison"] = (
        "registered_light_condition_comparison"
    )
    dataset_id: Literal["light-privacy-dual-20231127-v1"]
    scene_id: Literal["mannequin-als-occlusion"]
    left: PublicLightConditionSummary
    right: PublicLightConditionSummary
    difference_direction: Literal["right_minus_left"] = "right_minus_left"
    median_difference_lx: float
    median_ratio: float | None = Field(default=None, ge=0.0)
    iqr_difference_lx: float
    coefficient_of_variation_difference_ratio: float
    zero_fraction_difference: float = Field(ge=-1.0, le=1.0)
    provenance: tuple[PublicLightProvenance, PublicLightProvenance]
    independent_acquisitions: Literal[True] = True
    repeats_per_condition: Literal[1] = 1
    descriptive_only: Literal[True] = True
    causal_claim_allowed: Literal[False] = False
    cross_source_aggregation_allowed: Literal[False] = False
    gate_c_eligible: Literal[False] = False
    agent_ready: Literal[False] = False
    limitations: tuple[str, ...] = Field(min_length=1)


class PublicLightDistributionSummary(_FrozenStrictModel):
    minimum: float = Field(ge=0.0)
    q25: float = Field(ge=0.0)
    median: float = Field(ge=0.0)
    q75: float = Field(ge=0.0)
    maximum: float = Field(ge=0.0)
    unit: Literal["lx"] = "lx"


class NaturalisticLightContextResult(_FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    result_kind: Literal["naturalistic_light_context"] = "naturalistic_light_context"
    dataset_id: Literal["light-brighter-time-20220701-v1"]
    provenance: PublicLightProvenance
    participant_count: int = Field(ge=1)
    independent_series_count: int = Field(ge=1)
    observation_count: int = Field(ge=1)
    participant_median_lux_distribution: PublicLightDistributionSummary
    query_lux: float | None = Field(default=None, ge=0.0)
    query_percentile_rank_among_participant_medians: float | None = Field(
        default=None, ge=0.0, le=100.0
    )
    temporal_claim_allowed: Literal[False] = False
    causal_claim_allowed: Literal[False] = False
    calibration_claim_allowed: Literal[False] = False
    health_claim_allowed: Literal[False] = False
    cross_device_absolute_comparison_allowed: Literal[False] = False
    gate_c_eligible: Literal[False] = False
    agent_ready: Literal[False] = False
    limitations: tuple[str, ...] = Field(min_length=1)


class LightClaimAuditResult(_FrozenStrictModel):
    schema_version: Literal["1.0"] = "1.0"
    result_kind: Literal["light_claim_audit"] = "light_claim_audit"
    claim_kind: ClaimKind
    evidence_refs: tuple[str, ...]
    status: ClaimSupportStatus
    allowed_phrasing: tuple[str, ...]
    forbidden_phrasing: tuple[str, ...] = Field(min_length=1)
    required_missing_evidence: tuple[str, ...]
    gate_c_eligible: Literal[False] = False
    agent_ready: Literal[False] = False


def _dataset_profile(
    dataset_id: str,
) -> tuple[ProcessingLevel, DeviceClass, str]:
    if dataset_id == PRIVACY_DUAL_DATASET_ID:
        return "raw_sensor_series", "android_tablet", "AndroSensor Very Fast"
    if dataset_id == PHYPhOX_SNR_DATASET_ID:
        return "author_derived_series", "android_phone", "phyphox Light"
    if dataset_id == BRIGHTER_TIME_DATASET_ID:
        return (
            "aggregated_summary_series",
            "mixed_android_phones",
            "Brighter Time study application",
        )
    raise ValueError(f"unsupported public Light dataset: {dataset_id}")


def _find_recording(
    manifest: PublicReplayDatasetManifest,
    recording_id: str,
) -> PublicReplayRecording:
    matches = [item for item in manifest.recordings if item.recording_id == recording_id]
    if not matches:
        raise KeyError(
            f"Unknown public Light recording {recording_id!r} in {manifest.dataset_id}"
        )
    if len(matches) != 1:  # pragma: no cover - manifest validation requires uniqueness
        raise ValueError(f"duplicate public Light recording_id: {recording_id}")
    return matches[0]


def _load_recording_analysis(
    pack_dir: Path,
    manifest: PublicReplayDatasetManifest,
    recording: PublicReplayRecording,
) -> tuple[SensorRecordingUpload, SensorAnalysis]:
    upload = read_public_replay_recording(pack_dir, manifest, recording)
    analysis = analyze_sensor_recording(upload)
    if analysis.model_dump(mode="json") != recording.reference_analysis.model_dump(mode="json"):
        raise ValueError(
            f"public Light reference analysis regression for {recording.recording_id}"
        )
    return upload, analysis


def _provenance(
    manifest: PublicReplayDatasetManifest,
    recording: PublicReplayRecording | None,
) -> PublicLightProvenance:
    processing_level, device_class, acquisition_app = _dataset_profile(
        manifest.dataset_id
    )
    disclosures = (
        tuple(recording.processing_disclosures)
        if recording is not None
        else tuple(
            dict.fromkeys(
                disclosure
                for item in manifest.recordings
                for disclosure in item.processing_disclosures
            )
        )
    )
    invalidated_fields = (
        tuple(recording.invalidated_analysis_fields)
        if recording is not None
        else tuple(
            sorted(
                {
                    field
                    for item in manifest.recordings
                    for field in item.invalidated_analysis_fields
                }
            )
        )
    )
    invalidated_metrics = (
        tuple(recording.invalidated_metric_keys)
        if recording is not None
        else tuple(
            sorted(
                {
                    metric
                    for item in manifest.recordings
                    for metric in item.invalidated_metric_keys
                }
            )
        )
    )
    privacy_categories = tuple(
        dict.fromkeys(
            [
                *manifest.privacy_review.source_sensitive_categories,
                *manifest.privacy_review.replay_sensitive_categories,
            ]
        )
    )
    return PublicLightProvenance(
        dataset_id=manifest.dataset_id,
        recording_id=recording.recording_id if recording is not None else None,
        recording_sha256=recording.sha256 if recording is not None else None,
        data_class=manifest.data_class,
        processing_level=processing_level,
        device_class=device_class,
        device_alias=recording.device_alias if recording is not None else None,
        acquisition_app=acquisition_app,
        source_title=manifest.source.title,
        source_url=manifest.source.record_url,
        doi=manifest.source.doi,
        license_spdx=manifest.source.license_spdx,
        source_files=tuple(
            PublicLightSourceIntegrity(file_id=item.file_id, sha256=item.sha256)
            for item in manifest.source_files
        ),
        invalidated_analysis_fields=invalidated_fields,
        invalidated_metric_keys=invalidated_metrics,
        processing_disclosures=disclosures,
        claim_boundary=tuple(manifest.claim_boundary),
        privacy_categories=privacy_categories,
        deployment_scope=manifest.privacy_review.deployment_scope,
        allowed_operations=tuple(manifest.privacy_review.allowed_operations),
        requires_user_acknowledgement=(
            manifest.privacy_review.requires_user_acknowledgement
        ),
        local_replay_only=manifest.privacy_review.deployment_scope == "local_only",
    )


def _temporal_axis_valid(
    manifest: PublicReplayDatasetManifest,
    recording: PublicReplayRecording,
) -> bool:
    if _CADENCE_FIELDS.intersection(recording.invalidated_analysis_fields):
        return False
    source_roles = {
        item.file_id: item.role
        for item in manifest.source_files
    }
    if any(source_roles[source_id] != "raw" for source_id in recording.source_file_ids):
        return False
    transformations = {
        item.transformation_id: item for item in manifest.transformations
    }
    return not any(
        transformations[transformation_id].kind == "author_derived"
        for transformation_id in recording.transformation_ids
    )


def _metric_map(result: PublicLightTraceResult) -> dict[str, float]:
    return {item.key: item.value for item in result.metrics}


def inspect_public_light_trace(
    root: Path,
    dataset_id: str,
    recording_id: str,
) -> PublicLightTraceResult:
    pack_dir, manifest = get_public_replay_dataset(root, dataset_id)
    if manifest.sensor != "light":
        raise ValueError(f"public Light tools cannot inspect sensor={manifest.sensor}")
    _dataset_profile(manifest.dataset_id)
    verify_public_source_files(pack_dir, manifest)
    recording = _find_recording(manifest, recording_id)
    upload, analysis = _load_recording_analysis(pack_dir, manifest, recording)
    values = [sample.values["illuminance"] for sample in upload.samples]
    transition_count = sum(right != left for left, right in pairwise(values))
    temporal_axis_valid = _temporal_axis_valid(manifest, recording)
    invalidated_fields = set(recording.invalidated_analysis_fields)
    duration_s = None if "duration_s" in invalidated_fields else float(analysis.duration_s)
    sampling_rate_hz = (
        None
        if "sampling_rate_hz" in invalidated_fields
        else float(analysis.sampling_rate_hz)
    )
    transition_rate_hz = (
        float(transition_count / analysis.duration_s)
        if temporal_axis_valid and analysis.duration_s > 0
        else None
    )
    return PublicLightTraceResult(
        provenance=_provenance(manifest, recording),
        sample_count=len(values),
        confidence=analysis.confidence,
        warnings=tuple(analysis.warnings),
        metrics=tuple(
            PublicLightMetric(
                key=metric.key,
                label=metric.label,
                value=float(metric.value),
                unit=metric.unit,
            )
            for metric in analysis.metrics
        ),
        zero_fraction=float(sum(value == 0 for value in values) / len(values)),
        unique_levels=len(set(values)),
        transition_count=transition_count,
        temporal_axis_valid=temporal_axis_valid,
        duration_s=duration_s,
        sampling_rate_hz=sampling_rate_hz,
        transition_rate_hz=transition_rate_hz,
        transition_count_interpretation=(
            "time_ordered_raw_signal"
            if temporal_axis_valid
            else "order_only_processed_series"
        ),
    )


def _condition_summary(
    trace: PublicLightTraceResult,
    recording: PublicReplayRecording,
) -> PublicLightConditionSummary:
    metrics = _metric_map(trace)
    return PublicLightConditionSummary(
        recording_id=recording.recording_id,
        condition_id=recording.condition_id,
        acquisition_id=recording.acquisition_id,
        repeat_index=recording.repeat_index,
        sample_count=recording.sample_count,
        median_illuminance_lx=metrics["median_illuminance_lx"],
        illuminance_iqr_lx=metrics["illuminance_iqr_lx"],
        coefficient_of_variation_ratio=metrics["coefficient_of_variation_ratio"],
        zero_fraction=trace.zero_fraction,
    )


def compare_registered_light_conditions(
    root: Path,
    dataset_id: str,
    left_recording_id: str,
    right_recording_id: str,
) -> PublicLightConditionComparison:
    if dataset_id != PRIVACY_DUAL_DATASET_ID:
        raise ValueError(
            "the first registered Light comparison only supports Privacy Dual"
        )
    if left_recording_id == right_recording_id or {
        left_recording_id,
        right_recording_id,
    } != PRIVACY_DUAL_RECORDINGS:
        raise ValueError(
            "the registered comparison requires exactly the Privacy Dual occluder and touch recordings"
        )
    pack_dir, manifest = get_public_replay_dataset(root, dataset_id)
    verify_public_source_files(pack_dir, manifest)
    left_recording = _find_recording(manifest, left_recording_id)
    right_recording = _find_recording(manifest, right_recording_id)
    if (
        left_recording.scene_id != "mannequin-als-occlusion"
        or right_recording.scene_id != left_recording.scene_id
        or not left_recording.independent_measurement
        or not right_recording.independent_measurement
    ):
        raise ValueError("Privacy Dual registered comparison invariants are not satisfied")
    left_trace = inspect_public_light_trace(root, dataset_id, left_recording_id)
    right_trace = inspect_public_light_trace(root, dataset_id, right_recording_id)
    left = _condition_summary(left_trace, left_recording)
    right = _condition_summary(right_trace, right_recording)
    median_ratio = (
        float(right.median_illuminance_lx / left.median_illuminance_lx)
        if left.median_illuminance_lx > 0
        else None
    )
    return PublicLightConditionComparison(
        dataset_id=PRIVACY_DUAL_DATASET_ID,
        scene_id="mannequin-als-occlusion",
        left=left,
        right=right,
        median_difference_lx=float(
            right.median_illuminance_lx - left.median_illuminance_lx
        ),
        median_ratio=median_ratio,
        iqr_difference_lx=float(right.illuminance_iqr_lx - left.illuminance_iqr_lx),
        coefficient_of_variation_difference_ratio=float(
            right.coefficient_of_variation_ratio
            - left.coefficient_of_variation_ratio
        ),
        zero_fraction_difference=float(right.zero_fraction - left.zero_fraction),
        provenance=(left_trace.provenance, right_trace.provenance),
        limitations=(
            "Each registered condition has one acquisition, so the result is descriptive only.",
            "Different recording durations and sample counts are not treated as repeats.",
            "The Android tablet recordings were acquired with AndroSensor, not phyphox.",
            "The retained light rhythm is privacy-sensitive and is approved for local replay only.",
            "No cross-source averaging, causal inference, behavior identification, or Gate C credit is allowed.",
        ),
    )


def _linear_quantile(values: Sequence[float], proportion: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty public Light cohort")
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(ordered[lower_index])
    fraction = position - lower_index
    return float(
        ordered[lower_index]
        + (ordered[upper_index] - ordered[lower_index]) * fraction
    )


def summarize_naturalistic_light_context(
    root: Path,
    dataset_id: str,
    query_lux: float | None = None,
) -> NaturalisticLightContextResult:
    if dataset_id != BRIGHTER_TIME_DATASET_ID:
        raise ValueError(
            "the first naturalistic Light context tool only supports Brighter Time"
        )
    if query_lux is not None and (not math.isfinite(query_lux) or query_lux < 0):
        raise ValueError("query_lux must be a finite non-negative value")
    pack_dir, manifest = get_public_replay_dataset(root, dataset_id)
    verify_public_source_files(pack_dir, manifest)
    participant_medians: list[float] = []
    observation_count = 0
    for recording in manifest.recordings:
        _, analysis = _load_recording_analysis(pack_dir, manifest, recording)
        participant_medians.append(float(analysis.metric_value("median_illuminance_lx")))
        observation_count += recording.sample_count
    percentile = (
        float(
            100.0
            * sum(value <= query_lux for value in participant_medians)
            / len(participant_medians)
        )
        if query_lux is not None
        else None
    )
    return NaturalisticLightContextResult(
        dataset_id=BRIGHTER_TIME_DATASET_ID,
        provenance=_provenance(manifest, None),
        participant_count=len(manifest.recordings),
        independent_series_count=sum(
            recording.independent_measurement for recording in manifest.recordings
        ),
        observation_count=observation_count,
        participant_median_lux_distribution=PublicLightDistributionSummary(
            minimum=_linear_quantile(participant_medians, 0.0),
            q25=_linear_quantile(participant_medians, 0.25),
            median=_linear_quantile(participant_medians, 0.5),
            q75=_linear_quantile(participant_medians, 0.75),
            maximum=_linear_quantile(participant_medians, 1.0),
        ),
        query_lux=float(query_lux) if query_lux is not None else None,
        query_percentile_rank_among_participant_medians=percentile,
        limitations=(
            "Values are task-period log10(lx) summaries reconstructed to lx, not raw sensor samples.",
            "Participants used different unreported Android phone models across irregular multi-day observations.",
            "Participant-median percentiles are dataset context, not health, calibration, or causal thresholds.",
            "Absolute lux values are not pooled into a calibrated cross-device estimate.",
            "This public replay cannot validate temporal dynamics, distance laws, Gate C, or agent readiness.",
        ),
    )


def _known_evidence_ref(value: str) -> bool:
    if value in {
        PRIVACY_DUAL_DATASET_ID,
        PHYPhOX_SNR_DATASET_ID,
        BRIGHTER_TIME_DATASET_ID,
        f"{PHYPhOX_SNR_DATASET_ID}/{PHYPhOX_SNR_RECORDING_ID}",
    }:
        return True
    if value in {
        f"{PRIVACY_DUAL_DATASET_ID}/{recording_id}"
        for recording_id in PRIVACY_DUAL_RECORDINGS
    }:
        return True
    prefix = f"{BRIGHTER_TIME_DATASET_ID}/participant-series-"
    if value.startswith(prefix):
        suffix = value.removeprefix(prefix)
        return len(suffix) == 3 and suffix.isdigit() and 1 <= int(suffix) <= 66
    return False


def _references_dataset(evidence_refs: tuple[str, ...], dataset_id: str) -> bool:
    return any(value == dataset_id or value.startswith(f"{dataset_id}/") for value in evidence_refs)


def _references_registered_privacy_pair(evidence_refs: tuple[str, ...]) -> bool:
    if PRIVACY_DUAL_DATASET_ID in evidence_refs:
        return True
    required = {
        f"{PRIVACY_DUAL_DATASET_ID}/{recording_id}"
        for recording_id in PRIVACY_DUAL_RECORDINGS
    }
    return required <= set(evidence_refs)


def audit_light_claim_support(
    claim_kind: ClaimKind,
    evidence_refs: Sequence[str],
) -> LightClaimAuditResult:
    known_claims = {
        "descriptive_difference",
        "temporal_pattern",
        "naturalistic_context",
        "causal_effect",
        "distance_law",
        "behavior_identification",
        "absolute_calibration",
    }
    if claim_kind not in known_claims:
        raise ValueError(f"unknown public Light claim kind: {claim_kind}")
    if isinstance(evidence_refs, (str, bytes)):
        raise TypeError("evidence_refs must be a sequence of registered evidence IDs")
    refs = tuple(evidence_refs)
    if len(refs) != len(set(refs)):
        raise ValueError("evidence_refs must not contain duplicates")
    unknown = [value for value in refs if not _known_evidence_ref(value)]
    if unknown:
        raise ValueError(f"unregistered public Light evidence reference: {unknown[0]}")

    forbidden_common = (
        "Do not claim Gate C, market validation, or agent readiness from public replay.",
        "Do not average absolute lux values across these differently acquired sources.",
    )
    if claim_kind == "causal_effect":
        return LightClaimAuditResult(
            claim_kind=claim_kind,
            evidence_refs=refs,
            status="unsupported",
            allowed_phrasing=(),
            forbidden_phrasing=(
                "Do not state that touch, occlusion, or another action caused the observed difference.",
                *forbidden_common,
            ),
            required_missing_evidence=(
                "A preregistered controlled design with randomized conditions and independent repeats.",
                "Appropriate causal controls and analysis beyond a two-acquisition descriptive replay.",
            ),
        )
    if claim_kind == "distance_law":
        return LightClaimAuditResult(
            claim_kind=claim_kind,
            evidence_refs=refs,
            status="unsupported",
            allowed_phrasing=(),
            forbidden_phrasing=(
                "Do not claim inverse-square behavior or any fitted light-distance exponent.",
                *forbidden_common,
            ),
            required_missing_evidence=(
                "Registered source distances, background measurements, geometry controls, and repeated conditions.",
                "Self-collected phone evidence that passes the Light distance protocol gates.",
            ),
        )
    if claim_kind == "behavior_identification":
        return LightClaimAuditResult(
            claim_kind=claim_kind,
            evidence_refs=refs,
            status="unsupported",
            allowed_phrasing=(),
            forbidden_phrasing=(
                "Do not infer a person's action, identity, image, or private activity from the light trace.",
                *forbidden_common,
            ),
            required_missing_evidence=(
                "A separately authorized privacy and ethics protocol is required before any behavioral inference work.",
            ),
        )
    if claim_kind == "absolute_calibration":
        return LightClaimAuditResult(
            claim_kind=claim_kind,
            evidence_refs=refs,
            status="unsupported",
            allowed_phrasing=(),
            forbidden_phrasing=(
                "Do not claim that any replayed device is an absolutely calibrated lux meter.",
                *forbidden_common,
            ),
            required_missing_evidence=(
                "A traceable reference light meter and a preregistered device calibration procedure.",
            ),
        )

    supported = False
    allowed: tuple[str, ...]
    missing: tuple[str, ...]
    forbidden: tuple[str, ...]
    if claim_kind == "descriptive_difference":
        supported = _references_registered_privacy_pair(refs)
        allowed = (
            "The two registered Privacy Dual tablet acquisitions differ descriptively in the reported light metrics.",
        )
        forbidden = (
            "Do not describe the two single acquisitions as replicated or causal evidence.",
            *forbidden_common,
        )
        missing = (
            "Both registered Privacy Dual recordings are required for the descriptive comparison.",
        )
    elif claim_kind == "temporal_pattern":
        supported = _references_dataset(
            refs, PRIVACY_DUAL_DATASET_ID
        ) or _references_dataset(refs, PHYPhOX_SNR_DATASET_ID)
        allowed = (
            "A registered replay sequence contains ordered illuminance changes within its declared processing boundary.",
        )
        forbidden = (
            "Do not report original cadence or frequency from an upsampled or aggregated replay.",
            "Do not interpret the privacy-sensitive light rhythm as a specific human behavior.",
            *forbidden_common,
        )
        missing = (
            "A registered raw trace with a valid time axis is required for temporal rates.",
        )
    else:
        supported = BRIGHTER_TIME_DATASET_ID in refs
        allowed = (
            "A queried lux value can be located descriptively among Brighter Time participant-level medians.",
        )
        forbidden = (
            "Do not treat the cohort context as a health, calibration, causal, or temporal threshold.",
            *forbidden_common,
        )
        missing = (
            "The registered Brighter Time dataset is required for naturalistic participant-level context.",
        )
    return LightClaimAuditResult(
        claim_kind=claim_kind,
        evidence_refs=refs,
        status="supported_with_limitations" if supported else "unsupported",
        allowed_phrasing=allowed if supported else (),
        forbidden_phrasing=forbidden,
        required_missing_evidence=() if supported else missing,
    )
