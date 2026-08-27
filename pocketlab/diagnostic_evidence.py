from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pocketlab.schemas import (
    DiagnosticControlEffect,
    DiagnosticMeasurementFact,
    DiagnosticMeasurementTask,
)
from pocketlab.sensor_models import AnalysisMetric, SensorAnalysis, SensorKind
from pocketlab.sensor_requirements import sensor_requirement
from pocketlab.store import SessionStore

DiagnosticProvenance = Literal[
    "phyphox_remote",
    "phone_upload",
    "file_import",
    "public_replay",
    "test_fixture",
    "legacy_session",
]


@dataclass(frozen=True)
class DiagnosticRecordingView:
    session_id: str
    label: str
    device: str
    notes: str
    sensor: SensorKind
    analysis: SensorAnalysis
    provenance_source: DiagnosticProvenance
    provenance_details: dict[str, str]


def _legacy_analysis(session: object) -> SensorAnalysis:
    analysis = session.analysis
    return SensorAnalysis(
        sensor="accelerometer",
        analyzer_id="pocketlab.acceleration.legacy-adapter",
        analyzer_version="1.0.0",
        sample_count=analysis.sample_count,
        duration_s=analysis.duration_s,
        sampling_rate_hz=analysis.sampling_rate_hz,
        sampling_jitter_ratio=analysis.sampling_jitter_ratio,
        max_sampling_gap_ratio=analysis.max_sampling_gap_ratio,
        confidence=analysis.confidence,
        warnings=list(analysis.warnings),
        metrics=[
            AnalysisMetric(
                key="selected_axis_rms_m_s2",
                label="主导轴 RMS",
                value=analysis.rms_acceleration_m_s2,
                unit="m/s^2",
            ),
            AnalysisMetric(
                key="selected_axis_peak_to_peak_m_s2",
                label="主导轴峰峰值",
                value=analysis.peak_to_peak_m_s2,
                unit="m/s^2",
            ),
            AnalysisMetric(
                key="dominant_frequency_hz",
                label="主频",
                value=analysis.dominant_frequency_hz,
                unit="Hz",
            ),
            AnalysisMetric(
                key="spectral_snr_db",
                label="频谱信噪比",
                value=analysis.spectral_snr_db,
                unit="dB",
            ),
        ],
    )


def get_diagnostic_recording(
    sessions: SessionStore,
    session_id: str,
) -> DiagnosticRecordingView:
    """Return one closed sensor/analysis view without hiding v1/v2 provenance."""

    try:
        stored = sessions.get_sensor_recording(session_id)
    except KeyError:
        stored = sessions.get(session_id)
        return DiagnosticRecordingView(
            session_id=stored.session_id,
            label=stored.upload.label,
            device=stored.upload.device,
            notes=stored.upload.notes,
            sensor="accelerometer",
            analysis=_legacy_analysis(stored),
            provenance_source="legacy_session",
            provenance_details={},
        )
    provenance = stored.upload.provenance
    details = {
        key: str(value)
        for key, value in {
            "experiment_title": provenance.experiment_title,
            "remote_session": provenance.remote_session,
            "config_sha256": provenance.config_sha256,
            "capture_group_id": provenance.capture_group_id,
            "clock_id": provenance.clock_id,
            "general_case_id": provenance.general_case_id,
            "general_task_id": provenance.general_task_id,
            "public_dataset_id": provenance.public_dataset_id,
            "public_recording_id": provenance.public_recording_id,
            "public_source_url": provenance.public_source_url,
            "public_license_spdx": provenance.public_license_spdx,
        }.items()
        if value not in {None, ""}
    }
    return DiagnosticRecordingView(
        session_id=stored.session_id,
        label=stored.upload.label,
        device=stored.upload.device,
        notes=stored.upload.notes,
        sensor=stored.upload.sensor,
        analysis=stored.analysis,
        provenance_source=provenance.source,
        provenance_details=details,
    )


def selected_metric(
    task: DiagnosticMeasurementTask,
    recording: DiagnosticRecordingView,
) -> AnalysisMetric:
    requirement = sensor_requirement(task.required_sensor)
    metric_key = task.target_metric_key or requirement.default_metric_key
    if metric_key is None:
        raise ValueError(f"{task.required_sensor} has no numeric diagnostic metric")
    if metric_key not in requirement.accepted_metric_keys:
        raise ValueError(
            f"unregistered diagnostic metric for {task.required_sensor}: {metric_key}"
        )
    metric = next(
        (item for item in recording.analysis.metrics if item.key == metric_key),
        None,
    )
    if metric is None:
        return _raise_missing_metric(recording, metric_key)
    return metric


def _raise_missing_metric(
    recording: DiagnosticRecordingView,
    metric_key: str,
) -> AnalysisMetric:
    available = [item.key for item in recording.analysis.metrics]
    raise ValueError(
        f"recording {recording.session_id} does not expose {metric_key}; available={available}"
    )


def build_measurement_fact(
    *,
    task: DiagnosticMeasurementTask,
    recording: DiagnosticRecordingView,
    baseline: DiagnosticRecordingView | None = None,
) -> tuple[DiagnosticMeasurementFact, DiagnosticControlEffect | None]:
    metric = selected_metric(task, recording)
    baseline_metric = None
    relation: Literal[
        "single_observation", "increase", "decrease", "within_repeatability"
    ] = "single_observation"
    absolute_delta = None
    relative_delta = None
    effect = None
    source_ids = [recording.session_id]
    if baseline is not None:
        if baseline.sensor != recording.sensor:
            raise ValueError("a numeric control contrast requires the same sensor")
        baseline_metric = selected_metric(task, baseline)
        if baseline_metric.unit != metric.unit:
            raise ValueError("a numeric control contrast requires the same metric unit")
        absolute_delta = metric.value - baseline_metric.value
        logarithmic_metric = metric.unit.strip().casefold() in {"db", "dbfs"}
        relative_delta = (
            None
            if logarithmic_metric
            else absolute_delta / abs(baseline_metric.value)
            if abs(baseline_metric.value) > 1e-12
            else None
        )
        # A decibel is already a logarithmic ratio, so dividing a dB delta by a
        # negative baseline has no physical meaning.  Treat 3 dB as the smallest
        # material level contrast; ratio-scale metrics keep the registered 5% gate.
        threshold = (
            3.0
            if logarithmic_metric
            else max(
                0.05 * max(abs(metric.value), abs(baseline_metric.value), 1e-12),
                1e-12,
            )
        )
        relation = (
            "increase"
            if absolute_delta > threshold
            else "decrease"
            if absolute_delta < -threshold
            else "within_repeatability"
        )
        observed = (
            "increase"
            if relation == "increase"
            else "decrease"
            if relation == "decrease"
            else "no_change"
        )
        expected = task.expected_effect
        matches = expected == "unknown" or expected == observed
        if expected == "change":
            matches = observed in {"increase", "decrease"}
        warnings: list[str] = []
        if _source_family(baseline) != _source_family(recording):
            warnings.append(
                "数据来源不同"
                f"（{_source_family(baseline)} vs {_source_family(recording)}）"
            )
        sampling_ratio = max(
            baseline.analysis.sampling_rate_hz,
            recording.analysis.sampling_rate_hz,
        ) / max(
            min(baseline.analysis.sampling_rate_hz, recording.analysis.sampling_rate_hz),
            1e-9,
        )
        if sampling_ratio > 1.10:
            warnings.append("采样率偏差超过 10%")
        duration_ratio = max(
            baseline.analysis.duration_s,
            recording.analysis.duration_s,
        ) / max(min(baseline.analysis.duration_s, recording.analysis.duration_s), 1e-9)
        if duration_ratio > 1.50:
            warnings.append("记录时长偏差超过 50%")
        comparable = not warnings
        effect = DiagnosticControlEffect(
            baseline_task_id=task.comparison_task_id or "",
            baseline_session_id=baseline.session_id,
            sensor=recording.sensor,
            metric_key=metric.key,
            metric_unit=metric.unit,
            baseline_value=baseline_metric.value,
            current_value=metric.value,
            absolute_delta=absolute_delta,
            relative_change_ratio=relative_delta,
            rms_change_ratio=(
                relative_delta
                if recording.sensor == "accelerometer"
                and metric.key == "selected_axis_rms_m_s2"
                and relative_delta is not None
                else 0.0
            ),
            observed_effect=observed,
            matches_expected_effect=matches and comparable,
            comparable=comparable,
            comparison_warnings=warnings,
        )
        source_ids.insert(0, baseline.session_id)
    fact = DiagnosticMeasurementFact(
        fact_id=f"fact-{task.task_id}-{len(source_ids)}",
        sensor=recording.sensor,
        metric_key=metric.key,
        metric_label=metric.label,
        metric_unit=metric.unit,
        value=metric.value,
        quality=recording.analysis.confidence,
        source_session_ids=source_ids,
        provenance_source=recording.provenance_source,
        baseline_value=baseline_metric.value if baseline_metric is not None else None,
        absolute_delta=absolute_delta,
        relative_delta_ratio=relative_delta,
        relation=relation,
        analyzer_id=recording.analysis.analyzer_id,
        analyzer_version=recording.analysis.analyzer_version,
        sample_count=recording.analysis.sample_count,
        duration_s=recording.analysis.duration_s,
        sampling_rate_hz=recording.analysis.sampling_rate_hz,
        analysis_warnings=list(recording.analysis.warnings),
        companion_metrics=[
            item for item in recording.analysis.metrics if item.key != metric.key
        ],
    )
    return fact, effect


def _source_family(recording: DiagnosticRecordingView) -> str:
    if recording.provenance_source != "legacy_session":
        return recording.provenance_source
    normalized = recording.device.strip().casefold()
    if normalized.startswith("phyphox"):
        return "phyphox_remote"
    if "simulator" in normalized or "synthetic" in normalized:
        return "test_fixture"
    return "file_import"
