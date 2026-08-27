from __future__ import annotations

from typing import cast

from pocketlab.sensor_models import (
    SensorAnalysis,
    SensorCapability,
    SensorKind,
    SensorRecordingUpload,
)


def _analyzers():
    # Local imports keep the registry acyclic and make missing modules fail loudly.
    from pocketlab.analyzers.acceleration import AccelerationAnalyzer
    from pocketlab.analyzers.scalar import LightAnalyzer, PressureAnalyzer, ProximityAnalyzer
    from pocketlab.analyzers.sensitive import LocationAnalyzer, MicrophoneAnalyzer
    from pocketlab.analyzers.vector import GyroscopeAnalyzer, MagnetometerAnalyzer

    analyzers = (
        AccelerationAnalyzer(),
        GyroscopeAnalyzer(),
        MagnetometerAnalyzer(),
        LightAnalyzer(),
        PressureAnalyzer(),
        ProximityAnalyzer(),
        MicrophoneAnalyzer(),
        LocationAnalyzer(),
    )
    return {analyzer.sensor: analyzer for analyzer in analyzers}


def _apply_public_replay_boundary(
    upload: SensorRecordingUpload,
    analysis: SensorAnalysis,
) -> SensorAnalysis:
    provenance = upload.provenance
    if provenance.source != "public_replay":
        return analysis
    rank = {"low": 0, "medium": 1, "high": 2}
    ceiling = provenance.public_analysis_confidence_ceiling
    if ceiling is None:  # pragma: no cover - SensorProvenance validates this first
        raise ValueError("public replay provenance is missing its confidence ceiling")
    confidence = analysis.confidence
    if rank[confidence] > rank[ceiling]:
        confidence = ceiling
    invalidated = set(provenance.public_invalidated_metric_keys)
    warnings = list(analysis.warnings)
    for disclosure in provenance.public_processing_disclosures:
        if disclosure not in warnings:
            warnings.append(disclosure)
    return analysis.model_copy(
        update={
            "confidence": confidence,
            "warnings": warnings,
            "metrics": [item for item in analysis.metrics if item.key not in invalidated],
        }
    )


def analyze_sensor_recording(upload: SensorRecordingUpload) -> SensorAnalysis:
    analyzer = _analyzers().get(upload.sensor)
    if analyzer is None:
        raise ValueError(f"no deterministic v2 analyzer is registered for {upload.sensor}")
    return _apply_public_replay_boundary(upload, analyzer.analyze(upload))


def sensor_capabilities() -> list[SensorCapability]:
    analyzers = _analyzers()
    capabilities: list[SensorCapability] = []
    for sensor in (
        "accelerometer",
        "gyroscope",
        "magnetometer",
        "light",
        "pressure",
        "proximity",
        "microphone",
        "location",
        "bluetooth",
    ):
        typed_sensor = cast(SensorKind, sensor)
        analyzer = analyzers.get(typed_sensor)
        if analyzer is not None:
            capabilities.append(analyzer.capability())
        else:
            capabilities.append(
                SensorCapability(
                    sensor=typed_sensor,
                    maturity="detectable",
                    limitations=["尚无可授权诊断结论的确定性分析器。"],
                )
            )
    return capabilities
