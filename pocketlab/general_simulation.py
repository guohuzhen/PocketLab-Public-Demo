from __future__ import annotations

import hashlib
import math
from typing import Literal, Self

from pydantic import Field, model_validator

from pocketlab.analyzers import analyze_sensor_recording
from pocketlab.general_acquisition import (
    AcquisitionAlignmentAttestation,
    GeneralAcquisitionReference,
    GeneralEvidenceEnvelope,
    analyzed_acquisition_from_upload,
    bind_general_evidence,
)
from pocketlab.general_exploration_models import StrictFrozenModel
from pocketlab.general_exploration_state import GeneralExperimentCase
from pocketlab.sensor_models import (
    SensorChannelDefinition,
    SensorKind,
    SensorProvenance,
    SensorRecordingUpload,
    SensorSample,
)

GeneralSimulationProfile = Literal[
    "clear_contrast",
    "near_equal",
    "reversed_contrast",
    "inverse_square_light",
]


class GeneralSimulationMeasurementRequest(StrictFrozenModel):
    expected_revision: int = Field(ge=1)
    task_id: str = Field(
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$",
        max_length=80,
    )
    profile: GeneralSimulationProfile = "clear_contrast"
    controls_confirmed: bool


class GeneralSimulationCaptureMetadata(StrictFrozenModel):
    source: Literal["protocol_emulator"] = "protocol_emulator"
    profile: GeneralSimulationProfile
    sensors: tuple[SensorKind, ...] = Field(min_length=1, max_length=3)
    recording_ids: tuple[str, ...] = Field(min_length=1, max_length=3)
    physical_evidence: Literal[False] = False
    user_phone_evidence_candidate: Literal[False] = False
    gate_c_credit: Literal[False] = False

    @model_validator(mode="after")
    def capture_sets_are_closed(self) -> Self:
        if len(self.sensors) != len(set(self.sensors)):
            raise ValueError("simulated capture sensors must be unique")
        if len(self.recording_ids) != len(self.sensors):
            raise ValueError("simulated capture requires one recording per sensor")
        if len(self.recording_ids) != len(set(self.recording_ids)):
            raise ValueError("simulated recording IDs must be unique")
        return self


class GeneralSimulationMeasurementResponse(StrictFrozenModel):
    case: GeneralExperimentCase
    evidence: tuple[GeneralEvidenceEnvelope, ...] = Field(min_length=1, max_length=3)
    simulation: GeneralSimulationCaptureMetadata

    @model_validator(mode="after")
    def response_retains_simulated_lineage(self) -> Self:
        if any(
            item.lineage.source != "protocol_emulator"
            or not item.lineage.simulated
            or item.lineage.physical_evidence
            or item.lineage.user_phone_evidence_candidate
            for item in self.evidence
        ):
            raise ValueError("simulated response cannot claim physical or phone evidence")
        if {item.lineage.recording_id for item in self.evidence} != set(
            self.simulation.recording_ids
        ):
            raise ValueError("simulated metadata must match returned evidence")
        case_evidence_ids = {item.evidence_id for item in self.case.evidence}
        if not {item.evidence_id for item in self.evidence} <= case_evidence_ids:
            raise ValueError("simulated evidence must already be committed to the case")
        return self


def _condition_scale(
    condition_id: str,
    repeat_index: int,
    profile: GeneralSimulationProfile,
) -> float:
    if not 1 <= repeat_index <= 32:
        raise ValueError("simulation repeat index must stay inside the Exploration task budget")
    repeat_pattern = (-0.008, 0.0, 0.008)
    repeat_offset = repeat_pattern[(repeat_index - 1) % len(repeat_pattern)]
    if condition_id == "reference":
        return 1.0 + repeat_offset
    comparison = {
        "clear_contrast": 1.8,
        "near_equal": 1.015,
        "reversed_contrast": 0.55,
        "inverse_square_light": 0.255,
    }[profile]
    return comparison + repeat_offset


def build_general_simulated_recording(
    sensor: SensorKind,
    *,
    condition_id: str,
    repeat_index: int,
    profile: GeneralSimulationProfile,
) -> SensorRecordingUpload:
    """Generate deterministic analyzer-contract data for a labelled rehearsal only."""

    if sensor == "bluetooth":
        raise ValueError("Bluetooth has no numeric simulation contract")
    scale = _condition_scale(condition_id, repeat_index, profile)
    channels: dict[str, str]
    rows: list[tuple[float, dict[str, float]]]
    if sensor == "accelerometer":
        channels = {axis: "m/s^2" for axis in ("x", "y", "z")}
        rows = [
            (
                index * 20.0,
                {
                    "x": scale * math.sin(2.0 * math.pi * 2.0 * index / 50.0),
                    "y": 0.04 * math.sin(2.0 * math.pi * index / 50.0),
                    "z": 9.81,
                },
            )
            for index in range(250)
        ]
    elif sensor == "gyroscope":
        channels = {axis: "rad/s" for axis in ("x", "y", "z")}
        rows = [
            (
                index * 100.0,
                {
                    "x": scale * (0.1 if index % 2 == 0 else -0.1),
                    "y": scale * (0.2 if index % 2 == 0 else -0.2),
                    "z": 0.0,
                },
            )
            for index in range(30)
        ]
    elif sensor == "magnetometer":
        channels = {"x": "uT", "y": "uT", "z": "uT", "accuracy": "state"}
        rows = [
            (
                index * 100.0,
                {
                    "x": scale * (30.0 if index % 2 == 0 else -30.0),
                    "y": scale * (40.0 if index % 2 == 0 else -40.0),
                    "z": 0.0,
                    "accuracy": 3.0,
                },
            )
            for index in range(30)
        ]
    elif sensor == "light":
        channels = {"illuminance": "lx"}
        rows = [
            (index * 500.0, {"illuminance": scale * (30.0 + index * 0.1)})
            for index in range(12)
        ]
    elif sensor == "pressure":
        channels = {"pressure": "hPa"}
        trend_per_sample = 0.012 * scale
        rows = [
            (
                index * 500.0,
                {
                    "pressure": 1000.0
                    - trend_per_sample * index
                    + (0.002 if index % 2 else -0.002)
                },
            )
            for index in range(20)
        ]
    elif sensor == "proximity":
        channels = {"distance": "cm"}
        reference_block = 6
        comparison_block = (
            2 if profile == "clear_contrast" else 6 if profile == "near_equal" else 10
        )
        block = reference_block if condition_id == "reference" else comparison_block
        rows = [
            (index * 500.0, {"distance": 0.0 if (index // block) % 2 else 5.0})
            for index in range(20)
        ]
    elif sensor == "microphone":
        channels = {"level_db": "dB_relative"}
        base = 35.0 + 18.0 * (scale - 1.0)
        rows = [
            (index * 100.0, {"level_db": base + repeat_index * 0.05 + (index % 4) * 0.2})
            for index in range(30)
        ]
    elif sensor == "location":
        channels = {
            "lat": "deg",
            "lon": "deg",
            "accuracy": "m",
            "speed": "m/s",
            "status": "state",
        }
        step = 0.00001 * scale
        rows = [
            (
                index * 1000.0,
                {
                    "lat": 0.0,
                    "lon": index * step,
                    "accuracy": 0.2,
                    "speed": 1.1 * scale,
                    "status": 1.0,
                },
            )
            for index in range(12)
        ]
    else:  # pragma: no cover - SensorKind is closed above
        raise ValueError(f"unsupported simulated sensor: {sensor}")
    return SensorRecordingUpload(
        label=f"SIMULATED · {sensor} · {condition_id} · repeat {repeat_index}",
        device="PocketLab protocol simulator",
        sensor=sensor,
        notes="Synthetic analyzer-contract rehearsal; not physical or phone evidence.",
        channels={name: SensorChannelDefinition(unit=unit) for name, unit in channels.items()},
        samples=[
            SensorSample(timestamp_ms=timestamp_ms, values=values)
            for timestamp_ms, values in rows
        ],
        provenance=SensorProvenance(
            source="test_fixture",
            privacy_acknowledged=sensor in {"microphone", "location"},
        ),
    )


def build_general_simulated_evidence(
    case: GeneralExperimentCase,
    request: GeneralSimulationMeasurementRequest,
) -> tuple[GeneralEvidenceEnvelope, ...]:
    case = GeneralExperimentCase.model_validate(case.model_dump(mode="python"))
    task = case.current_task
    if case.status != "collecting" or task is None:
        raise ValueError("terminal cases cannot accept simulated measurements")
    if request.expected_revision != case.revision or request.task_id != task.task_id:
        raise ValueError("stale or foreign simulated measurement")
    if not request.controls_confirmed:
        raise ValueError("simulated rehearsal still requires control confirmation")
    if set(case.protocol.selected_sources) != {"protocol_emulator"}:
        raise ValueError("simulation is only available to an explicit rehearsal protocol")
    alignment = None
    if case.protocol.alignment == "simultaneous" and len(task.sensors) > 1:
        digest = hashlib.sha256(f"{case.case_id}:{task.task_id}".encode()).hexdigest()[:20]
        alignment = AcquisitionAlignmentAttestation(
            capture_group_id=f"sim-group-{digest}",
            clock_id="simulated-shared-clock",
            maximum_alignment_error_ms=0.0,
            method="server_timestamp_correlation",
        )
    evidence = []
    for sensor in task.sensors:
        upload = build_general_simulated_recording(
            sensor,
            condition_id=task.condition_id,
            repeat_index=task.repeat_index,
            profile=request.profile,
        )
        analysis = analyze_sensor_recording(upload)
        digest = hashlib.sha256(
            (
                f"{case.case_id}:{task.task_id}:{sensor}:{request.profile}:"
                f"{request.expected_revision}"
            ).encode()
        ).hexdigest()[:24]
        acquisition = analyzed_acquisition_from_upload(
            reference=GeneralAcquisitionReference(
                source="protocol_emulator",
                recording_id=f"simulated-{digest}",
                alignment=alignment,
            ),
            upload=upload,
            analysis=analysis,
        )
        evidence.append(
            bind_general_evidence(
                case.protocol,
                condition_id=task.condition_id,
                acquisition=acquisition,
            )
        )
    return tuple(evidence)
