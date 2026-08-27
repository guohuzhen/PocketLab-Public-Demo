from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from pocketlab.analyzers import analyze_sensor_recording
from pocketlab.general_exploration_models import (
    GeneralAcquisitionSource,
    GeneralExperimentProtocol,
    StrictFrozenModel,
)
from pocketlab.public_replay_dataset import (
    get_public_replay_dataset,
    read_public_replay_recording,
    verify_public_source_files,
)
from pocketlab.sensor_models import (
    AnalysisMetric,
    SensorAnalysis,
    SensorKind,
    SensorRecordingUpload,
)
from pocketlab.store import SessionStore

_IDENTIFIER = r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$"
_OPAQUE_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
_SHA256 = r"^[0-9a-f]{64}$"

PhysicalProvenanceSource = Literal[
    "phyphox_remote",
    "phone_upload",
    "file_import",
    "public_replay",
    "test_fixture",
]
EmulatorScenario = Literal[
    "transport_timeout",
    "malformed_payload",
    "missing_sensor",
    "alignment_failure",
]


class AcquisitionAlignmentAttestation(StrictFrozenModel):
    capture_group_id: str = Field(pattern=_OPAQUE_ID, max_length=80)
    clock_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    maximum_alignment_error_ms: float = Field(ge=0, le=1000)
    method: Literal[
        "shared_monotonic_clock",
        "server_timestamp_correlation",
        "hardware_trigger",
    ]


class GeneralAcquisitionReference(StrictFrozenModel):
    source: Literal[
        "phyphox_live",
        "phone_upload",
        "public_replay",
        "protocol_emulator",
    ]
    recording_id: str = Field(pattern=_OPAQUE_ID, max_length=100)
    dataset_id: str | None = Field(default=None, pattern=_IDENTIFIER, max_length=100)
    public_match_id: str | None = Field(default=None, pattern=_IDENTIFIER, max_length=80)
    alignment: AcquisitionAlignmentAttestation | None = None

    @model_validator(mode="after")
    def public_reference_is_complete(self) -> Self:
        if self.source == "public_replay":
            if self.dataset_id is None or self.public_match_id is None:
                raise ValueError("public replay references require dataset and semantic match IDs")
        elif self.dataset_id is not None or self.public_match_id is not None:
            raise ValueError("public replay metadata is only valid for public references")
        return self


class GeneralAcquisitionLineage(StrictFrozenModel):
    source: Literal[
        "phyphox_live",
        "phone_upload",
        "public_replay",
        "protocol_emulator",
    ]
    provenance_source: PhysicalProvenanceSource
    recording_id: str = Field(pattern=_OPAQUE_ID, max_length=100)
    content_sha256: str = Field(pattern=_SHA256)
    alignment: AcquisitionAlignmentAttestation | None = None
    public_dataset_id: str | None = Field(default=None, pattern=_IDENTIFIER, max_length=100)
    public_recording_id: str | None = Field(default=None, pattern=_IDENTIFIER, max_length=80)
    public_manifest_sha256: str | None = Field(default=None, pattern=_SHA256)
    public_match_id: str | None = Field(default=None, pattern=_IDENTIFIER, max_length=80)
    user_phone_evidence_candidate: bool
    gate_c_passed: Literal[False] = False
    physical_evidence: bool = True
    simulated: bool = False

    @model_validator(mode="after")
    def source_identity_is_consistent(self) -> Self:
        public_values = (
            self.public_dataset_id,
            self.public_recording_id,
            self.public_manifest_sha256,
            self.public_match_id,
        )
        if self.source == "public_replay":
            if self.provenance_source != "public_replay" or any(
                item is None for item in public_values
            ):
                raise ValueError("public replay lineage must retain complete attestation")
            if self.user_phone_evidence_candidate:
                raise ValueError("public replay cannot count as user phone evidence")
        elif any(item is not None for item in public_values):
            raise ValueError("public lineage fields are only valid for public replay")
        if self.source == "protocol_emulator":
            if (
                self.provenance_source != "test_fixture"
                or self.user_phone_evidence_candidate
                or self.physical_evidence
                or not self.simulated
            ):
                raise ValueError("protocol emulator lineage must remain explicitly simulated")
        elif not self.physical_evidence or self.simulated:
            raise ValueError("non-emulator lineage must remain physical and non-simulated")
        if self.provenance_source == "phyphox_remote":
            if self.source != "phyphox_live" or not self.user_phone_evidence_candidate:
                raise ValueError("phyphox provenance must remain live phone evidence")
        elif self.provenance_source == "phone_upload":
            if self.source != "phone_upload" or not self.user_phone_evidence_candidate:
                raise ValueError("phone uploads must retain their phone evidence candidate flag")
        elif self.provenance_source == "file_import" and (
            self.source != "phone_upload" or self.user_phone_evidence_candidate
        ):
            raise ValueError("generic file imports cannot self-claim phone evidence")
        elif self.provenance_source == "test_fixture" and self.source != "protocol_emulator":
            raise ValueError("test fixtures may only enter the simulated rehearsal path")
        return self


class GeneralPhysicalAcquisition(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    lineage: GeneralAcquisitionLineage
    sensor: SensorKind
    upload: SensorRecordingUpload
    analysis: SensorAnalysis

    @model_validator(mode="after")
    def recording_and_analysis_are_consistent(self) -> Self:
        if self.sensor == "bluetooth":
            raise ValueError("Bluetooth has no general physical acquisition contract")
        if self.upload.sensor != self.sensor or self.analysis.sensor != self.sensor:
            raise ValueError("acquisition sensor, upload and analysis must match")
        if self.analysis.analyzer_id == "" or self.analysis.analyzer_version == "":
            raise ValueError("physical acquisitions require analyzer identity")
        expected_sha = _acquisition_content_sha256(self.upload, self.analysis)
        if self.lineage.content_sha256 != expected_sha:
            raise ValueError("acquisition content SHA does not match upload and analysis")
        return self


class GeneralEvidenceEnvelope(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    evidence_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    protocol_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    protocol_draft_sha256: str = Field(pattern=_SHA256)
    condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    sensor: SensorKind
    role: Literal["primary", "supporting", "control"]
    metric: AnalysisMetric
    analysis: SensorAnalysis
    lineage: GeneralAcquisitionLineage
    quality: Literal["low", "medium", "high"]
    valid: bool
    rejection_reasons: tuple[str, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def evidence_state_is_consistent(self) -> Self:
        if self.analysis.sensor != self.sensor:
            raise ValueError("evidence must retain a matching analysis")
        if self.lineage.simulated != (self.lineage.source == "protocol_emulator"):
            raise ValueError("evidence simulation flag must match its source")
        if self.quality != self.analysis.confidence:
            raise ValueError("evidence quality must snapshot analysis confidence")
        if self.valid and self.rejection_reasons:
            raise ValueError("valid evidence cannot retain rejection reasons")
        if not self.valid and not self.rejection_reasons:
            raise ValueError("invalid evidence requires rejection reasons")
        if len(self.rejection_reasons) != len(set(self.rejection_reasons)):
            raise ValueError("evidence rejection reasons must be unique")
        return self


class GeneralConditionEvidenceGroup(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    protocol_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    condition_id: str = Field(pattern=_IDENTIFIER, max_length=64)
    evidence: tuple[GeneralEvidenceEnvelope, ...] = Field(default=(), max_length=9)
    required_sensors: tuple[SensorKind, ...] = Field(min_length=1, max_length=8)
    missing_sensors: tuple[SensorKind, ...] = Field(default=(), max_length=8)
    alignment_status: Literal["not_required", "verified", "unverified"]
    complete: bool
    valid: bool
    blocker_codes: tuple[str, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def group_state_is_consistent(self) -> Self:
        evidence_sensors = [item.sensor for item in self.evidence]
        if len(evidence_sensors) != len(set(evidence_sensors)):
            raise ValueError("condition evidence cannot duplicate a sensor")
        if set(evidence_sensors).intersection(self.missing_sensors):
            raise ValueError("present evidence cannot also be marked missing")
        expected_complete = not self.missing_sensors and self.alignment_status != "unverified"
        if self.complete != expected_complete:
            raise ValueError("condition completeness does not match sensor/alignment coverage")
        expected_valid = self.complete and all(item.valid for item in self.evidence)
        if self.valid != expected_valid:
            raise ValueError("condition validity does not match evidence validity")
        if self.valid and self.blocker_codes:
            raise ValueError("valid condition groups cannot retain blockers")
        if not self.valid and not self.blocker_codes:
            raise ValueError("invalid condition groups require blockers")
        return self


class ProtocolEmulatorTrace(StrictFrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    trace_id: str = Field(pattern=_IDENTIFIER, max_length=80)
    scenario: EmulatorScenario
    events: tuple[str, ...] = Field(min_length=1, max_length=16)
    physical_evidence: Literal[False] = False
    can_bind_as_evidence: Literal[False] = False
    user_phone_evidence_candidate: Literal[False] = False
    gate_c_passed: Literal[False] = False


class PhysicalAcquisitionSource(Protocol):
    def load(self, reference: GeneralAcquisitionReference) -> GeneralPhysicalAcquisition: ...


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _acquisition_content_sha256(
    upload: SensorRecordingUpload,
    analysis: SensorAnalysis,
) -> str:
    return _canonical_sha256(
        {
            "upload": upload.model_dump(mode="json"),
            "analysis": analysis.model_dump(mode="json"),
        }
    )


def _manifest_sha256(manifest: object) -> str:
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_from_provenance(source: str) -> GeneralAcquisitionSource:
    if source == "phyphox_remote":
        return "phyphox_live"
    if source in {"phone_upload", "file_import"}:
        return "phone_upload"
    if source == "public_replay":
        return "public_replay"
    if source == "test_fixture":
        return "protocol_emulator"
    raise ValueError("test fixtures and unknown provenance cannot become physical evidence")


def _physical_acquisition(
    *,
    reference: GeneralAcquisitionReference,
    upload: SensorRecordingUpload,
    analysis: SensorAnalysis,
    public_manifest_sha256: str | None = None,
) -> GeneralPhysicalAcquisition:
    upload = SensorRecordingUpload.model_validate(upload.model_dump(mode="python"))
    analysis = SensorAnalysis.model_validate(analysis.model_dump(mode="python"))
    if upload.provenance.source == "test_fixture" and reference.source != "protocol_emulator":
        raise ValueError("test fixtures cannot become physical evidence")
    source = _source_from_provenance(upload.provenance.source)
    if source != reference.source:
        raise ValueError("acquisition reference source does not match recording provenance")
    provenance_source: PhysicalProvenanceSource = upload.provenance.source  # type: ignore[assignment]
    public = source == "public_replay"
    if public and (
        upload.provenance.public_dataset_id != reference.dataset_id
        or upload.provenance.public_recording_id != reference.recording_id
        or public_manifest_sha256 is None
    ):
        raise ValueError("public replay reference does not match validated provenance")
    provenance_alignment = None
    provenance = upload.provenance
    if provenance.capture_group_id is not None:
        provenance_alignment = AcquisitionAlignmentAttestation(
            capture_group_id=provenance.capture_group_id,
            clock_id=provenance.clock_id,
            maximum_alignment_error_ms=provenance.maximum_alignment_error_ms,
            method=provenance.alignment_method,
        )
    if (
        reference.alignment is not None
        and provenance_alignment is not None
        and reference.alignment != provenance_alignment
    ):
        raise ValueError("acquisition reference conflicts with recording alignment provenance")
    lineage = GeneralAcquisitionLineage(
        source=source,
        provenance_source=provenance_source,
        recording_id=reference.recording_id,
        content_sha256=_acquisition_content_sha256(upload, analysis),
        alignment=reference.alignment or provenance_alignment,
        public_dataset_id=upload.provenance.public_dataset_id if public else None,
        public_recording_id=upload.provenance.public_recording_id if public else None,
        public_manifest_sha256=public_manifest_sha256 if public else None,
        public_match_id=reference.public_match_id if public else None,
        user_phone_evidence_candidate=provenance_source in {"phyphox_remote", "phone_upload"},
        physical_evidence=source != "protocol_emulator",
        simulated=source == "protocol_emulator",
    )
    return GeneralPhysicalAcquisition(
        lineage=lineage,
        sensor=upload.sensor,
        upload=upload,
        analysis=analysis,
    )


def analyzed_acquisition_from_upload(
    *,
    reference: GeneralAcquisitionReference,
    upload: SensorRecordingUpload,
    analysis: SensorAnalysis,
) -> GeneralPhysicalAcquisition:
    """Build one validated acquisition without granting extra source authority."""

    return _physical_acquisition(
        reference=reference,
        upload=upload,
        analysis=analysis,
    )


@dataclass(frozen=True)
class StoredRecordingAcquisitionSource:
    store: SessionStore

    def load(self, reference: GeneralAcquisitionReference) -> GeneralPhysicalAcquisition:
        if reference.source == "public_replay":
            raise ValueError("stored public records require the verified public replay adapter")
        stored = self.store.get_sensor_recording(reference.recording_id)
        return _physical_acquisition(
            reference=reference,
            upload=stored.upload,
            analysis=stored.analysis,
        )


@dataclass(frozen=True)
class PublicReplayAcquisitionSource:
    root: Path

    def load(self, reference: GeneralAcquisitionReference) -> GeneralPhysicalAcquisition:
        if reference.source != "public_replay" or reference.dataset_id is None:
            raise ValueError("public replay adapter requires a complete public reference")
        pack_dir, manifest = get_public_replay_dataset(self.root, reference.dataset_id)
        recording = next(
            (item for item in manifest.recordings if item.recording_id == reference.recording_id),
            None,
        )
        if recording is None:
            raise KeyError(f"Unknown public replay recording: {reference.recording_id}")
        verify_public_source_files(pack_dir, manifest)
        upload = read_public_replay_recording(pack_dir, manifest, recording)
        analysis = analyze_sensor_recording(upload)
        if analysis.model_dump(mode="json") != recording.reference_analysis.model_dump(mode="json"):
            raise ValueError("public replay analysis differs from its frozen reference")
        return _physical_acquisition(
            reference=reference,
            upload=upload,
            analysis=analysis,
            public_manifest_sha256=_manifest_sha256(manifest),
        )


@dataclass(frozen=True)
class ProtocolEmulatorSource:
    def run(self, scenario: EmulatorScenario) -> ProtocolEmulatorTrace:
        event_map: dict[EmulatorScenario, tuple[str, ...]] = {
            "transport_timeout": ("request-started", "timeout", "fallback-required"),
            "malformed_payload": ("payload-received", "schema-rejected"),
            "missing_sensor": ("capability-checked", "required-sensor-missing"),
            "alignment_failure": ("records-loaded", "clock-attestation-mismatch"),
        }
        return ProtocolEmulatorTrace(
            trace_id=f"emulator-{scenario.replace('_', '-')}",
            scenario=scenario,
            events=event_map[scenario],
        )


def bind_general_evidence(
    protocol: GeneralExperimentProtocol,
    *,
    condition_id: str,
    acquisition: GeneralPhysicalAcquisition,
) -> GeneralEvidenceEnvelope:
    protocol = GeneralExperimentProtocol.model_validate(protocol.model_dump(mode="python"))
    acquisition = GeneralPhysicalAcquisition.model_validate(acquisition.model_dump(mode="python"))
    if condition_id not in {item.condition_id for item in protocol.conditions}:
        raise ValueError("evidence condition is outside the protocol")
    requirement = next(
        (item for item in protocol.sensors if item.sensor == acquisition.sensor),
        None,
    )
    if requirement is None:
        raise ValueError("acquisition sensor is outside the protocol")
    if requirement.analyzer_id != acquisition.analysis.analyzer_id:
        raise ValueError("acquisition analyzer does not match the protocol")
    if acquisition.lineage.source not in protocol.selected_sources:
        raise ValueError("acquisition source is not authorized by the protocol")
    if (
        acquisition.lineage.source == "public_replay"
        and acquisition.lineage.public_match_id != protocol.public_replay_match_id
    ):
        raise ValueError("public acquisition is not bound to the protocol semantic match")
    metric = next(
        (
            item
            for item in acquisition.analysis.metrics
            if item.key == requirement.metric_key and item.unit == requirement.metric_unit
        ),
        None,
    )
    if metric is None:
        raise ValueError("analysis does not contain the protocol metric and unit")
    valid = acquisition.analysis.confidence in {"medium", "high"}
    rejection_reasons = () if valid else ("analysis-confidence-low",)
    evidence_hash = _canonical_sha256(
        {
            "protocol_id": protocol.protocol_id,
            "draft_sha256": protocol.draft_sha256,
            "condition_id": condition_id,
            "sensor": acquisition.sensor,
            "content_sha256": acquisition.lineage.content_sha256,
            "metric_key": metric.key,
            "metric_unit": metric.unit,
        }
    )
    return GeneralEvidenceEnvelope(
        evidence_id=f"general-evidence-{evidence_hash[:16]}",
        protocol_id=protocol.protocol_id,
        protocol_draft_sha256=protocol.draft_sha256,
        condition_id=condition_id,
        sensor=acquisition.sensor,
        role=requirement.role,
        metric=metric,
        analysis=acquisition.analysis,
        lineage=acquisition.lineage,
        quality=acquisition.analysis.confidence,
        valid=valid,
        rejection_reasons=rejection_reasons,
    )


def build_condition_evidence_group(
    protocol: GeneralExperimentProtocol,
    *,
    condition_id: str,
    evidence: list[GeneralEvidenceEnvelope],
) -> GeneralConditionEvidenceGroup:
    protocol = GeneralExperimentProtocol.model_validate(protocol.model_dump(mode="python"))
    evidence = [
        GeneralEvidenceEnvelope.model_validate(item.model_dump(mode="python")) for item in evidence
    ]
    if condition_id not in {item.condition_id for item in protocol.conditions}:
        raise ValueError("condition evidence group is outside the protocol")
    for item in evidence:
        if (
            item.protocol_id != protocol.protocol_id
            or item.protocol_draft_sha256 != protocol.draft_sha256
            or item.condition_id != condition_id
        ):
            raise ValueError("condition evidence group contains a foreign evidence reference")
    evidence_sensors = [item.sensor for item in evidence]
    if len(evidence_sensors) != len(set(evidence_sensors)):
        raise ValueError("condition evidence group cannot duplicate a sensor")
    required_sensors = tuple(
        item.sensor
        for item in protocol.sensors
        if item.sensor != "bluetooth" and item.activation == "required"
    )
    missing_sensors = tuple(
        sensor for sensor in required_sensors if sensor not in set(evidence_sensors)
    )
    blockers: list[str] = []
    if missing_sensors:
        blockers.append("required-sensor-evidence-missing")
    if any(not item.valid for item in evidence):
        blockers.append("low-quality-evidence-present")

    alignment_status: Literal["not_required", "verified", "unverified"] = "not_required"
    if protocol.alignment == "simultaneous" and len(required_sensors) > 1:
        attestations = [item.lineage.alignment for item in evidence]
        alignment_status = "verified"
        if len(attestations) != len(required_sensors) or any(item is None for item in attestations):
            alignment_status = "unverified"
        else:
            concrete = [item for item in attestations if item is not None]
            attestation_values = {
                (
                    item.capture_group_id,
                    item.clock_id,
                    item.maximum_alignment_error_ms,
                    item.method,
                )
                for item in concrete
            }
            if (
                len(attestation_values) != 1
                or any(item.maximum_alignment_error_ms > 250 for item in concrete)
            ):
                alignment_status = "unverified"
        if alignment_status == "unverified":
            blockers.append("simultaneous-alignment-not-verified")

    complete = not missing_sensors and alignment_status != "unverified"
    valid = complete and all(item.valid for item in evidence)
    return GeneralConditionEvidenceGroup(
        protocol_id=protocol.protocol_id,
        condition_id=condition_id,
        evidence=tuple(evidence),
        required_sensors=required_sensors,
        missing_sensors=missing_sensors,
        alignment_status=alignment_status,
        complete=complete,
        valid=valid,
        blocker_codes=tuple(dict.fromkeys(blockers)),
    )
