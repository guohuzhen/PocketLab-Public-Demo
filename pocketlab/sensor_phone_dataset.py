from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
import shutil
import sqlite3
import tempfile
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from pocketlab.analyzers.registry import analyze_sensor_recording
from pocketlab.investigation_models import InvestigationCase, SensorAnalysisSnapshot
from pocketlab.sensor_models import (
    SensorAnalysis,
    SensorChannelDefinition,
    SensorProvenance,
    SensorRecordingUpload,
    SensorSample,
)

SENSOR_PHONE_SCHEMA_VERSION = "2.0"
SENSOR_PHONE_MANIFEST = "manifest.json"
LIGHT_SAMPLE_COLUMNS = ("timestamp_ms", "illuminance_lx")
LIGHT_ANALYZER_ID = "pocketlab.light.v2"
LIGHT_ANALYZER_VERSION = "2.0.0"
MAX_RECORDING_BYTES = 64 * 1024 * 1024
MAX_SAMPLE_ROWS = 120_000
MAX_CSV_LINE_LENGTH = 4096

DataFlag = Literal[
    "insufficient_measurement_conditions",
    "insufficient_valid_replicates",
    "invalid_evidence_present",
    "protocol_mismatch",
]
LoopFlag = Literal[
    "missing_background_pair",
    "insufficient_loop_conditions",
    "insufficient_loop_replicates",
    "original_not_conclusive",
    "no_agent_planner_trace",
]

_PRIVATE_TEXT_PATTERNS = (
    re.compile(r"https?://|www\.", re.IGNORECASE),
    re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?!\d)"),
    re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE),
    re.compile(r"(?:[A-Z]:\\|\\\\)", re.IGNORECASE),
    re.compile(r"(?:sk-|api[_ -]?key|bearer\s+)", re.IGNORECASE),
    re.compile(r"(?:case|session|task|evidence|run)[_-][A-Za-z0-9._-]{6,}"),
    re.compile(r"\b[0-9a-f]{32,}\b", re.IGNORECASE),
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


def _validate_public_text(value: str, *, field_name: str) -> str:
    if any(pattern.search(value) for pattern in _PRIVATE_TEXT_PATTERNS):
        raise ValueError(f"{field_name} contains a private identifier, address, path or secret")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SensorPhoneProvenance(_StrictModel):
    collection_kind: Literal["self_collected_phone", "test_fixture"]
    capture_app: str = Field(min_length=2, max_length=80)
    collected_date: date
    exported_date: date
    consent_basis: str = Field(min_length=10, max_length=240)
    license_note: str = Field(min_length=3, max_length=240)
    original_recording_ids_removed: Literal[True] = True
    original_investigation_id_removed: Literal[True] = True
    user_identifiers_removed: Literal[True] = True
    network_addresses_removed: Literal[True] = True
    free_text_removed: Literal[True] = True
    exact_timestamps_removed: Literal[True] = True

    @field_validator("capture_app", "consent_basis", "license_note")
    @classmethod
    def public_text_only(cls, value: str, info: Any) -> str:
        return _validate_public_text(value, field_name=info.field_name)


class SensorPhoneScenario(_StrictModel):
    scene_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=64)
    scene_label: str = Field(min_length=3, max_length=120)
    environment: str = Field(min_length=3, max_length=240)
    light_source: str = Field(min_length=2, max_length=160)
    phone_placement: str = Field(min_length=3, max_length=160)
    phone_orientation: str = Field(min_length=3, max_length=160)
    controlled_variables: list[str] = Field(min_length=1, max_length=16)

    @field_validator(
        "scene_label",
        "environment",
        "light_source",
        "phone_placement",
        "phone_orientation",
    )
    @classmethod
    def public_text_only(cls, value: str, info: Any) -> str:
        return _validate_public_text(value, field_name=info.field_name)

    @field_validator("controlled_variables")
    @classmethod
    def public_control_text_only(cls, values: list[str]) -> list[str]:
        cleaned = [
            _validate_public_text(value, field_name="controlled_variables") for value in values
        ]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("controlled_variables must not contain duplicates")
        return cleaned


class SensorPhoneCondition(_StrictModel):
    condition_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=64)
    role: Literal["background", "measurement"]
    label: str = Field(min_length=2, max_length=100)
    distance_m: float | None = Field(default=None, ge=0.1, le=4.0)

    @field_validator("label")
    @classmethod
    def public_label_only(cls, value: str) -> str:
        return _validate_public_text(value, field_name="condition.label")

    @model_validator(mode="after")
    def distance_matches_role(self) -> Self:
        if self.role == "background" and self.distance_m is not None:
            raise ValueError("background conditions cannot define distance_m")
        if self.role == "measurement" and self.distance_m is None:
            raise ValueError("measurement conditions require distance_m")
        return self


class SensorPhoneRecording(_StrictModel):
    recording_id: str = Field(pattern=r"^recording-[0-9]{3}$")
    file: str = Field(min_length=5, max_length=180)
    condition_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=64)
    replicate_index: int = Field(ge=1, le=32)
    task_sequence: int = Field(ge=1, le=32)
    task_role: Literal["background", "condition", "replication", "correction", "exploration"]
    selection_source: Literal["protocol", "deterministic", "agent", "fallback"]
    selection_reason_code: str = Field(
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$",
        max_length=64,
    )
    capture_source: Literal["phyphox_remote", "phone_upload"]
    replay_source: Literal["file_import"] = "file_import"
    phone_alias: Literal["deidentified-phone-light-sensor"] = (
        "deidentified-phone-light-sensor"
    )
    phyphox_experiment: Literal["Light"] = "Light"
    config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sample_count: int = Field(ge=2, le=MAX_SAMPLE_ROWS)
    collected_date: date
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_analysis: SensorAnalysis
    evidence_valid: StrictBool
    rejection_reasons: list[str] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_recording_contract(self) -> Self:
        path = Path(self.file)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("recording file must be a safe pack-relative path")
        if not (self.file.endswith(".csv") or self.file.endswith(".csv.gz")):
            raise ValueError("recording file must end in .csv or .csv.gz")
        if self.reference_analysis.sensor != "light":
            raise ValueError("Light dataset references must contain light analysis")
        if (
            self.reference_analysis.analyzer_id != LIGHT_ANALYZER_ID
            or self.reference_analysis.analyzer_version != LIGHT_ANALYZER_VERSION
        ):
            raise ValueError("recording reference uses the wrong Light analyzer identity")
        if self.reference_analysis.sample_count != self.sample_count:
            raise ValueError("reference analysis sample_count must match recording")
        if self.evidence_valid and self.rejection_reasons:
            raise ValueError("valid evidence cannot contain rejection reasons")
        if not self.evidence_valid and not self.rejection_reasons:
            raise ValueError("invalid evidence requires rejection reasons")
        for reason in self.rejection_reasons:
            _validate_public_text(reason, field_name="rejection_reasons")
        return self


class SensorPhoneExpectation(_StrictModel):
    disposition: Literal["pilot_only", "gate_c_candidate"]
    min_measurement_conditions: Literal[2] = 2
    min_valid_replicates_per_condition: Literal[3] = 3
    expected_data_flags: list[DataFlag] = Field(default_factory=list)
    expected_loop_flags: list[LoopFlag] = Field(default_factory=list)

    @model_validator(mode="after")
    def flags_are_unique(self) -> Self:
        if len(self.expected_data_flags) != len(set(self.expected_data_flags)):
            raise ValueError("expected_data_flags must not contain duplicates")
        if len(self.expected_loop_flags) != len(set(self.expected_loop_flags)):
            raise ValueError("expected_loop_flags must not contain duplicates")
        return self


class SensorPhoneDatasetManifest(_StrictModel):
    schema_version: Literal["2.0"] = SENSOR_PHONE_SCHEMA_VERSION
    dataset_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=80)
    title: str = Field(min_length=3, max_length=140)
    description: str = Field(min_length=10, max_length=500)
    sensor: Literal["light"] = "light"
    analyzer_id: Literal["pocketlab.light.v2"] = LIGHT_ANALYZER_ID
    analyzer_version: Literal["2.0.0"] = LIGHT_ANALYZER_VERSION
    protocol_id: Literal["light-distance-law.v1"] = "light-distance-law.v1"
    protocol_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    channels: dict[Literal["illuminance"], Literal["lx"]]
    provenance: SensorPhoneProvenance
    scenario: SensorPhoneScenario
    conditions: list[SensorPhoneCondition] = Field(min_length=1, max_length=20)
    recordings: list[SensorPhoneRecording] = Field(min_length=1, max_length=64)
    expectation: SensorPhoneExpectation
    original_status: Literal[
        "completed_with_conclusion",
        "completed_inconclusive",
        "collecting",
        "cancelled",
    ]
    original_planner_decision_count: int = Field(ge=0, le=32)
    market_validated: Literal[False] = False
    agent_ready: Literal[False] = False

    @field_validator("title", "description")
    @classmethod
    def public_text_only(cls, value: str, info: Any) -> str:
        return _validate_public_text(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_manifest_graph(self) -> Self:
        if self.channels != {"illuminance": "lx"}:
            raise ValueError("Light v2 channels must be exactly illuminance/lx")
        condition_ids = [item.condition_id for item in self.conditions]
        recording_ids = [item.recording_id for item in self.recordings]
        files = [item.file for item in self.recordings]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("condition_id must be unique")
        if len(recording_ids) != len(set(recording_ids)):
            raise ValueError("recording_id must be unique")
        if len(files) != len(set(files)):
            raise ValueError("recording file must be unique")
        known_conditions = set(condition_ids)
        if any(item.condition_id not in known_conditions for item in self.recordings):
            raise ValueError("recording references an unknown condition")
        slots = [(item.condition_id, item.replicate_index) for item in self.recordings]
        if len(slots) != len(set(slots)):
            raise ValueError("condition replicate slots must be unique")
        if len({item.task_sequence for item in self.recordings}) != len(self.recordings):
            raise ValueError("task_sequence must be unique")
        return self


def _resolve_recording_path(pack_dir: Path, relative_path: str) -> Path:
    root = pack_dir.resolve()
    resolved = (root / relative_path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("recording file escapes the dataset pack")
    current = root
    for part in Path(relative_path).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("recording paths cannot contain symbolic links")
    return resolved


def load_sensor_phone_dataset(pack_dir: Path) -> SensorPhoneDatasetManifest:
    manifest_path = pack_dir / SENSOR_PHONE_MANIFEST
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return SensorPhoneDatasetManifest.model_validate(payload)


def read_sensor_phone_recording(
    pack_dir: Path,
    recording: SensorPhoneRecording,
) -> SensorRecordingUpload:
    path = _resolve_recording_path(pack_dir, recording.file)
    if not path.is_file():
        raise FileNotFoundError(f"missing recording file: {recording.file}")
    if path.stat().st_size > MAX_RECORDING_BYTES:
        raise ValueError("recording file exceeds the 64 MiB compressed/raw limit")
    observed_sha = _sha256(path)
    if observed_sha != recording.sha256:
        raise ValueError(
            f"recording checksum mismatch: expected={recording.sha256}, observed={observed_sha}"
        )

    raw_handle = path.open("rb")
    binary_handle: io.BufferedReader | gzip.GzipFile
    binary_handle = gzip.GzipFile(fileobj=raw_handle, mode="rb") if path.suffix == ".gz" else raw_handle
    samples: list[SensorSample] = []
    try:
        with io.TextIOWrapper(binary_handle, encoding="utf-8", newline="") as text_handle:
            for line_number, line in enumerate(text_handle, start=1):
                if len(line) > MAX_CSV_LINE_LENGTH:
                    raise ValueError(f"CSV line {line_number} exceeds the length limit")
                row = next(csv.reader([line]))
                if line_number == 1:
                    if tuple(row) != LIGHT_SAMPLE_COLUMNS:
                        raise ValueError("Light CSV columns must be timestamp_ms,illuminance_lx")
                    continue
                if len(row) != 2:
                    raise ValueError(f"CSV line {line_number} must contain exactly two values")
                try:
                    timestamp_ms, illuminance_lx = (float(value) for value in row)
                except ValueError as exc:
                    raise ValueError(f"CSV line {line_number} contains a non-numeric value") from exc
                if not math.isfinite(timestamp_ms) or not math.isfinite(illuminance_lx):
                    raise ValueError(f"CSV line {line_number} contains a non-finite value")
                if illuminance_lx < 0:
                    raise ValueError(f"CSV line {line_number} contains negative illuminance")
                samples.append(
                    SensorSample(
                        timestamp_ms=timestamp_ms,
                        values={"illuminance": illuminance_lx},
                    )
                )
                if len(samples) > MAX_SAMPLE_ROWS:
                    raise ValueError("recording exceeds the sample-row limit")
    finally:
        if not raw_handle.closed:
            raw_handle.close()
    if not samples:
        raise ValueError("recording contains no samples")
    if abs(samples[0].timestamp_ms) > 1e-9:
        raise ValueError("dataset timestamps must be rebased so the first sample is 0 ms")
    if len(samples) != recording.sample_count:
        raise ValueError(
            f"sample_count mismatch: manifest={recording.sample_count}, file={len(samples)}"
        )
    return SensorRecordingUpload(
        label=f"dataset-{recording.recording_id}",
        device=recording.phone_alias,
        sensor="light",
        notes="Deidentified Sensor Phone Dataset v2 replay.",
        channels={
            "illuminance": SensorChannelDefinition(
                unit="lx",
                description="Ambient illuminance reported by the phone light sensor.",
            )
        },
        samples=samples,
        provenance=SensorProvenance(
            source="file_import",
            experiment_title=recording.phyphox_experiment,
            config_sha256=recording.config_sha256,
            channel_mapping={"illuminance": "illuminance_lx"},
            privacy_acknowledged=True,
        ),
    )


def _analysis_matches(observed: SensorAnalysis, reference: SensorAnalysis) -> bool:
    return observed.model_dump(mode="json") == reference.model_dump(mode="json")


def _observed_data_flags(manifest: SensorPhoneDatasetManifest) -> list[DataFlag]:
    flags: set[DataFlag] = set()
    measurement_conditions = {
        item.condition_id for item in manifest.conditions if item.role == "measurement"
    }
    valid_counts = Counter(
        item.condition_id
        for item in manifest.recordings
        if item.evidence_valid and item.condition_id in measurement_conditions
    )
    if len(measurement_conditions) < manifest.expectation.min_measurement_conditions:
        flags.add("insufficient_measurement_conditions")
    if any(
        valid_counts[condition_id] < manifest.expectation.min_valid_replicates_per_condition
        for condition_id in measurement_conditions
    ):
        flags.add("insufficient_valid_replicates")
    if any(not item.evidence_valid for item in manifest.recordings):
        flags.add("invalid_evidence_present")
    config_hashes = {item.config_sha256 for item in manifest.recordings if item.config_sha256}
    if len(config_hashes) > 1:
        flags.add("protocol_mismatch")
    return sorted(flags)


def _observed_loop_flags(manifest: SensorPhoneDatasetManifest) -> list[LoopFlag]:
    flags: set[LoopFlag] = set()
    valid_backgrounds = sum(
        item.evidence_valid
        for item in manifest.recordings
        if next(
            condition
            for condition in manifest.conditions
            if condition.condition_id == item.condition_id
        ).role
        == "background"
    )
    if valid_backgrounds < 2:
        flags.add("missing_background_pair")
    measurement_conditions = {
        item.condition_id for item in manifest.conditions if item.role == "measurement"
    }
    valid_counts = Counter(
        item.condition_id
        for item in manifest.recordings
        if item.evidence_valid and item.condition_id in measurement_conditions
    )
    if len(measurement_conditions) < 4:
        flags.add("insufficient_loop_conditions")
    if any(valid_counts[condition_id] < 2 for condition_id in measurement_conditions):
        flags.add("insufficient_loop_replicates")
    if manifest.original_status != "completed_with_conclusion":
        flags.add("original_not_conclusive")
    if manifest.original_planner_decision_count < 1 or not any(
        item.selection_source == "agent" for item in manifest.recordings
    ):
        flags.add("no_agent_planner_trace")
    return sorted(flags)


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(item["passed"] for item in checks)
    return {
        "passed": passed,
        "total": len(checks),
        "score_percent": round(100.0 * passed / max(len(checks), 1), 2),
    }


def evaluate_sensor_phone_dataset(
    pack_dir: Path,
    *,
    replay_repeat: int = 3,
) -> dict[str, Any]:
    if replay_repeat < 1:
        raise ValueError("replay_repeat must be at least 1")
    manifest = load_sensor_phone_dataset(pack_dir)
    recording_results: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for recording in manifest.recordings:
        upload = read_sensor_phone_recording(pack_dir, recording)
        repeated = [analyze_sensor_recording(upload) for _ in range(replay_repeat)]
        deterministic = all(
            item.model_dump(mode="json") == repeated[0].model_dump(mode="json")
            for item in repeated[1:]
        )
        reference_matches = _analysis_matches(repeated[0], recording.reference_analysis)
        recording_checks = [
            _check("file_integrity", True, f"sha256={recording.sha256}"),
            _check("deterministic_replay", deterministic, f"repeat={replay_repeat}"),
            _check(
                "reference_analysis_regression",
                reference_matches,
                f"analyzer={LIGHT_ANALYZER_ID}@{LIGHT_ANALYZER_VERSION}",
            ),
        ]
        checks.extend(recording_checks)
        recording_results.append(
            {
                "recording_id": recording.recording_id,
                "condition_id": recording.condition_id,
                "replicate_index": recording.replicate_index,
                "evidence_valid": recording.evidence_valid,
                "analysis": repeated[0].model_dump(mode="json"),
                "checks": recording_checks,
            }
        )

    data_flags = _observed_data_flags(manifest)
    loop_flags = _observed_loop_flags(manifest)
    data_flags_match = data_flags == sorted(manifest.expectation.expected_data_flags)
    loop_flags_match = loop_flags == sorted(manifest.expectation.expected_loop_flags)
    expected_disposition = "gate_c_candidate" if not data_flags else "pilot_only"
    disposition_matches = expected_disposition == manifest.expectation.disposition
    boundary_checks = [
        _check("expected_data_flags", data_flags_match, json.dumps(data_flags)),
        _check("expected_loop_flags", loop_flags_match, json.dumps(loop_flags)),
        _check("expected_disposition", disposition_matches, expected_disposition),
        _check(
            "truthful_release_boundary",
            not manifest.market_validated and not manifest.agent_ready,
            "market_validated=false, agent_ready=false",
        ),
    ]
    checks.extend(boundary_checks)
    summary = _summary(checks)
    gate_c_candidate = (
        manifest.provenance.collection_kind == "self_collected_phone"
        and expected_disposition == "gate_c_candidate"
        and summary["passed"] == summary["total"]
    )
    return {
        "schema_version": SENSOR_PHONE_SCHEMA_VERSION,
        "dataset_id": manifest.dataset_id,
        "scene_id": manifest.scenario.scene_id,
        "sensor": manifest.sensor,
        "collection_kind": manifest.provenance.collection_kind,
        "recording_count": len(manifest.recordings),
        "valid_phone_recording_count": sum(
            item.evidence_valid for item in manifest.recordings
        ),
        "data_flags": data_flags,
        "loop_flags": loop_flags,
        "gate_c_candidate": gate_c_candidate,
        "loop_replay_candidate": gate_c_candidate and not loop_flags,
        "recordings": recording_results,
        "checks": checks,
        "summary": summary,
        "market_validated": False,
        "agent_ready": False,
    }


def run_sensor_phone_harness(
    pack_dirs: list[Path],
    *,
    replay_repeat: int = 3,
) -> dict[str, Any]:
    if not pack_dirs:
        return {
            "schema_version": SENSOR_PHONE_SCHEMA_VERSION,
            "execution_status": "blocked_missing_evidence",
            "packs": [],
            "real_phone_records": 0,
            "gate_c_qualified_real_phone_records": 0,
            "gate_c": "not_evaluated",
            "gate_e": "not_evaluated",
            "gate_c_passed": False,
            "market_validated": False,
            "agent_ready": False,
        }
    packs = [
        evaluate_sensor_phone_dataset(path, replay_repeat=replay_repeat)
        for path in pack_dirs
    ]
    candidates = [item for item in packs if item["gate_c_candidate"]]
    scenes = {item["scene_id"] for item in candidates}
    real_phone_records = sum(
        item["recording_count"]
        for item in packs
        if item["collection_kind"] == "self_collected_phone"
    )
    qualified_records = sum(item["valid_phone_recording_count"] for item in candidates)
    gate_c_passed = len(candidates) >= 3 and len(scenes) >= 3 and qualified_records >= 18
    all_checks_pass = all(
        pack["summary"]["passed"] == pack["summary"]["total"] for pack in packs
    )
    return {
        "schema_version": SENSOR_PHONE_SCHEMA_VERSION,
        "execution_status": "completed" if all_checks_pass else "failed_validation",
        "packs": packs,
        "pack_count": len(packs),
        "gate_c_candidate_pack_count": len(candidates),
        "distinct_candidate_scenes": len(scenes),
        "real_phone_records": real_phone_records,
        "gate_c_qualified_real_phone_records": qualified_records,
        "gate_c": "pass" if gate_c_passed else "fail",
        "gate_e": "not_evaluated",
        "gate_c_passed": gate_c_passed,
        "market_validated": False,
        "agent_ready": False,
    }


def discover_sensor_phone_packs(root: Path) -> list[Path]:
    packs: list[Path] = []
    if not root.is_dir():
        return packs
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") == SENSOR_PHONE_SCHEMA_VERSION:
            packs.append(manifest_path.parent)
    return packs


def render_sensor_phone_report(result: dict[str, Any]) -> str:
    lines = [
        "# Sensor Phone Dataset v2 Harness",
        "",
        f"- execution_status: `{result['execution_status']}`",
        f"- Gate C: `{result['gate_c']}`",
        f"- Gate E: `{result['gate_e']}`",
        f"- observed real phone records: `{result['real_phone_records']}`",
        (
            "- real phone records accepted for Gate C: "
            f"`{result['gate_c_qualified_real_phone_records']}`"
        ),
        f"- market_validated: `{str(result['market_validated']).lower()}`",
        f"- agent_ready: `{str(result['agent_ready']).lower()}`",
        "",
    ]
    if not result.get("packs"):
        lines.extend(
            [
                "No eligible Sensor Phone Dataset v2 pack was found. The Harness did not ",
                "invoke a model or substitute synthetic evidence.",
            ]
        )
        return "\n".join(lines) + "\n"
    lines.extend(["## Packs", ""])
    for pack in result["packs"]:
        lines.extend(
            [
                (
                    f"- `{pack['dataset_id']}` / scene `{pack['scene_id']}`: "
                    f"{pack['summary']['passed']}/{pack['summary']['total']}, "
                    f"Gate C candidate=`{str(pack['gate_c_candidate']).lower()}`"
                ),
                f"  - data flags: `{', '.join(pack['data_flags']) or 'none'}`",
                f"  - loop flags: `{', '.join(pack['loop_flags']) or 'none'}`",
            ]
        )
    return "\n".join(lines) + "\n"


def write_sensor_phone_results(
    result: dict[str, Any],
    *,
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> None:
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_sensor_phone_report(result), encoding="utf-8")


def _date_only(value: str) -> date:
    return datetime.fromisoformat(value).date()


def _distance_from_evidence(evidence: Any) -> float | None:
    for parameter in evidence.parameters:
        if parameter.key == "distance_m":
            return float(parameter.value)
    return None


def _condition_id(distance_m: float | None) -> str:
    if distance_m is None:
        return "background"
    return f"distance-{round(distance_m * 10_000):05d}"


def _write_light_csv(path: Path, upload: SensorRecordingUpload) -> None:
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed,
        io.TextIOWrapper(
            compressed,
            encoding="utf-8",
            newline="",
            write_through=True,
        ) as text,
    ):
        writer = csv.writer(text, lineterminator="\n")
        writer.writerow(LIGHT_SAMPLE_COLUMNS)
        first_timestamp = upload.samples[0].timestamp_ms
        for sample in upload.samples:
            writer.writerow(
                [
                    format(sample.timestamp_ms - first_timestamp, ".17g"),
                    format(sample.values["illuminance"], ".17g"),
                ]
            )
    path.write_bytes(buffer.getvalue())


def export_light_investigation(
    database_path: Path,
    output_dir: Path,
    *,
    dataset_id: str,
    scene_id: str,
    scene_label: str,
    environment: str,
    light_source: str,
    phone_placement: str,
    phone_orientation: str,
    controlled_variables: list[str],
    consent_basis: str,
    license_note: str,
    investigation_id: str | None = None,
) -> SensorPhoneDatasetManifest:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset pack: {output_dir}")
    database_path = database_path.resolve()
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    connection = sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    try:
        if investigation_id is not None:
            rows = connection.execute(
                "SELECT user_id, case_json, created_at FROM investigation_cases "
                "WHERE investigation_id = ?",
                (investigation_id,),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT user_id, case_json, created_at FROM investigation_cases "
                "ORDER BY updated_at DESC"
            ).fetchall()
        selected: tuple[sqlite3.Row, InvestigationCase] | None = None
        for row in rows:
            case = InvestigationCase.model_validate_json(row["case_json"])
            if case.protocol.protocol_id == "light-distance-law.v1" and case.status in {
                "completed_with_conclusion",
                "completed_inconclusive",
            }:
                selected = (row, case)
                break
        if selected is None:
            raise ValueError("no completed Light investigation is available for export")
        row, case = selected
        tasks = {item.task_id: item for item in case.completed_tasks}
        ordered_evidence = sorted(case.evidence, key=lambda item: tasks[item.task_id].sequence)
        exports: list[tuple[Any, Any, SensorRecordingUpload, SensorAnalysis, str, int]] = []
        condition_attempts: Counter[str] = Counter()
        for evidence in ordered_evidence:
            if evidence.recording.recording_type != "sensor_v2":
                raise ValueError("Light evidence must reference Sensor Recording v2")
            session_row = connection.execute(
                "SELECT upload_json, analysis_json, created_at FROM sessions "
                "WHERE session_id = ? AND user_id = ?",
                (evidence.recording.recording_id, row["user_id"]),
            ).fetchone()
            if session_row is None:
                raise ValueError("an evidence recording is missing or belongs to another user")
            upload = SensorRecordingUpload.model_validate_json(session_row["upload_json"])
            stored_analysis = SensorAnalysis.model_validate_json(session_row["analysis_json"])
            if upload.sensor != "light" or upload.provenance.source != evidence.recording.source:
                raise ValueError("recording sensor/source does not match the evidence snapshot")
            recomputed = analyze_sensor_recording(upload)
            if not _analysis_matches(recomputed, stored_analysis):
                raise ValueError("stored analysis does not match deterministic recomputation")
            snapshot = SensorAnalysisSnapshot.from_sensor_analysis(recomputed)
            if evidence.analysis is not None and evidence.analysis != snapshot:
                raise ValueError("evidence analysis snapshot does not match its recording")
            distance_m = _distance_from_evidence(evidence)
            local_condition = _condition_id(distance_m)
            condition_attempts[local_condition] += 1
            exports.append(
                (
                    evidence,
                    tasks[evidence.task_id],
                    upload,
                    recomputed,
                    session_row["created_at"],
                    condition_attempts[local_condition],
                )
            )
    finally:
        connection.close()

    condition_distances: dict[str, float | None] = {}
    for evidence, *_rest in exports:
        distance_m = _distance_from_evidence(evidence)
        condition_distances[_condition_id(distance_m)] = distance_m
    conditions = [
        SensorPhoneCondition(
            condition_id=condition_id,
            role="background" if distance is None else "measurement",
            label="Background" if distance is None else f"Distance {distance:.4g} m",
            distance_m=distance,
        )
        for condition_id, distance in sorted(
            condition_distances.items(),
            key=lambda item: (-1.0 if item[1] is None else item[1]),
        )
    ]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{dataset_id}-", dir=output_dir.parent) as temp_root:
        temp_pack = Path(temp_root) / dataset_id
        recordings_dir = temp_pack / "recordings"
        recordings_dir.mkdir(parents=True)
        definitions: list[SensorPhoneRecording] = []
        for index, (evidence, task, upload, analysis, created_at, replicate_index) in enumerate(
            exports,
            start=1,
        ):
            recording_id = f"recording-{index:03d}"
            relative_file = f"recordings/{recording_id}.csv.gz"
            recording_path = temp_pack / relative_file
            _write_light_csv(recording_path, upload)
            definitions.append(
                SensorPhoneRecording(
                    recording_id=recording_id,
                    file=relative_file,
                    condition_id=_condition_id(_distance_from_evidence(evidence)),
                    replicate_index=replicate_index,
                    task_sequence=task.sequence,
                    task_role=evidence.role,
                    selection_source=task.selection_source,
                    selection_reason_code=task.selection_reason_code,
                    capture_source=evidence.recording.source,
                    config_sha256=evidence.recording.config_sha256,
                    sample_count=len(upload.samples),
                    collected_date=_date_only(created_at),
                    sha256=_sha256(recording_path),
                    reference_analysis=analysis,
                    evidence_valid=evidence.valid,
                    rejection_reasons=evidence.rejection_reasons,
                )
            )

        provisional = SensorPhoneDatasetManifest(
            dataset_id=dataset_id,
            title=f"PocketLab Light phone pilot: {scene_label}",
            description=(
                "Deidentified, self-collected phone light-sensor evidence exported read-only "
                "for deterministic replay and readiness auditing."
            ),
            protocol_version=case.protocol.protocol_version,
            channels={"illuminance": "lx"},
            provenance=SensorPhoneProvenance(
                collection_kind="self_collected_phone",
                capture_app="phyphox Light with PocketLab remote capture",
                collected_date=min(item.collected_date for item in definitions),
                exported_date=datetime.now(UTC).date(),
                consent_basis=consent_basis,
                license_note=license_note,
            ),
            scenario=SensorPhoneScenario(
                scene_id=scene_id,
                scene_label=scene_label,
                environment=environment,
                light_source=light_source,
                phone_placement=phone_placement,
                phone_orientation=phone_orientation,
                controlled_variables=controlled_variables,
            ),
            conditions=conditions,
            recordings=definitions,
            expectation=SensorPhoneExpectation(
                disposition="pilot_only",
                expected_data_flags=[],
                expected_loop_flags=[],
            ),
            original_status=case.status,
            original_planner_decision_count=len(case.planner_trace),
        )
        data_flags = _observed_data_flags(provisional)
        loop_flags = _observed_loop_flags(provisional)
        manifest = provisional.model_copy(
            update={
                "expectation": SensorPhoneExpectation(
                    disposition="gate_c_candidate" if not data_flags else "pilot_only",
                    expected_data_flags=data_flags,
                    expected_loop_flags=loop_flags,
                )
            }
        )
        (temp_pack / SENSOR_PHONE_MANIFEST).write_text(
            json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (temp_pack / "README.md").write_text(
            "# Deidentified Light phone evidence pack\n\n"
            "This pack contains self-collected phone light-sensor data. It keeps only "
            "relative time, illuminance, reviewed scene metadata and deterministic analysis "
            "references. Original account, investigation, recording and network identifiers "
            "were removed. Read `../README.md` for the release boundary.\n",
            encoding="utf-8",
        )
        shutil.move(str(temp_pack), str(output_dir))
    return manifest
