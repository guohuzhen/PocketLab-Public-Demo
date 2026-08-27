from __future__ import annotations

import csv
import gzip
import hashlib
import ipaddress
import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pocketlab.analyzers.registry import analyze_sensor_recording
from pocketlab.integrity import (
    file_sha256 as _sha256,
)
from pocketlab.integrity import (
    normalized_text_sha256 as _normalized_text_sha256,
)
from pocketlab.integrity import (
    source_file_sha256 as _source_file_sha256,
)
from pocketlab.sensor_models import (
    SensorAnalysis,
    SensorChannelDefinition,
    SensorKind,
    SensorProvenance,
    SensorRecordingUpload,
    SensorSample,
)

PUBLIC_REPLAY_SCHEMA_VERSION = "1.0"
PUBLIC_REPLAY_MANIFEST = "manifest.json"
PUBLIC_SOURCE_REGISTRY_VERSION = "1.0"
PUBLIC_SOURCE_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "public" / "source-registry.json"
)
MAX_PUBLIC_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_PUBLIC_RECORDING_BYTES = 64 * 1024 * 1024
MAX_PUBLIC_DECOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_PUBLIC_COMPRESSION_RATIO = 200
MAX_PUBLIC_SAMPLE_ROWS = 120_000
MAX_PUBLIC_CSV_LINE_LENGTH = 4096

PublicDataClass = Literal[
    "public_real_phone_raw",
    "public_real_phone_derived",
    "source_numeric_replay",
    "synthetic",
]
PublicReplayStatus = Literal["source_validated"]
AgentValueStatus = Literal["not_evaluated"]
PublicSourceFileRole = Literal[
    "raw",
    "author_derived",
    "pocketlab_minimized",
    "documentation",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


def _safe_relative_path(value: str, *, field_name: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field_name} must be a safe pack-relative path")
    if not value or value.endswith(("/", "\\")):
        raise ValueError(f"{field_name} must identify a file")
    return value.replace("\\", "/")


def _public_https_url(value: str, *, field_name: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"{field_name} must be a public https URL without credentials")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise ValueError(f"{field_name} cannot reference a local host")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError(f"{field_name} cannot reference a private or local address")
    return value


def _resolve_pack_file(pack_dir: Path, relative_path: str) -> Path:
    root = pack_dir.resolve()
    resolved = (root / relative_path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("public dataset file escapes the pack directory")
    current = root
    for part in Path(relative_path).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("public dataset paths cannot contain symbolic links")
    return resolved


def _resolve_project_file(relative_path: str) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    resolved = (project_root / relative_path).resolve()
    if resolved != project_root and project_root not in resolved.parents:
        raise ValueError("public build script escapes the project directory")
    current = project_root
    for part in Path(relative_path).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("public build script paths cannot contain symbolic links")
    return resolved


def _recording_set_sha256(recordings: list[PublicReplayRecording]) -> str:
    rows = sorted((item.recording_id, item.sha256) for item in recordings)
    payload = json.dumps(rows, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_semantic_sha256(manifest: PublicReplayDatasetManifest) -> str:
    """Hash the complete validated manifest semantics, independent of formatting."""

    payload = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"public replay manifest contains duplicate key: {key}")
        result[key] = value
    return result


class PublicDatasetSource(_StrictModel):
    source_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=80)
    title: str = Field(min_length=3, max_length=240)
    authors: list[str] = Field(min_length=1, max_length=32)
    publisher: str = Field(min_length=2, max_length=120)
    publication_date: date
    accessed_date: date
    record_url: str
    doi: str | None = Field(default=None, pattern=r"^10\.[0-9]{4,9}/\S+$", max_length=160)
    license_spdx: str = Field(pattern=r"^[A-Za-z0-9.+-]+$", max_length=40)
    license_url: str
    upstream_record_id: str = Field(min_length=1, max_length=120)
    associated_work_title: str | None = Field(default=None, min_length=3, max_length=240)

    @field_validator("record_url", "license_url")
    @classmethod
    def urls_are_public_https(cls, value: str, info: Any) -> str:
        return _public_https_url(value, field_name=info.field_name)

    @model_validator(mode="after")
    def accessed_after_publication(self) -> Self:
        if self.accessed_date < self.publication_date:
            raise ValueError("accessed_date cannot precede publication_date")
        return self


class PublicSourceFile(_StrictModel):
    file_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=80)
    file: str = Field(min_length=3, max_length=240)
    role: PublicSourceFileRole
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_checksum: str
    media_type: str = Field(min_length=3, max_length=100)
    license_spdx: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9.+-]+$",
        max_length=40,
    )

    @field_validator("file")
    @classmethod
    def file_is_pack_relative(cls, value: str) -> str:
        return _safe_relative_path(value, field_name="source_files.file")

    @field_validator("upstream_checksum")
    @classmethod
    def checksum_matches_algorithm(cls, value: str) -> str:
        try:
            algorithm, digest = value.split(":", 1)
        except ValueError as exc:
            raise ValueError("upstream_checksum must be algorithm:hex") from exc
        lengths = {"md5": 32, "sha256": 64}
        if algorithm not in lengths or len(digest) != lengths[algorithm]:
            raise ValueError("upstream_checksum length does not match its algorithm")
        if any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("upstream_checksum must use lowercase hexadecimal")
        return value


class PublicSourceRegistryFile(_StrictModel):
    file_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=80)
    role: PublicSourceFileRole
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_checksum: str
    media_type: str = Field(min_length=3, max_length=100)
    license_spdx: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9.+-]+$",
        max_length=40,
    )

    @field_validator("upstream_checksum")
    @classmethod
    def checksum_matches_algorithm(cls, value: str) -> str:
        return PublicSourceFile.checksum_matches_algorithm(value)


class PublicBuildScriptAttestation(_StrictModel):
    file: str = Field(min_length=3, max_length=240)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("file")
    @classmethod
    def file_is_repository_relative(cls, value: str) -> str:
        return _safe_relative_path(value, field_name="source_registry.build_scripts.file")


class PublicSourceRegistryEntry(_StrictModel):
    dataset_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=100)
    source: PublicDatasetSource
    source_files: list[PublicSourceRegistryFile] = Field(min_length=1, max_length=64)
    privacy_review: PublicPrivacyReview
    build_scripts: list[PublicBuildScriptAttestation] = Field(min_length=1, max_length=8)
    recording_count: int = Field(ge=1, le=128)
    recording_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_date: date
    review_basis: str = Field(min_length=20, max_length=800)
    evidence_urls: list[str] = Field(min_length=1, max_length=16)
    approved_for_local_replay: Literal[True] = True

    @field_validator("evidence_urls")
    @classmethod
    def evidence_urls_are_public_https(cls, values: list[str]) -> list[str]:
        checked = [
            _public_https_url(value, field_name="source_registry.evidence_urls")
            for value in values
        ]
        if len(checked) != len(set(checked)):
            raise ValueError("source registry evidence URLs must be unique")
        return checked

    @model_validator(mode="after")
    def file_ids_are_unique(self) -> Self:
        file_ids = [item.file_id for item in self.source_files]
        if len(file_ids) != len(set(file_ids)):
            raise ValueError("source registry file ids must be unique")
        if self.review_date < self.source.publication_date:
            raise ValueError("source registry review cannot precede publication")
        script_files = [item.file for item in self.build_scripts]
        if len(script_files) != len(set(script_files)):
            raise ValueError("source registry build scripts must be unique")
        return self


class PublicSourceRegistry(_StrictModel):
    schema_version: Literal["1.0"] = PUBLIC_SOURCE_REGISTRY_VERSION
    registry_kind: Literal["public_source_review_registry"] = (
        "public_source_review_registry"
    )
    sources: list[PublicSourceRegistryEntry] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def source_ids_are_unique(self) -> Self:
        source_ids = [item.source.source_id for item in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source registry source ids must be unique")
        return self


class PublicPrivacyReview(_StrictModel):
    review_date: date
    review_basis: str = Field(min_length=10, max_length=500)
    source_sensitive_categories: list[
        Literal[
            "pseudonymous_participant_id",
            "behavioral_light_signature",
            "precise_location",
            "raw_audio",
            "bluetooth_hardware_identifier",
            "absolute_time",
            "multi_sensor_motion_stream",
        ]
    ] = Field(default_factory=list, max_length=8)
    replay_sensitive_categories: list[
        Literal[
            "pseudonymous_participant_id",
            "behavioral_light_signature",
            "precise_location",
            "raw_audio",
            "bluetooth_hardware_identifier",
            "absolute_time",
            "multi_sensor_motion_stream",
        ]
    ] = Field(default_factory=list, max_length=8)
    protections: list[str] = Field(min_length=1, max_length=16)
    deployment_scope: Literal["local_only", "local_and_deployed"]
    allowed_operations: list[
        Literal["catalog", "local_replay", "account_import", "agent_evaluation", "export"]
    ] = Field(min_length=1, max_length=5)
    requires_user_acknowledgement: bool
    approved_for_local_replay: Literal[True] = True

    @model_validator(mode="after")
    def sensitive_payload_is_not_approved(self) -> Self:
        forbidden = {
            "precise_location",
            "raw_audio",
            "bluetooth_hardware_identifier",
        }
        if forbidden.intersection(self.replay_sensitive_categories):
            raise ValueError(
                "public replay cannot retain precise location, raw audio, or hardware identifiers"
            )
        if len(self.source_sensitive_categories) != len(set(self.source_sensitive_categories)):
            raise ValueError("source_sensitive_categories must be unique")
        if len(self.replay_sensitive_categories) != len(set(self.replay_sensitive_categories)):
            raise ValueError("replay_sensitive_categories must be unique")
        if len(self.allowed_operations) != len(set(self.allowed_operations)):
            raise ValueError("allowed_operations must be unique")
        if "local_replay" not in self.allowed_operations:
            raise ValueError("approved public sources must allow local_replay")
        if "behavioral_light_signature" in self.replay_sensitive_categories:
            if self.deployment_scope != "local_only":
                raise ValueError("behavioral light signatures must remain local-only")
            if not self.requires_user_acknowledgement:
                raise ValueError("behavioral light signatures require user acknowledgement")
            forbidden_operations = {"account_import", "export"}
            if forbidden_operations.intersection(self.allowed_operations):
                raise ValueError(
                    "behavioral light signatures cannot be imported to accounts or exported"
                )
        restricted_source_categories = {
            "absolute_time",
            "multi_sensor_motion_stream",
        }
        if restricted_source_categories.intersection(self.source_sensitive_categories):
            if self.deployment_scope != "local_only":
                raise ValueError("sensitive raw source material must remain local-only")
            if not self.requires_user_acknowledgement:
                raise ValueError(
                    "sensitive raw source material requires user acknowledgement"
                )
            forbidden_operations = {"account_import", "export"}
            if forbidden_operations.intersection(self.allowed_operations):
                raise ValueError(
                    "sensitive raw source material cannot be imported to accounts or exported"
                )
        return self


class PublicTransformation(_StrictModel):
    transformation_id: str = Field(
        pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=80
    )
    kind: Literal["author_derived", "pocketlab_conversion"]
    input_file_ids: list[str] = Field(min_length=1, max_length=16)
    description: str = Field(min_length=10, max_length=500)
    reproducible_script: str | None = Field(default=None, max_length=240)
    omitted_source_columns: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("reproducible_script")
    @classmethod
    def script_is_pack_relative(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _safe_relative_path(
            value,
            field_name="transformation.reproducible_script (repository-relative)",
        )


class PublicOracleMetric(_StrictModel):
    metric_key: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=80)
    expected_value: float
    absolute_tolerance: float = Field(ge=0)
    method: str = Field(min_length=10, max_length=300)


class PublicReplayRecording(_StrictModel):
    recording_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=80)
    file: str = Field(min_length=3, max_length=240)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label: str = Field(min_length=2, max_length=120)
    device_alias: str = Field(min_length=3, max_length=120)
    experiment_title: str = Field(min_length=2, max_length=120)
    scene_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=80)
    condition_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=80)
    acquisition_id: str = Field(
        pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=100
    )
    repeat_index: int = Field(ge=1, le=1000)
    evidence_role: Literal[
        "baseline", "condition", "perturbation", "calibration", "exploration"
    ]
    timestamp_column: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=80)
    timestamp_unit: Literal["ms", "s"]
    csv_columns: dict[str, str] = Field(min_length=1, max_length=12)
    sample_count: int = Field(ge=2, le=MAX_PUBLIC_SAMPLE_ROWS)
    source_file_ids: list[str] = Field(min_length=1, max_length=16)
    transformation_ids: list[str] = Field(default_factory=list, max_length=16)
    independent_measurement: bool
    gate_c_eligible: Literal[False] = False
    analysis_confidence_ceiling: Literal["low", "medium", "high"]
    invalidated_analysis_fields: list[
        Literal[
            "duration_s",
            "sampling_rate_hz",
            "sampling_jitter_ratio",
            "max_sampling_gap_ratio",
        ]
    ] = Field(default_factory=list, max_length=4)
    invalidated_metric_keys: list[str] = Field(default_factory=list, max_length=32)
    processing_disclosures: list[str] = Field(min_length=1, max_length=16)
    oracle_metrics: list[PublicOracleMetric] = Field(min_length=3, max_length=32)
    reference_analysis: SensorAnalysis

    @field_validator("file")
    @classmethod
    def file_is_pack_relative(cls, value: str) -> str:
        return _safe_relative_path(value, field_name="recordings.file")

    @model_validator(mode="after")
    def columns_are_unambiguous(self) -> Self:
        if self.timestamp_column in self.csv_columns.values():
            raise ValueError("timestamp_column cannot also be a sensor value column")
        columns = list(self.csv_columns.values())
        if len(columns) != len(set(columns)):
            raise ValueError("csv_columns values must be unique")
        if self.reference_analysis.sample_count != self.sample_count:
            raise ValueError("reference analysis sample_count must match recording")
        oracle_keys = [item.metric_key for item in self.oracle_metrics]
        if len(oracle_keys) != len(set(oracle_keys)):
            raise ValueError("oracle metric keys must be unique")
        if len(self.invalidated_analysis_fields) != len(
            set(self.invalidated_analysis_fields)
        ):
            raise ValueError("invalidated_analysis_fields must be unique")
        if len(self.invalidated_metric_keys) != len(set(self.invalidated_metric_keys)):
            raise ValueError("invalidated_metric_keys must be unique")
        if len(self.processing_disclosures) != len(set(self.processing_disclosures)):
            raise ValueError("processing_disclosures must be unique")
        reference_metric_keys = {item.key for item in self.reference_analysis.metrics}
        if set(self.invalidated_metric_keys).intersection(reference_metric_keys):
            raise ValueError("invalidated metrics cannot remain in reference_analysis")
        if not set(oracle_keys) <= reference_metric_keys:
            raise ValueError("oracle metrics must reference eligible analysis metrics")
        return self


class PublicReplayDatasetManifest(_StrictModel):
    schema_version: Literal["1.0"] = PUBLIC_REPLAY_SCHEMA_VERSION
    dataset_kind: Literal["public_sensor_replay"] = "public_sensor_replay"
    dataset_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=100)
    title: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=10, max_length=700)
    data_class: PublicDataClass
    sensor: SensorKind
    analyzer_id: str = Field(min_length=3, max_length=120)
    analyzer_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    channels: dict[str, SensorChannelDefinition] = Field(min_length=1, max_length=12)
    source: PublicDatasetSource
    privacy_review: PublicPrivacyReview
    source_files: list[PublicSourceFile] = Field(min_length=1, max_length=32)
    transformations: list[PublicTransformation] = Field(default_factory=list, max_length=32)
    recordings: list[PublicReplayRecording] = Field(min_length=1, max_length=128)
    claim_boundary: list[str] = Field(min_length=1, max_length=24)
    public_replay_status: PublicReplayStatus
    agent_value_status: AgentValueStatus
    self_collected_phone: Literal[False] = False
    market_validated: Literal[False] = False
    agent_ready: Literal[False] = False

    @model_validator(mode="after")
    def validate_manifest_graph(self) -> Self:
        source_ids = [item.file_id for item in self.source_files]
        transformation_ids = [item.transformation_id for item in self.transformations]
        recording_ids = [item.recording_id for item in self.recordings]
        recording_files = [item.file for item in self.recordings]
        for values, label in (
            (source_ids, "source file id"),
            (transformation_ids, "transformation id"),
            (recording_ids, "recording id"),
            (recording_files, "recording file"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        independent = [item for item in self.recordings if item.independent_measurement]
        independent_acquisitions = [item.acquisition_id for item in independent]
        independent_hashes = [item.sha256 for item in independent]
        if len(independent_acquisitions) != len(set(independent_acquisitions)):
            raise ValueError("independent recordings require unique acquisition_id values")
        if len(independent_hashes) != len(set(independent_hashes)):
            raise ValueError("identical replay files cannot count as independent measurements")
        known_sources = set(source_ids)
        known_transformations = set(transformation_ids)
        for transformation in self.transformations:
            if not set(transformation.input_file_ids) <= known_sources:
                raise ValueError("transformation references an unknown source file")
        for recording in self.recordings:
            if not set(recording.source_file_ids) <= known_sources:
                raise ValueError("recording references an unknown source file")
            if not set(recording.transformation_ids) <= known_transformations:
                raise ValueError("recording references an unknown transformation")
            if recording.reference_analysis.sensor != self.sensor:
                raise ValueError("recording reference analysis sensor does not match manifest")
            if recording.reference_analysis.analyzer_id != self.analyzer_id:
                raise ValueError("recording reference analysis analyzer does not match manifest")
            if recording.reference_analysis.analyzer_version != self.analyzer_version:
                raise ValueError("recording reference analyzer version does not match manifest")
            if set(recording.csv_columns) != set(self.channels):
                raise ValueError("recording CSV mapping must cover exactly the manifest channels")
        if self.data_class == "public_real_phone_raw":
            if not any(item.role == "raw" for item in self.source_files):
                raise ValueError("public_real_phone_raw requires a raw source file")
        elif self.data_class == "public_real_phone_derived":
            if not any(item.role == "raw" for item in self.source_files):
                raise ValueError("public_real_phone_derived must retain the upstream raw file")
            if not any(item.role == "author_derived" for item in self.source_files):
                raise ValueError("public_real_phone_derived requires an author-derived source file")
            if not self.transformations:
                raise ValueError("public_real_phone_derived requires explicit transformations")
        if self.sensor == "location" and "precise_location" in (
            self.privacy_review.replay_sensitive_categories
        ):
            raise ValueError("location replay must remove precise coordinates")
        if self.sensor == "microphone" and "raw_audio" in (
            self.privacy_review.replay_sensitive_categories
        ):
            raise ValueError("microphone replay must exclude raw audio")
        if self.sensor == "bluetooth" and "bluetooth_hardware_identifier" in (
            self.privacy_review.replay_sensitive_categories
        ):
            raise ValueError("Bluetooth replay must remove hardware identifiers")
        return self


class PublicReplayRecordingSummary(_StrictModel):
    recording_id: str = Field(
        pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=80
    )
    label: str = Field(min_length=2, max_length=120)
    sample_count: int = Field(ge=2, le=MAX_PUBLIC_SAMPLE_ROWS)
    evidence_role: Literal[
        "baseline", "condition", "perturbation", "calibration", "exploration"
    ]
    independent_measurement: bool


class PublicReplayCatalogItem(_StrictModel):
    dataset_id: str = Field(
        pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$", max_length=100
    )
    title: str
    description: str
    sensor: SensorKind
    data_class: PublicDataClass
    public_replay_status: Literal["source_validated"] = "source_validated"
    agent_value_status: Literal["not_evaluated"] = "not_evaluated"
    source_title: str
    source_url: str
    doi: str | None
    license_spdx: str
    source_file_licenses: list[str] = Field(default_factory=list, max_length=8)
    privacy_risk_categories: list[str] = Field(default_factory=list, max_length=8)
    deployment_scope: Literal["local_only", "local_and_deployed"]
    import_allowed: bool
    requires_user_acknowledgement: bool
    recording_count: int = Field(ge=1, le=128)
    claim_boundary: list[str]
    recordings: list[PublicReplayRecordingSummary]
    public_replay_ready: Literal[False] = False
    agent_ready: Literal[False] = False


def load_public_source_registry(
    registry_path: Path = PUBLIC_SOURCE_REGISTRY_PATH,
) -> PublicSourceRegistry:
    if not registry_path.is_file():
        raise FileNotFoundError("missing public source review registry")
    if registry_path.stat().st_size > MAX_PUBLIC_MANIFEST_BYTES:
        raise ValueError("public source review registry exceeds the 4 MiB limit")
    payload = json.loads(
        registry_path.read_text(encoding="utf-8"),
        object_pairs_hook=_json_without_duplicate_keys,
    )
    return PublicSourceRegistry.model_validate(payload)


def verify_public_source_registration(
    manifest: PublicReplayDatasetManifest,
    *,
    registry_path: Path = PUBLIC_SOURCE_REGISTRY_PATH,
) -> PublicSourceRegistryEntry:
    registry = load_public_source_registry(registry_path)
    matches = [
        item for item in registry.sources if item.source.source_id == manifest.source.source_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"public source is not uniquely approved: {manifest.source.source_id}"
        )
    entry = matches[0]
    if entry.dataset_id != manifest.dataset_id:
        raise ValueError("public dataset id differs from the reviewed registry")
    if entry.source.model_dump(mode="json") != manifest.source.model_dump(mode="json"):
        raise ValueError("public source metadata differs from the reviewed registry")
    registered_files = {
        item.file_id: item.model_dump(mode="json") for item in entry.source_files
    }
    observed_files: dict[str, dict[str, object]] = {}
    for source_file in manifest.source_files:
        observed = PublicSourceRegistryFile(
            file_id=source_file.file_id,
            role=source_file.role,
            sha256=source_file.sha256,
            upstream_checksum=source_file.upstream_checksum,
            media_type=source_file.media_type,
            license_spdx=source_file.license_spdx,
        ).model_dump(mode="json")
        observed_files[source_file.file_id] = observed
        if registered_files.get(source_file.file_id) != observed:
            raise ValueError(
                "public source file differs from the reviewed registry: "
                f"{source_file.file_id}"
            )
    if observed_files != registered_files:
        raise ValueError("public source file set differs from the reviewed registry")
    if (
        entry.privacy_review.model_dump(mode="json")
        != manifest.privacy_review.model_dump(mode="json")
    ):
        raise ValueError("public privacy review differs from the reviewed registry")
    manifest_scripts = {
        item.reproducible_script
        for item in manifest.transformations
        if item.reproducible_script is not None
    }
    registered_scripts = {item.file for item in entry.build_scripts}
    if manifest_scripts != registered_scripts:
        raise ValueError("public build script set differs from the reviewed registry")
    for script in entry.build_scripts:
        script_path = _resolve_project_file(script.file)
        if (
            not script_path.is_file()
            or _normalized_text_sha256(script_path) != script.sha256
        ):
            raise ValueError(f"public build script attestation failed: {script.file}")
    if len(manifest.recordings) != entry.recording_count:
        raise ValueError("public recording count differs from the reviewed registry")
    if _recording_set_sha256(manifest.recordings) != entry.recording_set_sha256:
        raise ValueError("public recording set differs from the reviewed registry")
    if _manifest_semantic_sha256(manifest) != entry.manifest_sha256:
        raise ValueError("public manifest semantics differ from the reviewed registry")
    return entry


def load_public_replay_dataset(pack_dir: Path) -> PublicReplayDatasetManifest:
    manifest_path = _resolve_pack_file(pack_dir, PUBLIC_REPLAY_MANIFEST)
    if not manifest_path.is_file():
        raise FileNotFoundError("missing public replay manifest")
    if manifest_path.stat().st_size > MAX_PUBLIC_MANIFEST_BYTES:
        raise ValueError("public replay manifest exceeds the 4 MiB limit")
    payload = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=_json_without_duplicate_keys,
    )
    return PublicReplayDatasetManifest.model_validate(payload)


def verify_public_source_files(
    pack_dir: Path,
    manifest: PublicReplayDatasetManifest,
) -> None:
    verify_public_source_registration(manifest)
    for source_file in manifest.source_files:
        path = _resolve_pack_file(pack_dir, source_file.file)
        if not path.is_file():
            raise FileNotFoundError(f"missing public source file: {source_file.file}")
        observed = _source_file_sha256(path, source_file.media_type)
        if observed != source_file.sha256:
            raise ValueError(
                f"public source checksum mismatch: expected={source_file.sha256}, "
                f"observed={observed}"
            )


def _bounded_utf8_lines(binary_handle: Any, *, compressed_bytes: int) -> Any:
    expanded_bytes = 0
    compression_limit = max(1024 * 1024, compressed_bytes * MAX_PUBLIC_COMPRESSION_RATIO)
    while True:
        line = binary_handle.readline(MAX_PUBLIC_CSV_LINE_LENGTH + 1)
        if not line:
            return
        if len(line) > MAX_PUBLIC_CSV_LINE_LENGTH:
            raise ValueError("public replay CSV line exceeds the length limit")
        expanded_bytes += len(line)
        if expanded_bytes > MAX_PUBLIC_DECOMPRESSED_BYTES:
            raise ValueError("public replay recording exceeds the decompressed size limit")
        if expanded_bytes > compression_limit:
            raise ValueError("public replay recording exceeds the compression-ratio limit")
        try:
            yield line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("public replay CSV must be UTF-8") from exc


def read_public_replay_recording(
    pack_dir: Path,
    manifest: PublicReplayDatasetManifest,
    recording: PublicReplayRecording,
) -> SensorRecordingUpload:
    path = _resolve_pack_file(pack_dir, recording.file)
    if not path.is_file():
        raise FileNotFoundError(f"missing public replay recording: {recording.file}")
    if path.stat().st_size > MAX_PUBLIC_RECORDING_BYTES:
        raise ValueError("public replay recording exceeds the 64 MiB limit")
    observed_sha = _sha256(path)
    if observed_sha != recording.sha256:
        raise ValueError(
            f"public replay checksum mismatch: expected={recording.sha256}, "
            f"observed={observed_sha}"
        )

    compressed_bytes = path.stat().st_size
    raw_handle = path.open("rb")
    binary_handle = gzip.GzipFile(fileobj=raw_handle) if path.suffix == ".gz" else raw_handle
    try:
        reader = csv.DictReader(
            _bounded_utf8_lines(binary_handle, compressed_bytes=compressed_bytes)
        )
        try:
            expected = [recording.timestamp_column, *recording.csv_columns.values()]
            if reader.fieldnames != expected:
                raise ValueError(f"public replay CSV columns must be exactly {expected}")
            samples: list[SensorSample] = []
            for row_number, row in enumerate(reader, start=2):
                if row_number > MAX_PUBLIC_SAMPLE_ROWS + 1:
                    raise ValueError("public replay recording exceeds the row limit")
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("public replay CSV row has missing or extra fields")
                if any(len(value) > MAX_PUBLIC_CSV_LINE_LENGTH for value in row.values()):
                    raise ValueError("public replay CSV cell exceeds the length limit")
                try:
                    timestamp = float(row[recording.timestamp_column])
                    values = {
                        channel: float(row[column])
                        for channel, column in recording.csv_columns.items()
                    }
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"public replay CSV row {row_number} contains invalid numeric data"
                    ) from exc
                if recording.timestamp_unit == "s":
                    timestamp *= 1000.0
                if not math.isfinite(timestamp) or any(
                    not math.isfinite(value) for value in values.values()
                ):
                    raise ValueError(
                        f"public replay CSV row {row_number} contains non-finite data"
                    )
                samples.append(SensorSample(timestamp_ms=timestamp, values=values))
        except (gzip.BadGzipFile, EOFError) as exc:
            raise ValueError("public replay gzip payload is invalid") from exc
    finally:
        if binary_handle is not raw_handle:
            binary_handle.close()
        if not raw_handle.closed:
            raw_handle.close()

    if len(samples) != recording.sample_count:
        raise ValueError(
            f"public replay sample_count mismatch: expected={recording.sample_count}, "
            f"observed={len(samples)}"
        )
    return SensorRecordingUpload(
        label=recording.label,
        device=recording.device_alias,
        sensor=manifest.sensor,
        notes="",
        channels=manifest.channels,
        samples=samples,
        provenance=SensorProvenance(
            source="public_replay",
            experiment_title=recording.experiment_title,
            channel_mapping=dict(recording.csv_columns),
            privacy_acknowledged=(
                manifest.sensor in {"location", "microphone"}
                and manifest.privacy_review.approved_for_local_replay
            ),
            public_dataset_id=manifest.dataset_id,
            public_recording_id=recording.recording_id,
            public_data_class=manifest.data_class,
            public_source_url=manifest.source.record_url,
            public_license_spdx=manifest.source.license_spdx,
            public_analysis_confidence_ceiling=recording.analysis_confidence_ceiling,
            public_invalidated_analysis_fields=recording.invalidated_analysis_fields,
            public_invalidated_metric_keys=recording.invalidated_metric_keys,
            public_processing_disclosures=recording.processing_disclosures,
        ),
    )


def _reference_values_match(observed: object, reference: object) -> bool:
    if isinstance(observed, float) and isinstance(reference, float):
        return math.isclose(observed, reference, rel_tol=1e-12, abs_tol=1e-9)
    if isinstance(observed, dict) and isinstance(reference, dict):
        return observed.keys() == reference.keys() and all(
            _reference_values_match(observed[key], reference[key]) for key in observed
        )
    if isinstance(observed, list) and isinstance(reference, list):
        return len(observed) == len(reference) and all(
            _reference_values_match(left, right)
            for left, right in zip(observed, reference, strict=True)
        )
    return observed == reference


def public_replay_analysis_matches_reference(
    observed: SensorAnalysis,
    reference: SensorAnalysis,
) -> bool:
    """Compare frozen analyses while tolerating non-physical float tail differences."""

    return _reference_values_match(
        observed.model_dump(mode="json"),
        reference.model_dump(mode="json"),
    )


def evaluate_public_replay_dataset(
    pack_dir: Path,
    *,
    replay_repeat: int = 3,
) -> dict[str, Any]:
    if replay_repeat < 1:
        raise ValueError("replay_repeat must be at least 1")
    manifest = load_public_replay_dataset(pack_dir)
    registry_entry = verify_public_source_registration(manifest)
    checks: list[dict[str, Any]] = [
        {
            "name": "reviewed_source_registry",
            "passed": True,
            "detail": (
                f"source_id={manifest.source.source_id}, "
                f"review_date={registry_entry.review_date.isoformat()}"
            ),
        }
    ]
    for source_file in manifest.source_files:
        path = _resolve_pack_file(pack_dir, source_file.file)
        observed = (
            _source_file_sha256(path, source_file.media_type)
            if path.is_file()
            else None
        )
        checks.append(
            {
                "name": f"source_integrity:{source_file.file_id}",
                "passed": observed == source_file.sha256,
                "detail": f"expected={source_file.sha256}, observed={observed}",
            }
        )

    recording_results: list[dict[str, Any]] = []
    for recording in manifest.recordings:
        upload = read_public_replay_recording(pack_dir, manifest, recording)
        repeated = [analyze_sensor_recording(upload) for _ in range(replay_repeat)]
        canonical = [item.model_dump(mode="json") for item in repeated]
        repeat_consistent = all(item == canonical[0] for item in canonical[1:])
        reference_matches = public_replay_analysis_matches_reference(
            repeated[0],
            recording.reference_analysis,
        )
        oracle_checks: list[dict[str, Any]] = []
        metric_values = {item["key"]: item["value"] for item in canonical[0]["metrics"]}
        for oracle in recording.oracle_metrics:
            observed = metric_values.get(oracle.metric_key)
            passed_oracle = observed is not None and math.isclose(
                observed,
                oracle.expected_value,
                rel_tol=0.0,
                abs_tol=oracle.absolute_tolerance,
            )
            oracle_checks.append(
                {
                    "name": f"independent_oracle:{oracle.metric_key}",
                    "passed": passed_oracle,
                    "detail": (
                        f"expected={oracle.expected_value} ± {oracle.absolute_tolerance}, "
                        f"observed={observed}; {oracle.method}"
                    ),
                }
            )
        recording_checks = [
            {
                "name": "recording_integrity",
                "passed": True,
                "detail": recording.sha256,
            },
            {
                "name": "deterministic_replay",
                "passed": repeat_consistent,
                "detail": f"repeat={replay_repeat}",
            },
            {
                "name": "reference_analysis_regression",
                "passed": reference_matches,
                "detail": f"{manifest.analyzer_id}@{manifest.analyzer_version}",
            },
            *oracle_checks,
            {
                "name": "truthful_gate_c_boundary",
                "passed": recording.gate_c_eligible is False,
                "detail": "public replay records do not count as self-collected Gate C evidence",
            },
        ]
        checks.extend(
            {
                **item,
                "name": f"{recording.recording_id}:{item['name']}",
            }
            for item in recording_checks
        )
        recording_results.append(
            {
                "recording_id": recording.recording_id,
                "sample_count": recording.sample_count,
                "independent_measurement": recording.independent_measurement,
                "gate_c_eligible": False,
                "analysis": canonical[0],
                "checks": recording_checks,
            }
        )

    checks.append(
        {
            "name": "release_boundary",
            "passed": not manifest.self_collected_phone
            and not manifest.market_validated
            and not manifest.agent_ready,
            "detail": "self_collected_phone=false, market_validated=false, agent_ready=false",
        }
    )
    passed = sum(bool(item["passed"]) for item in checks)
    result = {
        "schema_version": PUBLIC_REPLAY_SCHEMA_VERSION,
        "dataset_id": manifest.dataset_id,
        "sensor": manifest.sensor,
        "data_class": manifest.data_class,
        "source_id": manifest.source.source_id,
        "license_spdx": manifest.source.license_spdx,
        "claim_boundary": list(manifest.claim_boundary),
        "recording_count": len(manifest.recordings),
        "independent_measurement_count": sum(
            item.independent_measurement for item in manifest.recordings
        ),
        "public_real_phone_record_count": (
            len(manifest.recordings)
            if manifest.data_class.startswith("public_real_phone")
            else 0
        ),
        "gate_c_credited_records": 0,
        "public_replay_status": manifest.public_replay_status,
        "agent_value_status": manifest.agent_value_status,
        "source_validated": passed == len(checks),
        "public_replay_ready": False,
        "market_validated": False,
        "agent_ready": False,
        "recordings": recording_results,
        "checks": checks,
        "summary": {
            "passed": passed,
            "total": len(checks),
            "score_percent": round(100.0 * passed / len(checks), 4),
        },
    }
    return result


def discover_public_replay_packs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path.parent for path in root.glob(f"*/{PUBLIC_REPLAY_MANIFEST}"))


def get_public_replay_dataset(
    root: Path,
    dataset_id: str,
) -> tuple[Path, PublicReplayDatasetManifest]:
    if not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", dataset_id):
        raise KeyError(f"Unknown public replay dataset: {dataset_id}")
    matches: list[tuple[Path, PublicReplayDatasetManifest]] = []
    for pack_dir in discover_public_replay_packs(root):
        try:
            manifest = load_public_replay_dataset(pack_dir)
        except (OSError, ValueError, TypeError, UnicodeError):
            if pack_dir.name == dataset_id:
                raise
            continue
        if manifest.dataset_id == dataset_id:
            matches.append((pack_dir, manifest))
    if not matches:
        raise KeyError(f"Unknown public replay dataset: {dataset_id}")
    if len(matches) != 1:
        raise ValueError(f"Duplicate public replay dataset_id: {dataset_id}")
    return matches[0]


def list_public_replay_catalog(root: Path) -> list[PublicReplayCatalogItem]:
    catalog: list[PublicReplayCatalogItem] = []
    dataset_ids: set[str] = set()
    for pack_dir in discover_public_replay_packs(root):
        manifest = load_public_replay_dataset(pack_dir)
        if manifest.dataset_id in dataset_ids:
            raise ValueError(f"Duplicate public replay dataset_id: {manifest.dataset_id}")
        dataset_ids.add(manifest.dataset_id)
        verification = evaluate_public_replay_dataset(pack_dir, replay_repeat=1)
        if not verification["source_validated"]:
            raise ValueError(f"public replay pack did not validate: {manifest.dataset_id}")
        catalog.append(
            PublicReplayCatalogItem(
                dataset_id=manifest.dataset_id,
                title=manifest.title,
                description=manifest.description,
                sensor=manifest.sensor,
                data_class=manifest.data_class,
                source_title=manifest.source.title,
                source_url=manifest.source.record_url,
                doi=manifest.source.doi,
                license_spdx=manifest.source.license_spdx,
                source_file_licenses=sorted(
                    {
                        item.license_spdx
                        for item in manifest.source_files
                        if item.license_spdx is not None
                    }
                ),
                privacy_risk_categories=list(
                    manifest.privacy_review.replay_sensitive_categories
                ),
                deployment_scope=manifest.privacy_review.deployment_scope,
                import_allowed=(
                    "account_import" in manifest.privacy_review.allowed_operations
                ),
                requires_user_acknowledgement=(
                    manifest.privacy_review.requires_user_acknowledgement
                ),
                recording_count=len(manifest.recordings),
                claim_boundary=list(manifest.claim_boundary),
                recordings=[
                    PublicReplayRecordingSummary(
                        recording_id=recording.recording_id,
                        label=recording.label,
                        sample_count=recording.sample_count,
                        evidence_role=recording.evidence_role,
                        independent_measurement=recording.independent_measurement,
                    )
                    for recording in manifest.recordings
                ],
            )
        )
    return catalog


def run_public_replay_harness(
    packs: list[Path],
    *,
    replay_repeat: int = 3,
) -> dict[str, Any]:
    if not packs:
        return {
            "schema_version": PUBLIC_REPLAY_SCHEMA_VERSION,
            "execution_status": "blocked_missing_evidence",
            "packs": [],
            "public_real_phone_records": 0,
            "gate_c_credited_records": 0,
            "public_replay_ready_sensors": [],
            "market_validated": False,
            "agent_ready": False,
        }
    results: list[dict[str, Any]] = []
    for path in packs:
        try:
            results.append(evaluate_public_replay_dataset(path, replay_repeat=replay_repeat))
        except (OSError, ValueError, KeyError, TypeError, UnicodeError, csv.Error) as exc:
            results.append(
                {
                    "schema_version": PUBLIC_REPLAY_SCHEMA_VERSION,
                    "dataset_id": path.name,
                    "sensor": "unknown",
                    "data_class": "unknown",
                    "license_spdx": "unknown",
                    "claim_boundary": ["Pack validation failed; no scientific claim is allowed."],
                    "recording_count": 0,
                    "independent_measurement_count": 0,
                    "public_real_phone_record_count": 0,
                    "gate_c_credited_records": 0,
                    "public_replay_status": "invalid_evidence",
                    "agent_value_status": "not_evaluated",
                    "source_validated": False,
                    "public_replay_ready": False,
                    "market_validated": False,
                    "agent_ready": False,
                    "recordings": [],
                    "checks": [
                        {
                            "name": "pack_validation",
                            "passed": False,
                            "detail": f"{type(exc).__name__}: {exc}",
                        }
                    ],
                    "summary": {"passed": 0, "total": 1, "score_percent": 0.0},
                }
            )
    validated = all(item["source_validated"] for item in results)
    return {
        "schema_version": PUBLIC_REPLAY_SCHEMA_VERSION,
        "execution_status": "completed" if validated else "failed",
        "packs": results,
        "public_real_phone_records": sum(
            item["public_real_phone_record_count"] for item in results
        ),
        "gate_c_credited_records": 0,
        "public_replay_ready_sensors": sorted(
            {item["sensor"] for item in results if item["public_replay_ready"]}
        ),
        "market_validated": False,
        "agent_ready": False,
    }


def render_public_replay_report(result: dict[str, Any]) -> str:
    lines = [
        "# Public Sensor Replay Harness",
        "",
        f"- execution_status: `{result['execution_status']}`",
        f"- public_real_phone_records: `{result['public_real_phone_records']}`",
        f"- gate_c_credited_records: `{result['gate_c_credited_records']}`",
        "- market_validated: `false`",
        "- agent_ready: `false`",
        "",
    ]
    for pack in result["packs"]:
        lines.extend(
            [
                f"## {pack['dataset_id']}",
                "",
                f"- sensor: `{pack['sensor']}`",
                f"- data_class: `{pack['data_class']}`",
                f"- license: `{pack['license_spdx']}`",
                f"- source_validated: `{str(pack['source_validated']).lower()}`",
                f"- public_replay_ready: `{str(pack['public_replay_ready']).lower()}`",
                f"- Agent Value: `{pack['agent_value_status']}`",
                f"- recordings: `{pack['recording_count']}`",
                f"- independent_measurements: `{pack['independent_measurement_count']}`",
                "- Gate C credit: `0`",
                "- Gate E: `not_evaluated`",
                "- Gate H: `not_evaluated`",
                (
                    "- artifact integrity / deterministic replay / independent oracle "
                    f"checks: `{pack['summary']['passed']}/{pack['summary']['total']}`"
                ),
                "- claim boundary:",
                *[f"  - {item}" for item in pack["claim_boundary"]],
                "",
            ]
        )
    return "\n".join(lines)


def write_public_replay_results(
    result: dict[str, Any],
    *,
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> None:
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_public_replay_report(result), encoding="utf-8")
