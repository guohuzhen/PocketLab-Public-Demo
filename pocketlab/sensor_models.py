from __future__ import annotations

import math
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, Field, model_validator

SensorKind = Literal[
    "accelerometer",
    "gyroscope",
    "magnetometer",
    "light",
    "pressure",
    "proximity",
    "microphone",
    "location",
    "bluetooth",
]
CapabilityMaturity = Literal[
    "detectable",
    "capture_ready",
    "analysis_ready",
    "agent_ready",
    "release_candidate",
]


class SensorChannelDefinition(BaseModel):
    """Meaning and unit of one numeric channel in a sensor recording."""

    unit: str = Field(min_length=1, max_length=24)
    description: str = Field(default="", max_length=120)


class SensorSample(BaseModel):
    timestamp_ms: float = Field(description="Monotonic timestamp in milliseconds")
    values: dict[str, float] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def values_must_be_finite(self) -> SensorSample:
        if not math.isfinite(self.timestamp_ms):
            raise ValueError("timestamp_ms must be finite")
        if any(not math.isfinite(value) for value in self.values.values()):
            raise ValueError("sensor values must be finite")
        return self


class PhyphoxBufferAlignmentReceipt(BaseModel):
    """Auditable proof of how one set of index-paired phyphox buffers was read.

    PocketLab never interpolates, sorts, fills or rate-converts a captured series at
    this boundary.  A bounded alignment may only discard unmatched values from the
    tail and therefore keeps one original, contiguous common prefix.
    """

    read_attempts: int = Field(ge=1, le=12)
    alignment_method: Literal["exact", "bounded_common_prefix"]
    original_lengths: dict[str, int] = Field(min_length=2, max_length=12)
    aligned_sample_count: int = Field(ge=2, le=120_000)
    discarded_tail_samples: dict[str, int] = Field(default_factory=dict, max_length=12)

    @model_validator(mode="after")
    def receipt_matches_lengths(self) -> PhyphoxBufferAlignmentReceipt:
        if any(length < self.aligned_sample_count for length in self.original_lengths.values()):
            raise ValueError("aligned sample count cannot exceed an original buffer length")
        expected_discarded = {
            role: length - self.aligned_sample_count
            for role, length in self.original_lengths.items()
            if length > self.aligned_sample_count
        }
        if self.discarded_tail_samples != expected_discarded:
            raise ValueError("discarded tail receipt must exactly match the original lengths")
        if self.alignment_method == "exact" and expected_discarded:
            raise ValueError("exact buffer alignment cannot discard samples")
        if self.alignment_method == "bounded_common_prefix" and not expected_discarded:
            raise ValueError("bounded common-prefix alignment must discard a non-empty tail")
        return self


class SensorProvenance(BaseModel):
    source: Literal[
        "phyphox_remote",
        "phone_upload",
        "file_import",
        "public_replay",
        "test_fixture",
    ]
    experiment_title: str = Field(default="", max_length=120)
    remote_session: str | None = Field(default=None, max_length=120)
    config_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 of canonical phyphox /config JSON when available.",
    )
    channel_mapping: dict[str, str] = Field(default_factory=dict, max_length=12)
    privacy_acknowledged: bool = False
    phyphox_buffer_receipt: PhyphoxBufferAlignmentReceipt | None = None
    capture_group_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        max_length=80,
    )
    clock_id: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$",
        max_length=80,
    )
    maximum_alignment_error_ms: float | None = Field(default=None, ge=0, le=250)
    alignment_method: Literal["shared_monotonic_clock"] | None = None
    general_case_id: str | None = Field(
        default=None,
        pattern=r"^general-[0-9a-f]{16}$",
        max_length=80,
    )
    general_task_id: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$",
        max_length=80,
    )
    public_dataset_id: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$",
        max_length=100,
    )
    public_recording_id: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$",
        max_length=80,
    )
    public_data_class: Literal[
        "public_real_phone_raw",
        "public_real_phone_derived",
        "source_numeric_replay",
        "synthetic",
    ] | None = None
    public_source_url: str | None = Field(default=None, max_length=500)
    public_license_spdx: str | None = Field(default=None, max_length=40)
    public_analysis_confidence_ceiling: Literal["low", "medium", "high"] | None = None
    public_invalidated_analysis_fields: list[
        Literal[
            "duration_s",
            "sampling_rate_hz",
            "sampling_jitter_ratio",
            "max_sampling_gap_ratio",
        ]
    ] = Field(default_factory=list, max_length=4)
    public_invalidated_metric_keys: list[str] = Field(default_factory=list, max_length=32)
    public_processing_disclosures: list[str] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def public_replay_metadata_is_complete(self) -> SensorProvenance:
        if self.phyphox_buffer_receipt is not None and self.source != "phyphox_remote":
            raise ValueError("only phyphox_remote provenance may attest buffer alignment")
        general_values = (self.general_case_id, self.general_task_id)
        if any(value is not None for value in general_values):
            if any(value is None for value in general_values):
                raise ValueError("general capture lineage requires case and task IDs")
            if self.source != "phyphox_remote":
                raise ValueError("only server-created phyphox records may attest a general task")
        alignment_values = (
            self.capture_group_id,
            self.clock_id,
            self.maximum_alignment_error_ms,
            self.alignment_method,
        )
        if any(value is not None for value in alignment_values):
            if any(value is None for value in alignment_values):
                raise ValueError("synchronized provenance requires a complete alignment attestation")
            if self.source != "phyphox_remote":
                raise ValueError("only server-created phyphox recordings may attest synchronization")
        public_values = (
            self.public_dataset_id,
            self.public_recording_id,
            self.public_data_class,
            self.public_source_url,
            self.public_license_spdx,
            self.public_analysis_confidence_ceiling,
        )
        if self.source == "public_replay":
            if any(value is None for value in public_values):
                raise ValueError("public_replay provenance requires complete public metadata")
            if not self.public_source_url.startswith("https://"):
                raise ValueError("public_replay source URL must use https")
            if not self.public_processing_disclosures:
                raise ValueError("public_replay provenance requires processing disclosures")
        elif (
            any(value is not None for value in public_values)
            or self.public_invalidated_analysis_fields
            or self.public_invalidated_metric_keys
            or self.public_processing_disclosures
        ):
            raise ValueError("public replay metadata is only valid for source=public_replay")
        for values, label in (
            (self.public_invalidated_analysis_fields, "public_invalidated_analysis_fields"),
            (self.public_invalidated_metric_keys, "public_invalidated_metric_keys"),
            (self.public_processing_disclosures, "public_processing_disclosures"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")
        return self


class SensorRecordingUpload(BaseModel):
    """Versioned, sensor-labelled numeric recording.

    Raw microphone waveforms are intentionally outside this contract. The microphone
    analyzer accepts derived level/amplitude buffers only, which reduces accidental
    capture of intelligible speech.
    """

    schema_version: Literal["2.0"] = "2.0"
    label: str = Field(min_length=1, max_length=80)
    device: str = Field(default="unknown phone", max_length=120)
    sensor: SensorKind
    notes: str = Field(default="", max_length=500)
    channels: dict[str, SensorChannelDefinition] = Field(min_length=1, max_length=12)
    samples: list[SensorSample] = Field(min_length=2, max_length=120_000)
    provenance: SensorProvenance

    @model_validator(mode="after")
    def validate_series_contract(self) -> SensorRecordingUpload:
        channel_names = set(self.channels)
        for channel in channel_names:
            if not channel or len(channel) > 40 or not channel.replace("_", "a").isalnum():
                raise ValueError(
                    "channel names must be 1-40 ASCII letters, digits or underscores"
                )
            if not channel[0].isalpha() or not channel.isascii() or channel.lower() != channel:
                raise ValueError(
                    "channel names must start with a lowercase ASCII letter"
                )
        for sample in self.samples:
            if set(sample.values) != channel_names:
                raise ValueError("every sample must contain exactly the declared channels")
        timestamps = [sample.timestamp_ms for sample in self.samples]
        if any(right <= left for left, right in pairwise(timestamps)):
            raise ValueError("timestamp_ms must be strictly increasing")
        if self.sensor == "location" and not self.provenance.privacy_acknowledged:
            raise ValueError("location recordings require privacy_acknowledged=true")
        return self


class AnalysisMetric(BaseModel):
    key: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=80)
    value: float
    unit: str = Field(max_length=24)

    @model_validator(mode="after")
    def metric_value_must_be_finite(self) -> AnalysisMetric:
        if not math.isfinite(self.value):
            raise ValueError("analysis metric values must be finite")
        return self


class SensorAnalysis(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    sensor: SensorKind
    analyzer_id: str
    analyzer_version: str
    sample_count: int
    duration_s: float
    sampling_rate_hz: float
    sampling_jitter_ratio: float
    max_sampling_gap_ratio: float
    confidence: Literal["low", "medium", "high"]
    warnings: list[str] = Field(default_factory=list)
    metrics: list[AnalysisMetric] = Field(default_factory=list)

    def metric_value(self, key: str) -> float:
        for metric in self.metrics:
            if metric.key == key:
                return metric.value
        raise KeyError(key)


class SensorRecordingCreated(BaseModel):
    session_id: str
    label: str
    sensor: SensorKind
    analysis: SensorAnalysis
    created_at: str


class SensorRecordingRecord(BaseModel):
    session_id: str
    upload: SensorRecordingUpload
    analysis: SensorAnalysis
    created_at: str


class SensorRecordingHistoryItem(BaseModel):
    session_id: str
    label: str
    device: str
    sensor: SensorKind
    sample_count: int
    provenance: SensorProvenance
    analysis: SensorAnalysis
    created_at: str


class SensorCapability(BaseModel):
    sensor: SensorKind
    maturity: CapabilityMaturity
    analyzer_id: str | None = None
    accepted_channels: list[str] = Field(default_factory=list)
    accepted_units: dict[str, list[str]] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class PhyphoxSensorProfile(BaseModel):
    sensor: SensorKind
    timestamp_buffer: str
    channel_buffers: dict[str, str] = Field(min_length=1, max_length=12)
    channel_units: dict[str, str] = Field(min_length=1, max_length=12)
    resolution_source: Literal["input_mapping", "official_raw_alias", "explicit_request"]


class PhyphoxSensorCaptureRequest(BaseModel):
    base_url: str = Field(min_length=10, max_length=200)
    sensor: SensorKind
    duration_s: float = Field(default=5.0, ge=1.0, le=300.0)
    label: str = Field(min_length=1, max_length=80)
    notes: str = Field(default="", max_length=500)
    privacy_acknowledged: bool = False


class PhyphoxSensorCaptureMetadata(BaseModel):
    source: Literal["phyphox_remote"] = "phyphox_remote"
    experiment_title: str
    remote_session: str | None = None
    config_sha256: str
    requested_duration_s: float
    actual_duration_s: float
    sample_count: int
    profile: PhyphoxSensorProfile


class PhyphoxSensorCaptureResponse(BaseModel):
    session: SensorRecordingCreated
    capture: PhyphoxSensorCaptureMetadata
    preview_samples: list[SensorSample]
