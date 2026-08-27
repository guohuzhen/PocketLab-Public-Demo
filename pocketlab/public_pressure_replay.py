from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pocketlab.integrity import source_file_sha256
from pocketlab.public_pressure_models import (
    PublicPressureGroundTruth,
    PublicPressureGroundTruthAnchor,
    PublicPressureLineage,
    PublicPressureSample,
    PublicPressureTrace,
)
from pocketlab.public_replay_dataset import (
    PublicReplayDatasetManifest,
    PublicSourceFile,
    load_public_replay_dataset,
    read_public_replay_recording,
    verify_public_source_registration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESSURE_PACK = (
    PROJECT_ROOT
    / "datasets"
    / "public"
    / "pressure-nist-perfloc-pixel-20180516-v1"
)
PERFLOC_TRANSFORM_ID = "pocketlab-perfloc-pressure-minimization"
PERFLOC_TRANSFORM_VERSION = "1.1.0"
_ATTESTATION_FILE_ID = "perfloc-upstream-attestation"
_CANDIDATE_CONTRACT = {
    "as4-stairwell-stable-ascent": {
        "anchor_file_id": "as4-elevation-anchors",
        "sensor_member": (
            "Google_Pixel/AS4/AS4_Google_PixelXL_google_Sensors_New.pbs"
        ),
    },
    "as5-elevator-stable-ascent": {
        "anchor_file_id": "as5-elevation-anchors",
        "sensor_member": (
            "Google_Pixel/AS5/AS5_Google_PixelXL_google_Sensors_New.pbs"
        ),
    },
}


class PublicPressureReplayError(ValueError):
    """A source-attested Pressure replay cannot be resolved safely."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PublicPressureReplayError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _safe_pack_file(pack_dir: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise PublicPressureReplayError("public Pressure path must be pack-relative")
    root = pack_dir.resolve()
    current = root
    for part in relative.parts:
        current /= part
        is_junction = getattr(current, "is_junction", lambda: False)
        if current.is_symlink() or is_junction():
            raise PublicPressureReplayError(
                "public Pressure paths cannot contain links or junctions"
            )
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise PublicPressureReplayError("public Pressure file is missing or escapes pack")
    return path


def _verified_source_file(
    pack_dir: Path,
    manifest: PublicReplayDatasetManifest,
    file_id: str,
) -> tuple[PublicSourceFile, Path]:
    matches = [item for item in manifest.source_files if item.file_id == file_id]
    if len(matches) != 1:
        raise PublicPressureReplayError(f"source file is not unique: {file_id}")
    source_file = matches[0]
    path = _safe_pack_file(pack_dir, source_file.file)
    if source_file_sha256(path, source_file.media_type) != source_file.sha256:
        raise PublicPressureReplayError(f"source file checksum mismatch: {file_id}")
    return source_file, path


def _source_recording_sha256(
    pack_dir: Path,
    manifest: PublicReplayDatasetManifest,
    member_name: str,
) -> str:
    _, path = _verified_source_file(pack_dir, manifest, _ATTESTATION_FILE_ID)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
        )
        members = payload["members"]
        matches = [item for item in members if item["member_name"] == member_name]
        if len(matches) != 1:
            raise PublicPressureReplayError(
                "raw source member is not uniquely attested"
            )
        digest = matches[0]["inner_sha256"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PublicPressureReplayError("invalid PerfLoc upstream attestation") from exc
    if not isinstance(digest, str):
        raise PublicPressureReplayError("attested source digest must be a string")
    return digest


def _ground_truth(
    pack_dir: Path,
    manifest: PublicReplayDatasetManifest,
    *,
    anchor_file_id: str,
    lineage: PublicPressureLineage,
) -> PublicPressureGroundTruth:
    source_file, path = _verified_source_file(pack_dir, manifest, anchor_file_id)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != [
                "dot_index",
                "relative_time_ms",
                "relative_elevation_m",
            ]:
                raise PublicPressureReplayError("Pressure anchor schema changed")
            anchors = tuple(
                PublicPressureGroundTruthAnchor(
                    dot_index=int(row["dot_index"]),
                    relative_time_s=float(row["relative_time_ms"]) / 1000.0,
                    relative_elevation_m=float(row["relative_elevation_m"]),
                )
                for row in reader
            )
    except (OSError, UnicodeError, csv.Error, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, PublicPressureReplayError):
            raise
        raise PublicPressureReplayError("invalid Pressure ground-truth anchors") from exc
    return PublicPressureGroundTruth(
        lineage=lineage,
        anchor_sha256=source_file.sha256,
        anchors=anchors,
    )


def load_verified_public_pressure_evidence(
    candidate_id: str,
    *,
    pack_dir: Path = DEFAULT_PRESSURE_PACK,
) -> tuple[PublicPressureTrace, PublicPressureGroundTruth]:
    """Resolve a frozen candidate from reviewed files; never accept caller samples/lineage."""

    try:
        candidate_contract = _CANDIDATE_CONTRACT[candidate_id]
    except KeyError as exc:
        raise PublicPressureReplayError(
            f"unknown public Pressure candidate: {candidate_id}"
        ) from exc
    pack_dir = pack_dir.resolve()
    manifest = load_public_replay_dataset(pack_dir)
    verify_public_source_registration(manifest)
    if manifest.sensor != "pressure" or manifest.dataset_id != pack_dir.name:
        raise PublicPressureReplayError("unexpected public Pressure dataset identity")
    transformations = manifest.transformations
    if (
        len(transformations) != 1
        or transformations[0].transformation_id != PERFLOC_TRANSFORM_ID
        or f"Transform {PERFLOC_TRANSFORM_VERSION}" not in transformations[0].description
    ):
        raise PublicPressureReplayError("public Pressure transform contract changed")
    recordings = [
        item for item in manifest.recordings if item.recording_id == candidate_id
    ]
    if len(recordings) != 1:
        raise PublicPressureReplayError("public Pressure candidate is not unique")
    recording = recordings[0]
    anchor_file_id = candidate_contract["anchor_file_id"]
    if anchor_file_id not in recording.source_file_ids:
        raise PublicPressureReplayError("candidate does not reference its hidden anchor file")
    upload = read_public_replay_recording(pack_dir, manifest, recording)
    lineage = PublicPressureLineage(
        source_id=manifest.source.source_id,
        candidate_id=candidate_id,
        source_recording_sha256=_source_recording_sha256(
            pack_dir,
            manifest,
            candidate_contract["sensor_member"],
        ),
        transform_id=PERFLOC_TRANSFORM_ID,
        transform_version=PERFLOC_TRANSFORM_VERSION,
    )
    trace = PublicPressureTrace(
        lineage=lineage,
        replay_sha256=recording.sha256,
        samples=tuple(
            PublicPressureSample(
                relative_time_s=sample.timestamp_ms / 1000.0,
                pressure_hpa=sample.values["pressure"],
            )
            for sample in upload.samples
        ),
    )
    truth = _ground_truth(
        pack_dir,
        manifest,
        anchor_file_id=anchor_file_id,
        lineage=lineage,
    )
    return trace, truth
