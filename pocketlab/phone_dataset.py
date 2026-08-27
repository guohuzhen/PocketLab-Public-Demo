from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pocketlab.schemas import AccelerationSample, SessionUpload, VibrationAnalysis
from pocketlab.signal_processing import analyze_acceleration

PHONE_DATASET_SCHEMA_VERSION = 1
PHONE_DATASET_MANIFEST = "manifest.json"
REQUIRED_SAMPLE_COLUMNS = ("timestamp_ms", "x_m_s2", "y_m_s2", "z_m_s2")


class PhoneDatasetProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection_kind: Literal["self_collected_phone"]
    capture_app: str = Field(min_length=2, max_length=80)
    exported_at: str
    consent_basis: str = Field(min_length=10, max_length=500)
    license_note: str = Field(min_length=3, max_length=500)
    original_session_id_removed: bool = True
    user_identifier_removed: bool = True
    network_address_removed: bool = True


class PhoneDatasetScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem_statement: str = Field(min_length=10, max_length=1000)
    environment: str = Field(min_length=3, max_length=1000)
    phone_placement: str = Field(min_length=3, max_length=500)
    intended_variable: str = Field(min_length=2, max_length=300)
    controlled_variables: list[str] = Field(default_factory=list, max_length=20)


class PhoneRecordingDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recording_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    file: str = Field(min_length=5, max_length=240)
    label: str = Field(min_length=1, max_length=120)
    condition_name: str = Field(min_length=2, max_length=120)
    condition_role: Literal["baseline", "control", "replicate", "unpaired", "unpaired_control"]
    variable_value: str = Field(min_length=1, max_length=300)
    capture_source: Literal["phyphox_remote", "phone_file_import"]
    device_label: str = Field(min_length=2, max_length=120)
    experiment_title: str = Field(min_length=1, max_length=120)
    requested_duration_s: float | None = Field(default=None, ge=1, le=3600)
    phone_orientation: str = Field(min_length=2, max_length=120)
    sample_count: int = Field(ge=64, le=2_000_000)
    collected_at: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reference_analysis: VibrationAnalysis
    review_notes: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def file_must_be_relative_data_file(self) -> PhoneRecordingDefinition:
        file_path = Path(self.file)
        if file_path.is_absolute() or ".." in file_path.parts:
            raise ValueError("recording file 必须是数据包内的安全相对路径。")
        if not (self.file.endswith(".csv") or self.file.endswith(".csv.gz")):
            raise ValueError("recording file 必须是 .csv 或 .csv.gz。")
        return self


class PhoneDatasetExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    purpose: Literal["measurement_integrity", "diagnostic_comparison"]
    expected_disposition: Literal[
        "analysis_only",
        "not_ready_for_causal_conclusion",
        "ready_for_comparison",
    ]
    min_conditions: int = Field(default=2, ge=1, le=20)
    min_replicates_per_condition: int = Field(default=3, ge=1, le=20)
    expected_flags: list[
        Literal[
            "single_condition",
            "insufficient_replicates",
            "unpaired_recording",
            "orientation_not_recorded",
            "protocol_mismatch",
        ]
    ] = Field(default_factory=list)
    allowed_conclusions: list[str] = Field(min_length=1, max_length=20)
    prohibited_conclusions: list[str] = Field(min_length=1, max_length=20)
    required_followup: list[str] = Field(default_factory=list, max_length=20)


class PhoneDatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,79}$")
    title: str = Field(min_length=3, max_length=160)
    description: str = Field(min_length=10, max_length=1000)
    provenance: PhoneDatasetProvenance
    scenario: PhoneDatasetScenario
    recordings: list[PhoneRecordingDefinition] = Field(min_length=1, max_length=100)
    expectation: PhoneDatasetExpectation

    @model_validator(mode="after")
    def recording_ids_are_unique(self) -> PhoneDatasetManifest:
        recording_ids = [item.recording_id for item in self.recordings]
        if len(recording_ids) != len(set(recording_ids)):
            raise ValueError("recording_id 必须唯一。")
        return self


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _summarize_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(checks)
    passed = sum(bool(item["passed"]) for item in checks)
    return {
        "passed": passed,
        "total": total,
        "score_percent": round(100.0 * passed / max(total, 1), 2),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_recording_path(pack_dir: Path, relative_path: str) -> Path:
    root = pack_dir.resolve()
    resolved = (root / relative_path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("recording file 逃逸出数据包目录。")
    return resolved


def load_phone_dataset(pack_dir: Path) -> PhoneDatasetManifest:
    manifest_path = pack_dir / PHONE_DATASET_MANIFEST
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return PhoneDatasetManifest.model_validate(payload)


def read_phone_recording(
    pack_dir: Path,
    recording: PhoneRecordingDefinition,
) -> SessionUpload:
    recording_path = _resolve_recording_path(pack_dir, recording.file)
    if not recording_path.is_file():
        raise FileNotFoundError(f"找不到 recording file：{recording.file}")
    observed_sha = _sha256(recording_path)
    if observed_sha != recording.sha256:
        raise ValueError(
            f"recording checksum 不匹配：expected={recording.sha256}, observed={observed_sha}"
        )

    raw_handle = recording_path.open("rb")
    binary_handle: io.BufferedReader | gzip.GzipFile
    if recording_path.suffix == ".gz":
        binary_handle = gzip.GzipFile(fileobj=raw_handle, mode="rb")
    else:
        binary_handle = raw_handle
    samples: list[AccelerationSample] = []
    try:
        with io.TextIOWrapper(binary_handle, encoding="utf-8", newline="") as text_handle:
            reader = csv.DictReader(text_handle)
            if tuple(reader.fieldnames or ()) != REQUIRED_SAMPLE_COLUMNS:
                raise ValueError(
                    "CSV 列必须严格为：" + ",".join(REQUIRED_SAMPLE_COLUMNS)
                )
            for row_number, row in enumerate(reader, start=2):
                values = [float(row[column]) for column in REQUIRED_SAMPLE_COLUMNS]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"CSV 第 {row_number} 行包含非有限数值。")
                samples.append(
                    AccelerationSample(
                        timestamp_ms=values[0],
                        x=values[1],
                        y=values[2],
                        z=values[3],
                    )
                )
    finally:
        if not raw_handle.closed:
            raw_handle.close()
    if len(samples) != recording.sample_count:
        raise ValueError(
            f"sample_count 不匹配：manifest={recording.sample_count}, file={len(samples)}"
        )
    return SessionUpload(
        label=recording.label,
        device=recording.device_label,
        notes=(
            f"Phone dataset replay; recording={recording.recording_id}; "
            f"condition={recording.condition_name}; source={recording.capture_source}."
        ),
        samples=samples,
    )


def _analysis_matches_reference(
    observed: VibrationAnalysis,
    reference: VibrationAnalysis,
) -> tuple[bool, str]:
    numeric_fields = (
        "duration_s",
        "sampling_rate_hz",
        "rms_acceleration_m_s2",
        "peak_to_peak_m_s2",
        "dominant_frequency_hz",
        "spectral_snr_db",
    )
    differences = {
        field: abs(float(getattr(observed, field)) - float(getattr(reference, field)))
        for field in numeric_fields
    }
    tolerances = {
        "duration_s": 0.002,
        "sampling_rate_hz": 0.01,
        "rms_acceleration_m_s2": 0.00001,
        "peak_to_peak_m_s2": 0.00001,
        "dominant_frequency_hz": 0.0001,
        "spectral_snr_db": 0.002,
    }
    passed = (
        observed.sample_count == reference.sample_count
        and observed.selected_axis == reference.selected_axis
        and observed.confidence == reference.confidence
        and observed.warnings == reference.warnings
        and all(differences[field] <= tolerances[field] for field in numeric_fields)
    )
    return passed, json.dumps(differences, ensure_ascii=False)


def _dataset_flags(
    manifest: PhoneDatasetManifest,
    analyses: list[VibrationAnalysis],
) -> list[str]:
    flags: set[str] = set()
    conditions = Counter(item.condition_name for item in manifest.recordings)
    required_conditions = max(2, manifest.expectation.min_conditions)
    required_replicates = max(3, manifest.expectation.min_replicates_per_condition)
    if len(conditions) < required_conditions:
        flags.add("single_condition")
    if any(
        count < required_replicates
        for count in conditions.values()
    ):
        flags.add("insufficient_replicates")
    if any(item.condition_role in {"unpaired", "unpaired_control"} for item in manifest.recordings):
        flags.add("unpaired_recording")
    if any(
        item.phone_orientation.strip().casefold() in {"unknown", "not recorded", "未记录"}
        for item in manifest.recordings
    ):
        flags.add("orientation_not_recorded")
    if len(analyses) > 1:
        rates = [item.sampling_rate_hz for item in analyses]
        durations = [item.duration_s for item in analyses]
        orientations = {
            item.phone_orientation.strip().casefold()
            for item in manifest.recordings
            if item.phone_orientation.strip().casefold()
            not in {"unknown", "not recorded", "未记录"}
        }
        if (
            max(rates) / max(min(rates), 1e-9) > 1.05
            or max(durations) / max(min(durations), 1e-9) > 1.25
            or len(orientations) > 1
        ):
            flags.add("protocol_mismatch")
    return sorted(flags)


def evaluate_phone_dataset(
    pack_dir: Path,
    *,
    replay_repeat: int = 3,
) -> dict[str, Any]:
    if replay_repeat < 1:
        raise ValueError("replay_repeat 必须至少为 1。")
    manifest = load_phone_dataset(pack_dir)
    checks: list[dict[str, Any]] = []
    recording_results: list[dict[str, Any]] = []
    analyses: list[VibrationAnalysis] = []
    for recording in manifest.recordings:
        try:
            upload = read_phone_recording(pack_dir, recording)
            repeated = [analyze_acceleration(upload.samples) for _ in range(replay_repeat)]
            analysis = repeated[0]
            analyses.append(analysis)
            deterministic = all(
                item.model_dump() == analysis.model_dump() for item in repeated[1:]
            )
            reference_matches, difference_detail = _analysis_matches_reference(
                analysis,
                recording.reference_analysis,
            )
            recording_checks = [
                _check("file_integrity", True, f"sha256={recording.sha256}"),
                _check(
                    "deterministic_replay",
                    deterministic,
                    f"repeat={replay_repeat}",
                ),
                _check(
                    "reference_analysis_regression",
                    reference_matches,
                    difference_detail,
                ),
            ]
            recording_results.append(
                {
                    "recording_id": recording.recording_id,
                    "condition_name": recording.condition_name,
                    "analysis": analysis.model_dump(),
                    "checks": recording_checks,
                    "summary": _summarize_checks(recording_checks),
                }
            )
            checks.extend(
                {**item, "name": f"{recording.recording_id}:{item['name']}"}
                for item in recording_checks
            )
        except Exception as exc:  # noqa: BLE001 - data validation must record all faults
            failed = _check(
                f"{recording.recording_id}:load_and_analyze",
                False,
                f"{type(exc).__name__}: {str(exc)[:500]}",
            )
            checks.append(failed)
            recording_results.append(
                {
                    "recording_id": recording.recording_id,
                    "condition_name": recording.condition_name,
                    "error": failed["detail"],
                    "checks": [failed],
                    "summary": _summarize_checks([failed]),
                }
            )

    observed_flags = _dataset_flags(manifest, analyses)
    expected_flags = sorted(manifest.expectation.expected_flags)
    flags_match = expected_flags == observed_flags
    valid_recordings = len(analyses) == len(manifest.recordings)
    condition_counts = Counter(item.condition_name for item in manifest.recordings)
    required_conditions = max(2, manifest.expectation.min_conditions)
    required_replicates = max(3, manifest.expectation.min_replicates_per_condition)
    causal_ready = (
        valid_recordings
        and manifest.expectation.purpose == "diagnostic_comparison"
        and manifest.expectation.expected_disposition == "ready_for_comparison"
        and len(condition_counts) >= required_conditions
        and all(
            count >= required_replicates
            for count in condition_counts.values()
        )
        and not observed_flags
    )
    expected_disposition = manifest.expectation.expected_disposition
    disposition_matched = (
        (expected_disposition == "ready_for_comparison" and causal_ready)
        or (
            expected_disposition == "not_ready_for_causal_conclusion"
            and not causal_ready
        )
        or (expected_disposition == "analysis_only" and valid_recordings)
    )
    dataset_checks = [
        _check(
            "expected_quality_flags",
            flags_match,
            f"expected={expected_flags}, observed={observed_flags}",
        ),
        _check(
            "expected_disposition",
            disposition_matched,
            f"expected={expected_disposition}, causal_ready={causal_ready}",
        ),
        _check(
            "privacy_metadata",
            (
                manifest.provenance.original_session_id_removed
                and manifest.provenance.user_identifier_removed
                and manifest.provenance.network_address_removed
            ),
            "原 Session ID、用户标识和网络地址必须从数据包中移除。",
        ),
    ]
    checks.extend(dataset_checks)
    if causal_ready:
        user_disposition = "可进入 Agent 对照诊断，但仍需结合实验边界解释。"
    elif valid_recordings:
        user_disposition = "可用于单次信号描述；当前不能支持因果比较或故障确诊。"
    else:
        user_disposition = "数据包损坏或格式不完整，不能用于分析。"
    return {
        "dataset_id": manifest.dataset_id,
        "title": manifest.title,
        "collection_kind": manifest.provenance.collection_kind,
        "recording_count": len(manifest.recordings),
        "condition_count": len(condition_counts),
        "total_sample_count": sum(item.sample_count for item in manifest.recordings),
        "observed_quality_flags": observed_flags,
        "causal_ready": causal_ready,
        "expected_disposition": expected_disposition,
        "user_disposition": user_disposition,
        "allowed_conclusions": manifest.expectation.allowed_conclusions,
        "prohibited_conclusions": manifest.expectation.prohibited_conclusions,
        "required_followup": manifest.expectation.required_followup,
        "recordings": recording_results,
        "checks": checks,
        "summary": _summarize_checks(checks),
    }


def run_phone_dataset_harness(
    root: Path,
    *,
    replay_repeat: int = 3,
) -> dict[str, Any]:
    def is_acceleration_v1_pack(path: Path) -> bool:
        manifest_path = path / PHONE_DATASET_MANIFEST
        if not manifest_path.is_file():
            return False
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return payload.get("schema_version") == PHONE_DATASET_SCHEMA_VERSION

    pack_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and not path.name.startswith("_")
        and is_acceleration_v1_pack(path)
    )
    datasets = [
        evaluate_phone_dataset(path, replay_repeat=replay_repeat) for path in pack_dirs
    ]
    checks = [
        {
            **item,
            "name": f"{dataset['dataset_id']}:{item['name']}",
        }
        for dataset in datasets
        for item in dataset["checks"]
    ]
    real_recordings = sum(item["recording_count"] for item in datasets)
    causal_ready_count = sum(bool(item["causal_ready"]) for item in datasets)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "phone-replay",
        "root": str(root),
        "replay_repeat": replay_repeat,
        "dataset_count": len(datasets),
        "real_recording_count": real_recordings,
        "causal_ready_dataset_count": causal_ready_count,
        "causal_readiness_rate": round(
            causal_ready_count / max(len(datasets), 1),
            4,
        ),
        "market_data_gate": {
            "passed": len(datasets) >= 3 and causal_ready_count >= 3 and real_recordings >= 18,
            "required": "至少 3 个可对照场景，每场景 2 个条件且每条件至少 3 次真实手机重复测量。",
        },
        "datasets": datasets,
        "checks": checks,
        "summary": _summarize_checks(checks),
    }


def render_phone_dataset_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# PocketLab Real Phone Dataset Replay Report",
        "",
        f"- Generated: `{result['generated_at']}`",
        f"- Dataset packs: **{result['dataset_count']}**",
        f"- Real phone recordings: **{result['real_recording_count']}**",
        f"- Harness score: **{summary['score_percent']:.2f}%** ({summary['passed']}/{summary['total']})",
        f"- Causal-ready datasets: **{result['causal_ready_dataset_count']}**",
        f"- Market data gate: **{'PASS' if result['market_data_gate']['passed'] else 'NOT YET'}**",
        "",
        "## Dataset results",
        "",
        "| Dataset | Recordings | Conditions | Replay | Causal ready |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in result["datasets"]:
        lines.append(
            f"| `{dataset['dataset_id']}` | {dataset['recording_count']} | "
            f"{dataset['condition_count']} | {dataset['summary']['score_percent']:.2f}% | "
            f"{'yes' if dataset['causal_ready'] else 'no'} |"
        )
    for dataset in result["datasets"]:
        lines.extend(
            [
                "",
                f"## {dataset['title']}",
                "",
                f"- User disposition: {dataset['user_disposition']}",
                f"- Quality flags: `{json.dumps(dataset['observed_quality_flags'], ensure_ascii=False)}`",
                "- Allowed conclusions:",
                *[f"  - {item}" for item in dataset["allowed_conclusions"]],
                "- Prohibited conclusions:",
                *[f"  - {item}" for item in dataset["prohibited_conclusions"]],
                "- Required follow-up:",
                *[f"  - {item}" for item in dataset["required_followup"]],
            ]
        )
        for recording in dataset["recordings"]:
            analysis = recording.get("analysis")
            if analysis:
                lines.append(
                    f"- `{recording['recording_id']}`: "
                    f"{analysis['duration_s']:.3f}s, {analysis['sampling_rate_hz']:.2f}Hz, "
                    f"RMS={analysis['rms_acceleration_m_s2']:.6f}m/s², "
                    f"dominant={analysis['dominant_frequency_hz']:.4f}Hz, "
                    f"confidence={analysis['confidence']}"
                )
    lines.extend(
        [
            "",
            "## Market-readiness boundary",
            "",
            result["market_data_gate"]["required"],
            (
                "当前分数只证明数据包完整、回放确定且系统正确限制结论；"
                "没有配对重复实验时，100% Harness 分数不等于已具备故障诊断效度。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_phone_harness_result(
    result: dict[str, Any],
    output_json: Path,
) -> tuple[Path, Path]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_md = output_json.with_suffix(".md")
    output_md.write_text(render_phone_dataset_report(result), encoding="utf-8")
    return output_json, output_md


def _scrub_network_addresses(text: str) -> str:
    return re.sub(
        r"https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/\S*)?",
        "[NETWORK_ADDRESS_REMOVED]",
        text,
    )


def _write_recording_csv_gz(path: Path, samples: list[AccelerationSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw_handle,
        gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle,
        io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="") as text_handle,
    ):
        writer = csv.writer(text_handle, lineterminator="\n")
        writer.writerow(REQUIRED_SAMPLE_COLUMNS)
        for sample in samples:
            writer.writerow((sample.timestamp_ms, sample.x, sample.y, sample.z))


def export_phone_session(
    *,
    database_path: Path,
    session_id: str,
    output_dir: Path,
    dataset_id: str,
    consent_basis: str,
) -> PhoneDatasetManifest:
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    database_uri = f"file:{database_path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT upload_json, analysis_json, created_at FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        upload = SessionUpload.model_validate_json(row["upload_json"])
        if not upload.device.casefold().startswith("phyphox"):
            raise ValueError("只允许导出明确标记为 phyphox 的真实手机 Session。")

        matching_case: dict[str, Any] | None = None
        matching_evidence: dict[str, Any] | None = None
        matching_task: dict[str, Any] | None = None
        for case_row in connection.execute("SELECT case_json FROM diagnostic_cases"):
            case = json.loads(case_row["case_json"])
            evidence = next(
                (
                    item
                    for item in case.get("evidence", [])
                    if item.get("session_id") == session_id
                ),
                None,
            )
            if evidence:
                tasks = {
                    item["task_id"]: item for item in case.get("completed_tasks", [])
                }
                matching_case = case
                matching_evidence = evidence
                matching_task = tasks.get(evidence["task_id"])
                break
    finally:
        connection.close()

    output_dir.mkdir(parents=True, exist_ok=False)
    recording_id = "recording-01"
    relative_file = f"recordings/{recording_id}.csv.gz"
    recording_path = output_dir / relative_file
    _write_recording_csv_gz(recording_path, upload.samples)
    analysis = analyze_acceleration(upload.samples)
    task_kind = matching_task.get("task_kind") if matching_task else None
    condition_role = "unpaired_control" if task_kind == "control" else "unpaired"
    expected_flags = [
        "single_condition",
        "insufficient_replicates",
        "unpaired_recording",
        "orientation_not_recorded",
    ]
    case_title = matching_case.get("title") if matching_case else upload.label
    problem_statement = (
        matching_case.get("problem_statement")
        if matching_case
        else f"复查手机采集记录 {upload.label} 的信号质量与可解释边界。"
    )
    environment = (
        matching_case.get("context") if matching_case else "采集环境未完整记录。"
    )
    phone_placement = environment or "未记录"
    intended_variable = (
        matching_task.get("variable_to_change", "仅记录当前工况")
        if matching_task
        else "仅记录当前工况"
    )
    controls = matching_task.get("controlled_variables", []) if matching_task else []
    experiment_match = re.search(r"实验=([^；]+)", upload.notes)
    requested_match = re.search(r"请求时长=([\d.]+)s", upload.notes)
    review_notes = [
        "这是从 PocketLab 数据库只读导出的真实 phyphox 加速度记录。",
        "数据包只有一个真实工况且没有重复测量，不能与模拟基线混合作因果比较。",
    ]
    if matching_evidence:
        review_notes.append(
            "原诊断流程将该证据标为需复查；导出包不把 Agent 判断当作物理真值。"
        )
    manifest = PhoneDatasetManifest(
        schema_version=PHONE_DATASET_SCHEMA_VERSION,
        dataset_id=dataset_id,
        title=f"{case_title} · 真实手机采集完整性样本",
        description=(
            "真实 phyphox 三轴加速度原始记录，用于验证数据包完整性、确定性回放和"
            "不充分证据的结论限制；不是已完成的故障确诊数据集。"
        ),
        provenance=PhoneDatasetProvenance(
            collection_kind="self_collected_phone",
            capture_app="phyphox remote access",
            exported_at=datetime.now(UTC).isoformat(),
            consent_basis=consent_basis,
            license_note="仅限 PocketLab 项目开发、评测与演示，不对外再分发。",
        ),
        scenario=PhoneDatasetScenario(
            problem_statement=problem_statement,
            environment=_scrub_network_addresses(environment or "未记录"),
            phone_placement=phone_placement,
            intended_variable=intended_variable,
            controlled_variables=controls,
        ),
        recordings=[
            PhoneRecordingDefinition(
                recording_id=recording_id,
                file=relative_file,
                label=upload.label,
                condition_name=(
                    matching_task.get("title", "当前实测工况")
                    if matching_task
                    else "当前实测工况"
                ),
                condition_role=condition_role,
                variable_value=intended_variable,
                capture_source="phyphox_remote",
                device_label=upload.device,
                experiment_title=(
                    experiment_match.group(1) if experiment_match else "phyphox acceleration"
                ),
                requested_duration_s=(
                    float(requested_match.group(1)) if requested_match else None
                ),
                phone_orientation="未记录",
                sample_count=len(upload.samples),
                collected_at=row["created_at"],
                sha256=_sha256(recording_path),
                reference_analysis=analysis,
                review_notes=review_notes,
            )
        ],
        expectation=PhoneDatasetExpectation(
            purpose="measurement_integrity",
            expected_disposition="not_ready_for_causal_conclusion",
            min_conditions=2,
            min_replicates_per_condition=3,
            expected_flags=expected_flags,
            allowed_conclusions=[
                "描述本次记录的采样率、时长、RMS、主频、峰峰值和算法置信度。",
                "指出当前只有单工况单次测量，并给出重新采集方案。",
            ],
            prohibited_conclusions=[
                "把本次单条记录解释为洗衣机故障已经确诊。",
                "把真实手机记录与模拟基线直接比较后宣称衣物分布造成变化。",
            ],
            required_followup=[
                "同一手机、位置、朝向、采样率和时长下重新采集基线与对照。",
                "每个条件至少重复 3 次，并记录洗衣机转速、衣物总量和现场干扰。",
                "检查手机是否被碰动或滑移，并保留现场观察。",
            ],
        ),
    )
    (output_dir / PHONE_DATASET_MANIFEST).write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {manifest.title}",
                "",
                manifest.description,
                "",
                "- 来源：项目所有者现有 SQLite 数据库中的真实 phyphox Session，只读导出。",
                "- 隐私：原 Session ID、账号标识和 phyphox 局域网地址未写入数据包。",
                "- 用途：回放信号处理、验证数据完整性、测试系统能否拒绝越界结论。",
                "- 边界：只有一个真实工况，不能用于证明故障原因或对照效应。",
                "",
                "运行：",
                "",
                "```powershell",
                "uv run python scripts/run_agent_harness.py --mode phone-replay --repeat 5",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest
