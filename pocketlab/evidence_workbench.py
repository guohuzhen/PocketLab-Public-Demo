from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from pocketlab.auth import get_current_user_id
from pocketlab.diagnostic_evidence import DiagnosticRecordingView
from pocketlab.persistence import DEFAULT_USER_ID, SQLiteDatabase, database, utc_now
from pocketlab.sensor_models import AnalysisMetric, SensorKind


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceRecordAudit(_StrictModel):
    recording_id: str
    label: str
    sensor: SensorKind
    source: str
    source_details: dict[str, str] = Field(default_factory=dict)
    analyzer_id: str
    analyzer_version: str
    confidence: Literal["low", "medium", "high"]
    quality_score: int = Field(ge=0, le=100)
    sample_count: int = Field(ge=0)
    duration_s: float = Field(ge=0)
    sampling_rate_hz: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    metrics: list[AnalysisMetric] = Field(default_factory=list)


class EvidenceComparabilityGroup(_StrictModel):
    sensor: SensorKind | None
    recording_ids: list[str]
    status: Literal["direct", "limited", "context_only"]
    shared_metric_keys: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class EvidenceMetricContrast(_StrictModel):
    sensor: SensorKind
    metric_key: str
    metric_label: str
    unit: str
    left_recording_id: str
    right_recording_id: str
    left_value: float
    right_value: float
    absolute_delta: float
    relative_delta_percent: float | None = None


class EvidenceCitation(_StrictModel):
    citation_id: str = Field(pattern=r"^E[1-4]$")
    recording_id: str
    label: str
    sensor: SensorKind
    source: str
    confidence: Literal["low", "medium", "high"]


class EvidenceComparabilityCell(_StrictModel):
    left_recording_id: str
    right_recording_id: str
    status: Literal["same_record", "direct", "limited", "context_only"]
    shared_metric_keys: list[str] = Field(default_factory=list)
    reason: str


class EvidenceChartPoint(_StrictModel):
    citation_id: str = Field(pattern=r"^E[1-4]$")
    recording_id: str
    label: str
    value: float


class EvidenceChartSeries(_StrictModel):
    chart_id: str
    sensor: SensorKind
    metric_key: str
    metric_label: str
    unit: str
    comparability: Literal["direct", "limited"]
    points: list[EvidenceChartPoint] = Field(min_length=2, max_length=4)


class EvidenceQualitySummary(_StrictModel):
    overall_confidence: Literal["low", "medium", "high"]
    high_count: int = Field(ge=0)
    medium_count: int = Field(ge=0)
    low_count: int = Field(ge=0)
    direct_comparison_count: int = Field(ge=0)
    limited_comparison_count: int = Field(ge=0)


class EvidenceWorkbenchReport(_StrictModel):
    report_id: str
    question: str
    answer: str
    analysis_status: Literal["model_generated", "deterministic_only"]
    model: str
    recording_ids: list[str]
    sensor_kinds: list[SensorKind]
    quality: EvidenceQualitySummary
    audits: list[EvidenceRecordAudit]
    comparability: list[EvidenceComparabilityGroup]
    contrasts: list[EvidenceMetricContrast]
    citations: list[EvidenceCitation] = Field(default_factory=list)
    comparability_matrix: list[EvidenceComparabilityCell] = Field(default_factory=list)
    charts: list[EvidenceChartSeries] = Field(default_factory=list)
    boundaries: list[str]
    user_note: str = Field(default="", max_length=2000)
    created_at: str
    updated_at: str


class EvidenceWorkbenchHistoryItem(_StrictModel):
    report_id: str
    question: str
    analysis_status: Literal["model_generated", "deterministic_only"]
    model: str
    recording_count: int
    sensor_kinds: list[SensorKind]
    overall_confidence: Literal["low", "medium", "high"]
    created_at: str
    updated_at: str


class EvidenceWorkbenchNoteUpdate(_StrictModel):
    user_note: str = Field(default="", max_length=2000)


def _quality_score(recording: DiagnosticRecordingView) -> int:
    score = {"high": 94, "medium": 74, "low": 45}[recording.analysis.confidence]
    return max(0, score - min(25, len(recording.analysis.warnings) * 5))


def _metric_map(recording: DiagnosticRecordingView) -> dict[str, AnalysisMetric]:
    return {metric.key: metric for metric in recording.analysis.metrics}


def build_evidence_audit(
    recordings: list[DiagnosticRecordingView],
) -> tuple[
    list[EvidenceRecordAudit],
    list[EvidenceComparabilityGroup],
    list[EvidenceMetricContrast],
    EvidenceQualitySummary,
    list[str],
]:
    audits = [
        EvidenceRecordAudit(
            recording_id=item.session_id,
            label=item.label,
            sensor=item.sensor,
            source=item.provenance_source,
            source_details=item.provenance_details,
            analyzer_id=item.analysis.analyzer_id,
            analyzer_version=item.analysis.analyzer_version,
            confidence=item.analysis.confidence,
            quality_score=_quality_score(item),
            sample_count=item.analysis.sample_count,
            duration_s=item.analysis.duration_s,
            sampling_rate_hz=item.analysis.sampling_rate_hz,
            warnings=list(item.analysis.warnings),
            metrics=list(item.analysis.metrics),
        )
        for item in recordings
    ]
    grouped: dict[SensorKind, list[DiagnosticRecordingView]] = defaultdict(list)
    for item in recordings:
        grouped[item.sensor].append(item)

    comparability: list[EvidenceComparabilityGroup] = []
    contrasts: list[EvidenceMetricContrast] = []
    for sensor, members in grouped.items():
        metric_maps = [_metric_map(item) for item in members]
        shared_keys = set(metric_maps[0]) if metric_maps else set()
        for metric_map in metric_maps[1:]:
            shared_keys &= set(metric_map)
        shared_keys = {
            key
            for key in shared_keys
            if len({metric_map[key].unit for metric_map in metric_maps}) == 1
        }
        reasons: list[str] = []
        if len(members) < 2:
            status: Literal["direct", "limited", "context_only"] = "context_only"
            reasons.append("该传感器只有一条记录，只能描述，不能形成条件对照。")
        elif not shared_keys:
            status = "context_only"
            reasons.append("记录之间没有同名且同单位的注册指标。")
        elif any(item.analysis.confidence == "low" for item in members):
            status = "limited"
            reasons.append("至少一条记录为低置信度，差异只能作为线索。")
        else:
            status = "direct"
            reasons.append("同传感器、同指标且单位一致，可进行数值对照。")
        if len({item.provenance_source for item in members}) > 1:
            if status == "direct":
                status = "limited"
            reasons.append("来源类型不同，采集链差异限制了直接归因。")
        comparability.append(
            EvidenceComparabilityGroup(
                sensor=sensor,
                recording_ids=[item.session_id for item in members],
                status=status,
                shared_metric_keys=sorted(shared_keys),
                reasons=reasons,
            )
        )
        for left, right in combinations(members, 2):
            left_metrics = _metric_map(left)
            right_metrics = _metric_map(right)
            for key in sorted(shared_keys):
                left_metric = left_metrics[key]
                right_metric = right_metrics[key]
                delta = right_metric.value - left_metric.value
                relative = (
                    delta / abs(left_metric.value) * 100
                    if abs(left_metric.value) > 1e-12
                    else None
                )
                contrasts.append(
                    EvidenceMetricContrast(
                        sensor=sensor,
                        metric_key=key,
                        metric_label=left_metric.label,
                        unit=left_metric.unit,
                        left_recording_id=left.session_id,
                        right_recording_id=right.session_id,
                        left_value=left_metric.value,
                        right_value=right_metric.value,
                        absolute_delta=delta,
                        relative_delta_percent=relative,
                    )
                )

    if len(grouped) > 1:
        comparability.append(
            EvidenceComparabilityGroup(
                sensor=None,
                recording_ids=[item.session_id for item in recordings],
                status="context_only",
                reasons=[
                    "不同传感器测量不同物理量，只能组合解释机制，不能直接比较数值大小。"
                ],
            )
        )

    counts = {level: sum(item.analysis.confidence == level for item in recordings) for level in ("high", "medium", "low")}
    overall = "low" if counts["low"] else "medium" if counts["medium"] else "high"
    quality = EvidenceQualitySummary(
        overall_confidence=overall,
        high_count=counts["high"],
        medium_count=counts["medium"],
        low_count=counts["low"],
        direct_comparison_count=sum(item.status == "direct" for item in comparability),
        limited_comparison_count=sum(item.status == "limited" for item in comparability),
    )
    boundaries = [
        "工作台只解释已保存证据，不会改写诊断或探索案例，也不会自动创建新测量。",
        "没有注册阈值或校准标准时，绝对数值不能被标记为正常、异常、合格或不合格。",
    ]
    if any(item.provenance_source in {"public_replay", "test_fixture"} for item in recordings):
        boundaries.append("公开回放或测试夹具只能演示方法，不能冒充用户现场证据。")
    if quality.low_count:
        boundaries.append("至少一条记录为低置信度，模型解释不能把它升级为确定结论。")
    if len(grouped) > 1:
        boundaries.append("跨传感器信息用于机制互证，不进行不同单位之间的数值排序。")
    return audits, comparability, contrasts, quality, boundaries


def build_evidence_presentation(
    audits: list[EvidenceRecordAudit],
    comparability: list[EvidenceComparabilityGroup],
) -> tuple[
    list[EvidenceCitation],
    list[EvidenceComparabilityCell],
    list[EvidenceChartSeries],
]:
    """Build deterministic citation, matrix, and chart contracts.

    The presentation layer never invents a comparison: charts are emitted only
    for a same-sensor group whose registered metric key and unit survived the
    comparability audit.
    """

    citations = [
        EvidenceCitation(
            citation_id=f"E{index}",
            recording_id=item.recording_id,
            label=item.label,
            sensor=item.sensor,
            source=item.source,
            confidence=item.confidence,
        )
        for index, item in enumerate(audits, start=1)
    ]
    citation_by_recording = {item.recording_id: item for item in citations}
    audit_by_recording = {item.recording_id: item for item in audits}
    group_by_sensor = {
        item.sensor: item for item in comparability if item.sensor is not None
    }

    matrix: list[EvidenceComparabilityCell] = []
    for left in audits:
        for right in audits:
            if left.recording_id == right.recording_id:
                status: Literal[
                    "same_record", "direct", "limited", "context_only"
                ] = "same_record"
                shared_keys = sorted(_metric_map_from_audit(left))
                reason = "同一条证据，不构成独立条件对照。"
            elif left.sensor != right.sensor:
                status = "context_only"
                shared_keys = []
                reason = "不同传感器测量不同物理量，只能用于机制互证。"
            else:
                group = group_by_sensor[left.sensor]
                status = group.status
                shared_keys = group.shared_metric_keys
                reason = "；".join(group.reasons)
            matrix.append(
                EvidenceComparabilityCell(
                    left_recording_id=left.recording_id,
                    right_recording_id=right.recording_id,
                    status=status,
                    shared_metric_keys=shared_keys,
                    reason=reason,
                )
            )

    charts: list[EvidenceChartSeries] = []
    for group in comparability:
        if group.sensor is None or group.status == "context_only":
            continue
        for metric_key in sorted(
            group.shared_metric_keys,
            key=_presentation_metric_priority,
        ):
            members = [audit_by_recording[item] for item in group.recording_ids]
            metrics = [_metric_map_from_audit(item)[metric_key] for item in members]
            charts.append(
                EvidenceChartSeries(
                    chart_id=f"chart-{group.sensor}-{metric_key}",
                    sensor=group.sensor,
                    metric_key=metric_key,
                    metric_label=metrics[0].label,
                    unit=metrics[0].unit,
                    comparability=group.status,
                    points=[
                        EvidenceChartPoint(
                            citation_id=citation_by_recording[item.recording_id].citation_id,
                            recording_id=item.recording_id,
                            label=item.label,
                            value=metric.value,
                        )
                        for item, metric in zip(members, metrics, strict=True)
                    ],
                )
            )
            if len(charts) >= 4:
                return citations, matrix, charts
    return citations, matrix, charts


def _metric_map_from_audit(audit: EvidenceRecordAudit) -> dict[str, AnalysisMetric]:
    return {metric.key: metric for metric in audit.metrics}


def _presentation_metric_priority(metric_key: str) -> tuple[int, str]:
    primary_markers = (
        "median",
        "rms",
        "dominant_frequency",
        "mean",
        "magnitude",
        "amplitude",
        "distance",
        "speed",
        "pressure",
        "altitude",
    )
    quality_markers = (
        "sampling",
        "jitter",
        "quantization",
        "coefficient_of_variation",
    )
    if any(marker in metric_key for marker in primary_markers):
        return 0, metric_key
    if any(marker in metric_key for marker in quality_markers):
        return 2, metric_key
    return 1, metric_key


def deterministic_workbench_answer(
    question: str,
    audits: list[EvidenceRecordAudit],
    comparability: list[EvidenceComparabilityGroup],
    quality: EvidenceQualitySummary,
) -> str:
    sensors = "、".join(dict.fromkeys(item.sensor for item in audits))
    direct = quality.direct_comparison_count
    limited = quality.limited_comparison_count
    return (
        f"已完成 {len(audits)} 条证据的确定性审计，涉及 {sensors}。"
        f"整体证据质量为 {quality.overall_confidence}；可直接比较的同类组 {direct} 个，"
        f"受限比较组 {limited} 个。模型解释当前不可用，因此没有补造物理原因。"
        f"请先依据下方指标和可比性审计阅读问题“{question}”，或恢复模型后重新运行以获得机制解释。"
    )


class EvidenceWorkbenchStore:
    def __init__(
        self,
        storage: SQLiteDatabase | None = None,
        *,
        user_id: str | None = DEFAULT_USER_ID,
    ) -> None:
        self._database = storage or SQLiteDatabase(":memory:")
        self._user_id = user_id

    @property
    def _active_user_id(self) -> str:
        return self._user_id or get_current_user_id()

    def save(self, report: EvidenceWorkbenchReport) -> EvidenceWorkbenchReport:
        self._database.execute(
            """
            INSERT INTO evidence_workbench_reports(
                report_id, user_id, report_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                report.report_id,
                self._active_user_id,
                report.model_dump_json(),
                report.created_at,
                report.updated_at,
            ),
        )
        return report

    def create_report(
        self,
        *,
        question: str,
        answer: str,
        analysis_status: Literal["model_generated", "deterministic_only"],
        model: str,
        recording_ids: list[str],
        sensor_kinds: list[SensorKind],
        audits: list[EvidenceRecordAudit],
        comparability: list[EvidenceComparabilityGroup],
        contrasts: list[EvidenceMetricContrast],
        citations: list[EvidenceCitation],
        comparability_matrix: list[EvidenceComparabilityCell],
        charts: list[EvidenceChartSeries],
        quality: EvidenceQualitySummary,
        boundaries: list[str],
    ) -> EvidenceWorkbenchReport:
        now = utc_now()
        return self.save(
            EvidenceWorkbenchReport(
                report_id=f"workbench-{uuid4().hex[:16]}",
                question=question,
                answer=answer,
                analysis_status=analysis_status,
                model=model,
                recording_ids=recording_ids,
                sensor_kinds=sensor_kinds,
                quality=quality,
                audits=audits,
                comparability=comparability,
                contrasts=contrasts,
                citations=citations,
                comparability_matrix=comparability_matrix,
                charts=charts,
                boundaries=boundaries,
                created_at=now,
                updated_at=now,
            )
        )

    def get(self, report_id: str) -> EvidenceWorkbenchReport:
        row = self._database.fetch_one(
            "SELECT report_json FROM evidence_workbench_reports "
            "WHERE report_id = ? AND user_id = ?",
            (report_id, self._active_user_id),
        )
        if row is None:
            raise KeyError(f"Unknown evidence workbench report: {report_id}")
        return EvidenceWorkbenchReport.model_validate_json(row["report_json"])

    def list(self, *, limit: int = 30) -> list[EvidenceWorkbenchHistoryItem]:
        rows = self._database.fetch_all(
            "SELECT report_json FROM evidence_workbench_reports WHERE user_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (self._active_user_id, limit),
        )
        reports = [EvidenceWorkbenchReport.model_validate_json(row["report_json"]) for row in rows]
        return [
            EvidenceWorkbenchHistoryItem(
                report_id=item.report_id,
                question=item.question,
                analysis_status=item.analysis_status,
                model=item.model,
                recording_count=len(item.recording_ids),
                sensor_kinds=item.sensor_kinds,
                overall_confidence=item.quality.overall_confidence,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in reports
        ]

    def update_note(self, report_id: str, note: str) -> EvidenceWorkbenchReport:
        report = self.get(report_id).model_copy(update={"user_note": note, "updated_at": utc_now()})
        self._database.execute(
            "UPDATE evidence_workbench_reports SET report_json = ?, updated_at = ? "
            "WHERE report_id = ? AND user_id = ?",
            (report.model_dump_json(), report.updated_at, report_id, self._active_user_id),
        )
        return report

    def clear(self) -> None:
        self._database.execute(
            "DELETE FROM evidence_workbench_reports WHERE user_id = ?",
            (self._active_user_id,),
        )


def workbench_report_markdown(report: EvidenceWorkbenchReport) -> str:
    lines = [
        f"# PocketLab 证据报告：{report.question}",
        "",
        f"- 报告编号：`{report.report_id}`",
        f"- 生成时间：{report.created_at}",
        f"- 分析模式：{report.analysis_status}",
        f"- 模型：{report.model}",
        f"- 整体证据质量：{report.quality.overall_confidence}",
        "",
        "## Agent 解释",
        "",
        report.answer,
        "",
        "## 证据引用",
        "",
    ]
    lines.extend(
        f"- [{item.citation_id}] `{item.recording_id}` · {item.label} · "
        f"{item.sensor} · {item.source} · {item.confidence}"
        for item in report.citations
    )
    lines.extend(
        [
            "",
        "## 记录审计",
        "",
        ]
    )
    for item in report.audits:
        lines.extend(
            [
                f"### {item.label}",
                "",
                f"- 传感器：{item.sensor}",
                f"- 来源：{item.source}",
                f"- 分析器：{item.analyzer_id} {item.analyzer_version}",
                f"- 质量：{item.confidence}（{item.quality_score}/100）",
            ]
        )
        for metric in item.metrics:
            lines.append(f"- {metric.label}：{metric.value:g} {metric.unit}".rstrip())
        for key, value in item.source_details.items():
            lines.append(f"- 来源字段 {key}：{value}")
        for warning in item.warnings:
            lines.append(f"- 警告：{warning}")
        lines.append("")
    lines.extend(["## 可比性矩阵", ""])
    citation_ids = {
        item.recording_id: item.citation_id for item in report.citations
    }
    for item in report.comparability_matrix:
        left = citation_ids.get(item.left_recording_id, item.left_recording_id)
        right = citation_ids.get(item.right_recording_id, item.right_recording_id)
        metrics = "、".join(item.shared_metric_keys) or "无直接数值指标"
        lines.append(
            f"- [{left}] × [{right}]：{item.status}；{metrics}；{item.reason}"
        )
    lines.extend(["", "## 数值对照", ""])
    if report.contrasts:
        for item in report.contrasts:
            left = citation_ids.get(item.left_recording_id, item.left_recording_id)
            right = citation_ids.get(item.right_recording_id, item.right_recording_id)
            relative = (
                "—"
                if item.relative_delta_percent is None
                else f"{item.relative_delta_percent:g}%"
            )
            lines.append(
                f"- [{right}]−[{left}] {item.metric_label}："
                f"{item.absolute_delta:g} {item.unit}；相对变化 {relative}"
            )
    else:
        lines.append("- 当前没有通过同传感器、同指标、同单位门禁的成对差值。")
    lines.extend(["## 适用边界", ""])
    lines.extend(f"- {item}" for item in report.boundaries)
    if report.user_note:
        lines.extend(["", "## 用户注释", "", report.user_note])
    return "\n".join(lines).strip() + "\n"


evidence_workbench_store = EvidenceWorkbenchStore(database, user_id=None)
