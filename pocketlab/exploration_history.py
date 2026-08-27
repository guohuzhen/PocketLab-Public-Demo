from __future__ import annotations

import sqlite3
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from pocketlab.auth import get_current_user_id
from pocketlab.general_exploration_state import GeneralExperimentCase
from pocketlab.investigation_models import InvestigationCase
from pocketlab.persistence import DEFAULT_USER_ID, SQLiteDatabase, database, utc_now
from pocketlab.public_light_models import PublicLightExploreResult
from pocketlab.public_pressure_agent_models import PublicPressureExploreResult
from pocketlab.public_sensor_agent_models import PublicSensorExploreResult
from pocketlab.sensor_models import SensorKind


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


ExplorationHistoryKind = Literal[
    "investigation",
    "general_exploration",
    "public_replay",
]
ExplorationHistoryStatus = Literal[
    "in_progress",
    "completed",
    "limited",
    "unsupported",
    "inconclusive",
]
PublicExplorationResult = Annotated[
    PublicLightExploreResult | PublicPressureExploreResult | PublicSensorExploreResult,
    Field(union_mode="left_to_right"),
]


class ExplorationHistoryItem(_StrictModel):
    record_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    record_kind: ExplorationHistoryKind
    protocol_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    protocol_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    title: str = Field(min_length=1, max_length=180)
    research_question: str = Field(min_length=5, max_length=1200)
    primary_sensor: SensorKind
    status: ExplorationHistoryStatus
    data_source: Literal[
        "phone_or_import",
        "public_replay",
        "simulated_rehearsal",
    ]
    compiler_source: Literal["manual", "bounded_agent_compiler"] | None = None
    resumable: bool
    superseded_by_case_id: str | None = Field(default=None, max_length=128)
    evidence_count: int = Field(ge=0, le=128)
    tool_count: int = Field(ge=0, le=256)
    artifact_count: int = Field(ge=0, le=32)
    report_summary: str | None = Field(default=None, max_length=1600)
    created_at: str = Field(min_length=10, max_length=64)
    updated_at: str = Field(min_length=10, max_length=64)


class PublicExplorationHistoryDetail(_StrictModel):
    history: ExplorationHistoryItem
    result: PublicExplorationResult


def investigation_history_item(
    case: InvestigationCase,
    *,
    created_at: str,
    updated_at: str,
) -> ExplorationHistoryItem:
    terminal = case.status in {"completed_with_conclusion", "completed_inconclusive"}
    status: ExplorationHistoryStatus
    if case.status == "completed_with_conclusion":
        status = "completed"
    elif case.status == "completed_inconclusive":
        status = "inconclusive"
    else:
        status = "in_progress"
    summary = case.report.conclusion if case.report is not None else None
    return ExplorationHistoryItem(
        record_id=case.case_id,
        record_kind="investigation",
        protocol_id=case.protocol.protocol_id,
        protocol_version=case.protocol.protocol_version,
        title=case.title,
        research_question=case.research_question,
        primary_sensor=case.protocol.primary_sensor,
        status=status,
        data_source="phone_or_import",
        resumable=not terminal,
        evidence_count=len(case.evidence),
        tool_count=len(case.tool_trace),
        artifact_count=len(case.artifacts),
        report_summary=summary,
        created_at=created_at,
        updated_at=updated_at,
    )


def general_exploration_history_item(
    case: GeneralExperimentCase,
    *,
    created_at: str,
    updated_at: str,
) -> ExplorationHistoryItem:
    terminal = case.status not in {"collecting", "awaiting_user_decision"}
    superseded = case.superseded_by_case_id is not None
    status: ExplorationHistoryStatus
    if case.status == "completed_descriptive":
        status = "completed"
    elif case.status == "completed_inconclusive":
        status = "inconclusive"
    else:
        status = "in_progress"
    primary_sensor = next(
        item.sensor for item in case.protocol.sensors if item.role == "primary"
    )
    source = (
        "simulated_rehearsal"
        if case.protocol.selected_sources == ("protocol_emulator",)
        else
        "public_replay"
        if case.protocol.selected_sources == ("public_replay",)
        else "phone_or_import"
    )
    return ExplorationHistoryItem(
        record_id=case.case_id,
        record_kind="general_exploration",
        protocol_id=case.protocol.protocol_id,
        protocol_version=case.protocol.protocol_version,
        title=case.protocol.title,
        research_question=case.protocol.question,
        primary_sensor=primary_sensor,
        status=status,
        data_source=source,
        compiler_source=case.compiler_provenance.source,
        resumable=not terminal and not superseded,
        superseded_by_case_id=case.superseded_by_case_id,
        evidence_count=len(case.evidence),
        tool_count=len(case.planner_trace),
        artifact_count=0,
        report_summary=(case.report.answer if case.report is not None else None),
        created_at=created_at,
        updated_at=updated_at,
    )


class ExplorationHistoryStore:
    """Persist privacy-safe public Agent results for one authenticated user."""

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

    def save_public(
        self,
        result: PublicLightExploreResult
        | PublicPressureExploreResult
        | PublicSensorExploreResult,
    ) -> ExplorationHistoryItem:
        kind = self._result_kind(result)
        now = utc_now()
        try:
            self._database.execute(
                """
                INSERT INTO public_exploration_runs(
                    run_id, user_id, result_kind, result_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result.run_id,
                    self._active_user_id,
                    kind,
                    result.model_dump_json(),
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("探索运行编号已经存在，历史记录未被覆盖。") from exc
        return self._public_item(result, created_at=now, updated_at=now)

    def list_public(self, *, limit: int = 100) -> list[ExplorationHistoryItem]:
        rows = self._database.fetch_all(
            """
            SELECT result_kind, result_json, created_at, updated_at
            FROM public_exploration_runs
            WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?
            """,
            (self._active_user_id, limit),
        )
        return [
            self._public_item(
                self._load_result(row["result_kind"], row["result_json"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def get_public(self, run_id: str) -> PublicExplorationHistoryDetail:
        row = self._database.fetch_one(
            """
            SELECT result_kind, result_json, created_at, updated_at
            FROM public_exploration_runs
            WHERE run_id = ? AND user_id = ?
            """,
            (run_id, self._active_user_id),
        )
        if row is None:
            raise KeyError(f"Unknown public exploration run: {run_id}")
        result = self._load_result(row["result_kind"], row["result_json"])
        return PublicExplorationHistoryDetail(
            history=self._public_item(
                result,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            ),
            result=result,
        )

    def clear_public(self) -> None:
        self._database.execute(
            "DELETE FROM public_exploration_runs WHERE user_id = ?",
            (self._active_user_id,),
        )

    @staticmethod
    def _result_kind(
        result: PublicLightExploreResult
        | PublicPressureExploreResult
        | PublicSensorExploreResult,
    ) -> Literal["light", "pressure", "sensor"]:
        if isinstance(result, PublicLightExploreResult):
            return "light"
        if isinstance(result, PublicPressureExploreResult):
            return "pressure"
        if isinstance(result, PublicSensorExploreResult):
            return "sensor"
        raise TypeError("Unsupported public exploration result type")

    @staticmethod
    def _load_result(
        kind: str,
        payload: str,
    ) -> PublicLightExploreResult | PublicPressureExploreResult | PublicSensorExploreResult:
        if kind == "light":
            return PublicLightExploreResult.model_validate_json(payload)
        if kind == "pressure":
            return PublicPressureExploreResult.model_validate_json(payload)
        if kind == "sensor":
            return PublicSensorExploreResult.model_validate_json(payload)
        raise ValueError(f"Unknown persisted exploration result kind: {kind}")

    @staticmethod
    def _public_item(
        result: PublicLightExploreResult
        | PublicPressureExploreResult
        | PublicSensorExploreResult,
        *,
        created_at: str,
        updated_at: str,
    ) -> ExplorationHistoryItem:
        if isinstance(result, PublicLightExploreResult):
            sensor: SensorKind = "light"
        elif isinstance(result, PublicPressureExploreResult):
            sensor = "pressure"
        else:
            sensor = result.sensor
        return ExplorationHistoryItem(
            record_id=result.run_id,
            record_kind="public_replay",
            protocol_id=result.protocol_id,
            protocol_version=result.protocol_version,
            title=result.report.title,
            research_question=result.research_question,
            primary_sensor=sensor,
            status=result.execution_status,
            data_source="public_replay",
            resumable=False,
            evidence_count=len(result.evidence),
            tool_count=len(result.tool_trace),
            artifact_count=0,
            report_summary=result.report.summary,
            created_at=created_at,
            updated_at=updated_at,
        )


exploration_history_store = ExplorationHistoryStore(database, user_id=None)
