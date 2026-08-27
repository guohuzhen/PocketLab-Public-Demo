from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import uuid4

from pocketlab.analyzers import analyze_sensor_recording
from pocketlab.auth import get_current_user_id
from pocketlab.persistence import DEFAULT_USER_ID, SQLiteDatabase, database, utc_now
from pocketlab.schemas import SessionUpload, VibrationAnalysis
from pocketlab.sensor_models import SensorAnalysis, SensorRecordingUpload
from pocketlab.signal_processing import analyze_acceleration


@dataclass(frozen=True)
class StoredSession:
    session_id: str
    upload: SessionUpload
    analysis: VibrationAnalysis
    created_at: str


@dataclass(frozen=True)
class StoredSensorRecording:
    session_id: str
    upload: SensorRecordingUpload
    analysis: SensorAnalysis
    created_at: str


class SessionStore:
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

    def create(self, upload: SessionUpload) -> StoredSession:
        user_id = self._active_user_id
        session = StoredSession(
            session_id=uuid4().hex[:12],
            upload=upload,
            analysis=analyze_acceleration(upload.samples),
            created_at=utc_now(),
        )
        self._database.execute(
            """
            INSERT INTO sessions(
                session_id, user_id, upload_json, analysis_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                session.session_id,
                user_id,
                session.upload.model_dump_json(),
                session.analysis.model_dump_json(),
                session.created_at,
            ),
        )
        return session

    def create_sensor_recording(
        self,
        upload: SensorRecordingUpload,
    ) -> StoredSensorRecording:
        user_id = self._active_user_id
        recording = StoredSensorRecording(
            session_id=uuid4().hex[:12],
            upload=upload,
            analysis=analyze_sensor_recording(upload),
            created_at=utc_now(),
        )
        self._database.execute(
            """
            INSERT INTO sessions(
                session_id, user_id, upload_json, analysis_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                recording.session_id,
                user_id,
                recording.upload.model_dump_json(),
                recording.analysis.model_dump_json(),
                recording.created_at,
            ),
        )
        return recording

    def create_sensor_recordings(
        self,
        uploads: tuple[SensorRecordingUpload, ...],
    ) -> tuple[StoredSensorRecording, ...]:
        """Analyze every upload first, then persist the synchronized set atomically."""

        if not 2 <= len(uploads) <= 3:
            raise ValueError("synchronized recording batches require 2 to 3 uploads")
        user_id = self._active_user_id
        created_at = utc_now()
        recordings = tuple(
            StoredSensorRecording(
                session_id=uuid4().hex[:12],
                upload=upload,
                analysis=analyze_sensor_recording(upload),
                created_at=created_at,
            )
            for upload in uploads
        )
        if len({item.upload.sensor for item in recordings}) != len(recordings):
            raise ValueError("synchronized recording batches cannot duplicate sensors")
        attestations = {
            (
                item.upload.provenance.capture_group_id,
                item.upload.provenance.clock_id,
                item.upload.provenance.maximum_alignment_error_ms,
                item.upload.provenance.alignment_method,
            )
            for item in recordings
        }
        if (
            len(attestations) != 1
            or any(value is None for value in next(iter(attestations)))
        ):
            raise ValueError(
                "synchronized recording batches require one identical shared attestation"
            )
        with self._database.transaction() as connection:
            connection.executemany(
                """
                INSERT INTO sessions(
                    session_id, user_id, upload_json, analysis_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.session_id,
                        user_id,
                        item.upload.model_dump_json(),
                        item.analysis.model_dump_json(),
                        item.created_at,
                    )
                    for item in recordings
                ],
            )
        return recordings

    def get(self, session_id: str) -> StoredSession:
        user_id = self._active_user_id
        row = self._database.fetch_one(
            "SELECT * FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )
        if row is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        if self._is_v2_row(row):
            raise KeyError(f"Session is not a legacy acceleration session: {session_id}")
        return self._from_row(row)

    def get_sensor_recording(self, session_id: str) -> StoredSensorRecording:
        user_id = self._active_user_id
        row = self._database.fetch_one(
            "SELECT * FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )
        if row is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        if not self._is_v2_row(row):
            raise KeyError(f"Session is not a v2 sensor recording: {session_id}")
        return self._sensor_from_row(row)

    def list(self, *, limit: int = 200) -> list[StoredSession]:
        user_id = self._active_user_id
        rows = self._database.fetch_all(
            """
            SELECT * FROM sessions
            WHERE user_id = ?
              AND COALESCE(json_extract(upload_json, '$.schema_version'), '1.0') != '2.0'
            ORDER BY created_at DESC LIMIT ?
            """,
            (user_id, limit),
        )
        return [self._from_row(row) for row in rows]

    def list_sensor_recordings(self, *, limit: int = 200) -> list[StoredSensorRecording]:
        user_id = self._active_user_id
        rows = self._database.fetch_all(
            """
            SELECT * FROM sessions
            WHERE user_id = ? AND json_extract(upload_json, '$.schema_version') = '2.0'
            ORDER BY created_at DESC LIMIT ?
            """,
            (user_id, limit),
        )
        return [self._sensor_from_row(row) for row in rows]

    def delete(self, session_id: str) -> None:
        user_id = self._active_user_id
        self._database.execute(
            "DELETE FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )

    def count(self) -> int:
        user_id = self._active_user_id
        row = self._database.fetch_one(
            "SELECT COUNT(*) AS count FROM sessions WHERE user_id = ?",
            (user_id,),
        )
        return int(row["count"]) if row is not None else 0

    def clear(self) -> None:
        user_id = self._active_user_id
        self._database.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    @staticmethod
    def _from_row(row: object) -> StoredSession:
        return StoredSession(
            session_id=row["session_id"],
            upload=SessionUpload.model_validate_json(row["upload_json"]),
            analysis=VibrationAnalysis.model_validate_json(row["analysis_json"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _sensor_from_row(row: object) -> StoredSensorRecording:
        return StoredSensorRecording(
            session_id=row["session_id"],
            upload=SensorRecordingUpload.model_validate_json(row["upload_json"]),
            analysis=SensorAnalysis.model_validate_json(row["analysis_json"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _is_v2_row(row: object) -> bool:
        try:
            payload = json.loads(row["upload_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("stored session upload JSON is invalid") from exc
        return isinstance(payload, dict) and payload.get("schema_version") == "2.0"


session_store = SessionStore(database, user_id=None)
