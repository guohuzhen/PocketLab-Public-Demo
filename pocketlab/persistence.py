from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

DEFAULT_USER_ID = "local-user"
DEFAULT_DISPLAY_NAME = "本地实验者"


def utc_now() -> str:
    """Return a stable, sortable UTC timestamp for API and database records."""

    return datetime.now(UTC).isoformat()


def default_database_path() -> Path | str:
    configured = os.getenv("POCKETLAB_DB_PATH", "").strip()
    if configured == ":memory:":
        return configured
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "data" / "pocketlab.sqlite3"


class SQLiteDatabase:
    """Small serialized SQLite boundary shared by the repository-style stores."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = path if path is not None else default_database_path()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(
            str(self.path),
            timeout=10.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._migrate()

    def _migrate(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT,
                    username_key TEXT,
                    password_hash TEXT,
                    display_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    upload_json TEXT NOT NULL,
                    analysis_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user_created
                    ON sessions(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS diagnostic_cases (
                    case_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    case_json TEXT NOT NULL,
                    latest_agent_message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_cases_user_updated
                    ON diagnostic_cases(user_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS investigation_cases (
                    investigation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    case_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_investigations_user_updated
                    ON investigation_cases(user_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS general_exploration_cases (
                    case_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    case_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_general_explorations_user_updated
                    ON general_exploration_cases(user_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS general_compilation_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    draft_sha256 TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    consumed_case_id TEXT,
                    created_at TEXT NOT NULL,
                    consumed_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_general_compilation_receipts_user_created
                    ON general_compilation_receipts(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS general_clarification_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    reservation_token TEXT,
                    reserved_at TEXT,
                    consumed_resolution_sha256 TEXT,
                    created_at TEXT NOT NULL,
                    consumed_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_general_clarification_receipts_user_created
                    ON general_clarification_receipts(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS public_exploration_runs (
                    run_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    result_kind TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, run_id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                    CHECK (result_kind IN ('light', 'pressure', 'sensor'))
                );
                CREATE INDEX IF NOT EXISTS idx_public_explorations_user_updated
                    ON public_exploration_runs(user_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS phyphox_devices (
                    device_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    buffer_mapping_json TEXT NOT NULL,
                    experiment_title TEXT NOT NULL DEFAULT '',
                    compatible INTEGER NOT NULL DEFAULT 0,
                    is_default INTEGER NOT NULL DEFAULT 1,
                    last_seen_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_devices_user_default
                    ON phyphox_devices(user_id, is_default DESC, updated_at DESC);

                CREATE TABLE IF NOT EXISTS model_profiles (
                    profile_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    api_key_hint TEXT NOT NULL,
                    secret_ref TEXT NOT NULL UNIQUE,
                    input_cost_per_million REAL,
                    output_cost_per_million REAL,
                    reasoning_strategy TEXT NOT NULL DEFAULT 'auto',
                    is_default INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 1,
                    probe_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_model_profiles_user_default
                    ON model_profiles(user_id, is_default DESC, updated_at DESC);

                CREATE TABLE IF NOT EXISTS evidence_workbench_reports (
                    report_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_workbench_reports_user_created
                    ON evidence_workbench_reports(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS agent_run_audits (
                    run_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    audit_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_agent_run_audits_user_created
                    ON agent_run_audits(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS auth_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_auth_sessions_token
                    ON auth_sessions(token_hash);
                CREATE INDEX IF NOT EXISTS idx_auth_sessions_expiry
                    ON auth_sessions(expires_at);

                CREATE TABLE IF NOT EXISTS app_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            existing_columns = {
                row["name"]
                for row in self._connection.execute("PRAGMA table_info(users)").fetchall()
            }
            for column_name in ("username", "username_key", "password_hash"):
                if column_name not in existing_columns:
                    self._connection.execute(
                        f"ALTER TABLE users ADD COLUMN {column_name} TEXT"
                    )
            investigation_columns = {
                row["name"]
                for row in self._connection.execute(
                    "PRAGMA table_info(investigation_cases)"
                ).fetchall()
            }
            if "revision" not in investigation_columns:
                self._connection.execute(
                    "ALTER TABLE investigation_cases "
                    "ADD COLUMN revision INTEGER NOT NULL DEFAULT 1"
                )
            model_profile_columns = {
                row["name"]
                for row in self._connection.execute(
                    "PRAGMA table_info(model_profiles)"
                ).fetchall()
            }
            if "reasoning_strategy" not in model_profile_columns:
                self._connection.execute(
                    "ALTER TABLE model_profiles ADD COLUMN reasoning_strategy "
                    "TEXT NOT NULL DEFAULT 'auto'"
                )
            self._connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_key
                ON users(username_key) WHERE username_key IS NOT NULL
                """
            )
            now = utc_now()
            self._connection.execute(
                """
                INSERT INTO users(
                    user_id, username, username_key, password_hash,
                    display_name, created_at, updated_at
                )
                VALUES (?, NULL, NULL, NULL, ?, ?, ?)
                ON CONFLICT(user_id) DO NOTHING
                """,
                (DEFAULT_USER_ID, DEFAULT_DISPLAY_NAME, now, now),
            )
            self._connection.execute("PRAGMA user_version = 11")

    @contextmanager
    def transaction(self) -> Iterable[sqlite3.Connection]:
        """Run a group of statements atomically behind the database lock."""

        with self._lock, self._connection:
            yield self._connection

    def execute(self, sql: str, parameters: Iterable[Any] = ()) -> None:
        with self._lock, self._connection:
            self._connection.execute(sql, tuple(parameters))

    def fetch_one(self, sql: str, parameters: Iterable[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._connection.execute(sql, tuple(parameters)).fetchone()

    def fetch_all(self, sql: str, parameters: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._connection.execute(sql, tuple(parameters)).fetchall()

    def get_user(self, user_id: str = DEFAULT_USER_ID) -> sqlite3.Row:
        row = self.fetch_one("SELECT * FROM users WHERE user_id = ?", (user_id,))
        if row is None:
            raise KeyError(f"Unknown user_id: {user_id}")
        return row

    def update_user(self, display_name: str, user_id: str = DEFAULT_USER_ID) -> sqlite3.Row:
        self.execute(
            "UPDATE users SET display_name = ?, updated_at = ? WHERE user_id = ?",
            (display_name, utc_now(), user_id),
        )
        return self.get_user(user_id)

    def get_default_device(self, user_id: str = DEFAULT_USER_ID) -> sqlite3.Row | None:
        return self.fetch_one(
            """
            SELECT * FROM phyphox_devices
            WHERE user_id = ? AND is_default = 1
            ORDER BY updated_at DESC LIMIT 1
            """,
            (user_id,),
        )

    def save_default_device(
        self,
        *,
        device_id: str,
        name: str,
        base_url: str,
        buffer_mapping: dict[str, str],
        experiment_title: str,
        compatible: bool,
        last_seen_at: str,
        user_id: str = DEFAULT_USER_ID,
    ) -> sqlite3.Row:
        now = utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE phyphox_devices SET is_default = 0 WHERE user_id = ?",
                (user_id,),
            )
            self._connection.execute(
                """
                INSERT INTO phyphox_devices(
                    device_id, user_id, name, base_url, buffer_mapping_json,
                    experiment_title, compatible, is_default, last_seen_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(device_id) DO UPDATE SET
                    name = excluded.name,
                    base_url = excluded.base_url,
                    buffer_mapping_json = excluded.buffer_mapping_json,
                    experiment_title = excluded.experiment_title,
                    compatible = excluded.compatible,
                    is_default = 1,
                    last_seen_at = excluded.last_seen_at,
                    updated_at = excluded.updated_at
                """,
                (
                    device_id,
                    user_id,
                    name,
                    base_url,
                    json.dumps(buffer_mapping, ensure_ascii=False),
                    experiment_title,
                    int(compatible),
                    last_seen_at,
                    now,
                    now,
                ),
            )
        row = self.get_default_device(user_id)
        if row is None:  # pragma: no cover - guarded by the insert above
            raise RuntimeError("Default device was not saved")
        return row

    def delete_default_device(self, user_id: str = DEFAULT_USER_ID) -> None:
        self.execute(
            "DELETE FROM phyphox_devices WHERE user_id = ? AND is_default = 1",
            (user_id,),
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()


database = SQLiteDatabase()
