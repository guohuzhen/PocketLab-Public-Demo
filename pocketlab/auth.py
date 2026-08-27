from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from pwdlib import PasswordHash

from pocketlab.persistence import DEFAULT_USER_ID, SQLiteDatabase, database, utc_now

SESSION_COOKIE_NAME = "pocketlab_session"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 7
_LOCAL_DATA_CLAIM_KEY = "local_data_claimed_by"
_current_user_id: ContextVar[str] = ContextVar(
    "pocketlab_current_user_id",
    default=DEFAULT_USER_ID,
)


class UsernameTakenError(ValueError):
    pass


class InvalidCredentialsError(ValueError):
    pass


@dataclass(frozen=True)
class Account:
    user_id: str
    username: str
    display_name: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class RegistrationResult:
    account: Account
    claimed_local_data: bool


def get_current_user_id() -> str:
    return _current_user_id.get()


@contextmanager
def user_context(user_id: str) -> Iterator[None]:
    token = _current_user_id.set(user_id)
    try:
        yield
    finally:
        _current_user_id.reset(token)


def cookie_secure() -> bool:
    return os.getenv("POCKETLAB_COOKIE_SECURE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def normalize_username(username: str) -> tuple[str, str]:
    normalized = unicodedata.normalize("NFKC", username).strip()
    if not 3 <= len(normalized) <= 32:
        raise ValueError("用户名长度必须为 3 到 32 个字符。")
    if not normalized[0].isalnum() or not normalized[-1].isalnum():
        raise ValueError("用户名必须以字母或数字开头和结尾。")
    if not all(character.isalnum() or character in "._-" for character in normalized):
        raise ValueError("用户名只能包含字母、数字、点、下划线和连字符。")
    return normalized, normalized.casefold()


class AuthStore:
    def __init__(self, storage: SQLiteDatabase | None = None) -> None:
        self._database = storage or SQLiteDatabase(":memory:")
        self._password_hash = PasswordHash.recommended()
        self._dummy_hash = self._password_hash.hash("pocketlab-dummy-password")

    def register(
        self,
        *,
        username: str,
        password: str,
        display_name: str | None = None,
        claim_local_data: bool = True,
    ) -> RegistrationResult:
        normalized_username, username_key = normalize_username(username)
        if not 8 <= len(password) <= 128:
            raise ValueError("密码长度必须为 8 到 128 个字符。")
        normalized_display_name = (display_name or normalized_username).strip()
        if not 1 <= len(normalized_display_name) <= 60:
            raise ValueError("显示名称长度必须为 1 到 60 个字符。")

        user_id = f"usr_{uuid4().hex}"
        now = utc_now()
        password_hash = self._password_hash.hash(password)
        claimed_local_data = False
        try:
            with self._database.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO users(
                        user_id, username, username_key, password_hash,
                        display_name, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        normalized_username,
                        username_key,
                        password_hash,
                        normalized_display_name,
                        now,
                        now,
                    ),
                )
                claim_owner = connection.execute(
                    "SELECT value FROM app_meta WHERE key = ?",
                    (_LOCAL_DATA_CLAIM_KEY,),
                ).fetchone()
                if claim_local_data and claim_owner is None:
                    legacy_count = 0
                    for table_name in (
                        "sessions",
                        "diagnostic_cases",
                        "investigation_cases",
                        "general_exploration_cases",
                        "general_compilation_receipts",
                        "public_exploration_runs",
                        "phyphox_devices",
            "model_profiles",
            "evidence_workbench_reports",
            "agent_run_audits",
                    ):
                        count_row = connection.execute(
                            f"SELECT COUNT(*) AS count FROM {table_name} WHERE user_id = ?",
                            (DEFAULT_USER_ID,),
                        ).fetchone()
                        legacy_count += int(count_row["count"])
                        connection.execute(
                            f"UPDATE {table_name} SET user_id = ? WHERE user_id = ?",
                            (user_id, DEFAULT_USER_ID),
                        )
                    connection.execute(
                        "INSERT INTO app_meta(key, value) VALUES (?, ?)",
                        (_LOCAL_DATA_CLAIM_KEY, user_id),
                    )
                    claimed_local_data = legacy_count > 0
        except sqlite3.IntegrityError as exc:
            if "username" in str(exc).lower():
                raise UsernameTakenError("该用户名已被使用。") from exc
            raise
        return RegistrationResult(
            account=self.get_account(user_id),
            claimed_local_data=claimed_local_data,
        )

    def authenticate(self, username: str, password: str) -> Account:
        try:
            _, username_key = normalize_username(username)
        except ValueError:
            username_key = ""
        row = self._database.fetch_one(
            "SELECT * FROM users WHERE username_key = ?",
            (username_key,),
        )
        stored_hash = row["password_hash"] if row is not None else self._dummy_hash
        verified = self._password_hash.verify(password, stored_hash)
        if row is None or not verified or not row["username"]:
            raise InvalidCredentialsError("用户名或密码错误。")
        return self._account_from_row(row)

    def create_session(self, user_id: str) -> str:
        self.get_account(user_id)
        token = secrets.token_urlsafe(32)
        now = datetime.now(UTC)
        self._database.execute(
            """
            INSERT INTO auth_sessions(
                session_id, user_id, token_hash, created_at, expires_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                uuid4().hex,
                user_id,
                self._token_hash(token),
                now.isoformat(),
                (now + timedelta(seconds=SESSION_MAX_AGE_SECONDS)).isoformat(),
                now.isoformat(),
            ),
        )
        return token

    def resolve_session(self, token: str | None) -> Account | None:
        if not token:
            return None
        now = utc_now()
        row = self._database.fetch_one(
            """
            SELECT users.*
            FROM auth_sessions
            JOIN users ON users.user_id = auth_sessions.user_id
            WHERE auth_sessions.token_hash = ? AND auth_sessions.expires_at > ?
            """,
            (self._token_hash(token), now),
        )
        if row is None or not row["username"]:
            return None
        return self._account_from_row(row)

    def revoke_session(self, token: str | None) -> None:
        if token:
            self._database.execute(
                "DELETE FROM auth_sessions WHERE token_hash = ?",
                (self._token_hash(token),),
            )

    def get_account(self, user_id: str) -> Account:
        row = self._database.get_user(user_id)
        if not row["username"]:
            raise KeyError(f"User has no login account: {user_id}")
        return self._account_from_row(row)

    def local_data_available(self) -> bool:
        claim_owner = self._database.fetch_one(
            "SELECT value FROM app_meta WHERE key = ?",
            (_LOCAL_DATA_CLAIM_KEY,),
        )
        if claim_owner is not None:
            return False
        for table_name in ("sessions", "diagnostic_cases", "phyphox_devices"):
            row = self._database.fetch_one(
                f"SELECT 1 FROM {table_name} WHERE user_id = ? LIMIT 1",
                (DEFAULT_USER_ID,),
            )
            if row is not None:
                return True
        return False

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _account_from_row(row: object) -> Account:
        return Account(
            user_id=row["user_id"],
            username=row["username"],
            display_name=row["display_name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


auth_store = AuthStore(database)
