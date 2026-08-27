from __future__ import annotations

import json
from uuid import uuid4

from pocketlab.auth import get_current_user_id
from pocketlab.persistence import DEFAULT_USER_ID, SQLiteDatabase, database, utc_now
from pocketlab.phyphox import PhyphoxProbe
from pocketlab.schemas import (
    LocalProfile,
    PocketLabSettings,
    SavedPhyphoxDevice,
)


class SettingsStore:
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

    def get(self) -> PocketLabSettings:
        user_id = self._active_user_id
        user = self._database.get_user(user_id)
        device = self._database.get_default_device(user_id)
        return PocketLabSettings(
            profile=LocalProfile(
                user_id=user["user_id"],
                display_name=user["display_name"],
                created_at=user["created_at"],
                updated_at=user["updated_at"],
            ),
            default_phyphox_device=self._device_from_row(device) if device else None,
        )

    def update_profile(self, display_name: str) -> LocalProfile:
        row = self._database.update_user(display_name.strip(), self._active_user_id)
        return LocalProfile(
            user_id=row["user_id"],
            display_name=row["display_name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def save_default_phyphox(self, name: str, probe: PhyphoxProbe) -> SavedPhyphoxDevice:
        user_id = self._active_user_id
        current = self._database.get_default_device(user_id)
        row = self._database.save_default_device(
            device_id=current["device_id"] if current else uuid4().hex[:12],
            name=name.strip(),
            base_url=probe.base_url,
            buffer_mapping=probe.buffer_mapping.model_dump(),
            experiment_title=probe.experiment_title,
            compatible=probe.compatible,
            last_seen_at=utc_now(),
            user_id=user_id,
        )
        return self._device_from_row(row)

    def refresh_default_phyphox(self, probe: PhyphoxProbe) -> SavedPhyphoxDevice:
        current = self._database.get_default_device(self._active_user_id)
        if current is None:
            raise KeyError("尚未保存默认 phyphox 设备。")
        return self.save_default_phyphox(current["name"], probe)

    def delete_default_phyphox(self) -> None:
        self._database.delete_default_device(self._active_user_id)

    @staticmethod
    def _device_from_row(row: object) -> SavedPhyphoxDevice:
        return SavedPhyphoxDevice(
            device_id=row["device_id"],
            name=row["name"],
            base_url=row["base_url"],
            buffer_mapping=json.loads(row["buffer_mapping_json"]),
            experiment_title=row["experiment_title"],
            compatible=bool(row["compatible"]),
            is_default=bool(row["is_default"]),
            last_seen_at=row["last_seen_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


settings_store = SettingsStore(database, user_id=None)
