\
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class Profile:
    name: str
    data: dict[str, Any]

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    @property
    def sandbox(self) -> str:
        return self.data.get("sandbox", "local")

    @property
    def docker(self) -> dict[str, Any]:
        return self.data.get("docker", {}) or {}

    @property
    def budgets(self) -> dict[str, Any]:
        return self.data.get("budgets", {}) or {}

    @property
    def contracts(self) -> dict[str, Any]:
        return self.data.get("contracts", {}) or {}

    @property
    def allowlist_profile(self) -> str:
        return self.data.get("allowlist_profile", "dev")

    @property
    def event_log_path(self) -> str:
        return self.data.get("event_log_path", "artifacts/events.jsonl")

    @property
    def publish(self) -> dict[str, Any]:
        return self.data.get("publish", {}) or {}

    @property
    def signing(self) -> dict[str, Any]:
        return self.data.get("signing", {}) or {}


def load_profiles(path: str) -> dict[str, Profile]:
    with open(path, encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    profiles = obj.get("profiles", {}) or {}
    return {k: Profile(k, v or {}) for k, v in profiles.items()}


def resolve_profile(profile_name: str, explicit_path: str | None = None) -> Profile:
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.append(os.environ.get("RFSN_PROFILES"))
    candidates.append(os.path.join(os.path.dirname(__file__), "profiles.default.yaml"))
    candidates.append("profiles.yaml")

    for p in candidates:
        if not p:
            continue
        if os.path.exists(p):
            profs = load_profiles(p)
            if profile_name in profs:
                return profs[profile_name]

    raise FileNotFoundError(
        f"Profile '{profile_name}' not found. Looked in: {', '.join([c for c in candidates if c])}"
    )
