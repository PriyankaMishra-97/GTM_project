"""Fixed internal-team login + per-user data scope.

Role in architecture: the only place user identity and region/segment
permissions are defined. `UserStore.authenticate()` is the single gate the UI
calls; `UserProfile.allowed_scope_values()` is what router/rules.py and
sql/guard.py consult to enforce a user's permitted regions/segments.

No self-signup, no SSO: a fixed team, provisioned by editing
`assets/users.yaml` (gitignored - see `assets/users.example.yaml`).

In:  username + password, or a username lookup.
Out: `UserProfile | None` - `None` on any auth failure (unknown user and wrong
     password are indistinguishable to the caller, so a login form can't be
     used to enumerate valid usernames).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bcrypt
import yaml

from core import config
from sql.schema import REGIONS, SEGMENTS


class UsersFileMissing(FileNotFoundError):
    """The user config file is not where the app expects it."""


@dataclass(frozen=True)
class UserProfile:
    """One logged-in user's identity and data scope.

    `allowed_regions`/`allowed_segments` are `None` when the user's yaml entry
    says `"all"` - `None` means unrestricted, so callers never need to special
    -case the string "all" themselves.
    """

    username: str
    allowed_regions: frozenset[str] | None
    allowed_segments: frozenset[str] | None

    def allowed_scope_values(self) -> frozenset[str] | None:
        """Union of allowed regions + segments, or None if fully unrestricted.

        `segment_or_region` is a single router slot covering both concepts
        (router/slots.py), so scope checks work against one combined set. A
        user restricted on only ONE dimension (e.g. region-limited but
        segment-unrestricted) must NOT collapse to "no restriction" - the
        unrestricted dimension is materialised to its full enum (REGIONS/
        SEGMENTS) before unioning, so the other dimension's restriction still
        holds.
        """
        if self.allowed_regions is None and self.allowed_segments is None:
            return None  # unrestricted on both - no scope check needed at all
        regions = self.allowed_regions if self.allowed_regions is not None else REGIONS
        segments = self.allowed_segments if self.allowed_segments is not None else SEGMENTS
        return regions | segments


def _scope_set(raw: Any) -> frozenset[str] | None:
    """Parse a yaml scope list; `["all"]` (or omitted) means unrestricted."""
    if not raw:
        return None
    values = {str(v).strip() for v in raw}
    if any(v.lower() == "all" for v in values):
        return None
    return frozenset(values)


class UserStore:
    """Loads the fixed user list once at construction, like SchemaCatalog."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or config.USERS_PATH)
        self._users: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            raise UsersFileMissing(
                f"User config not found at {self.path}.\n"
                f"Copy assets/users.example.yaml to {self.path.name} and add "
                "your team's credentials."
            )
        with self.path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data.get("users", {})

    def authenticate(self, username: str, password: str) -> UserProfile | None:
        """Verify credentials; None on any failure (unknown user or bad password)."""
        entry = self._users.get(username)
        if entry is None:
            return None
        stored_hash = entry.get("password_hash", "")
        try:
            ok = bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
        except ValueError:  # malformed hash in the config
            ok = False
        if not ok:
            return None
        return UserProfile(
            username=username,
            allowed_regions=_scope_set(entry.get("allowed_regions")),
            allowed_segments=_scope_set(entry.get("allowed_segments")),
        )
