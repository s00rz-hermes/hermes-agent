"""Privacy-safe local telemetry for the dashboard slash-command worker.

Events are bounded JSON records written through Hermes' normal GUI logger. They
contain only fixed enums, counters/buckets, and opaque identifiers; command
arguments and response/message content never enter the record builder.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

SCHEMA = "hermes.dashboard.slash_worker.v1"
PREFIX = "HERMES_TELEMETRY "
MAX_WIRE_BYTES = 2048
MAX_SUMMARY = 240

_COMMAND_CATEGORIES = frozenset(
    {
        "agents", "background", "clear", "compress", "config", "goal", "help",
        "kanban", "learn", "model", "new", "plugins", "queue", "reload",
        "reload-mcp", "resume", "retry", "skills", "snapshot", "status", "steer",
        "stop", "tools", "undo", "usage", "voice",
    }
)
_ALLOWED_STATES = frozenset(
    {
        "starting", "initializing", "ready", "accepted", "rejected",
        "deduplicated", "dispatched", "started", "completed", "failed",
        "timed_out", "stalled", "abandoned", "degraded", "recovering",
        "recovered", "stopping", "stopped", "crashed", "suppressed",
    }
)
_ALLOWED_REASONS = frozenset(
    {
        "spawn_requested", "child_initializing", "child_ready", "initialization_failure", "command_received",
        "command_dispatched", "command_started", "command_completed", "command_failed",
        "command_timeout", "duplicate_request", "invalid_request", "queue_pressure",
        "queue_stall", "queue_recovered", "parent_exit", "graceful_shutdown",
        "forced_shutdown", "shutdown_failed",
        "pipe_closed", "abrupt_exit", "restart_attempt", "restart_loop",
        "restart_recovered", "response_failure", "delivery_handoff_failure",
        "telemetry_backpressure", "cardinality_budget", "unknown",
    }
)
_SECRET_RE = re.compile(
    r"(?i)(?:authorization\s*[:=]?\s*bearer|bearer|api[_-]?key\s*[:=]?|"
    r"token\s*[:=]?|password\s*[:=]?|secret\s*[:=]?)\s*[^\s,;]+"
)
_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}")
_PATH_RE = re.compile(r"(?i)(?:[a-z]:[\\/]|/home/|/users/)[^\s,;]+")
_URL_RE = re.compile(r"(?i)\bhttps?://[^\s,;]+")
_CONTENT_RE = re.compile(r"(?i)\b(?:prompt|message|content|text|body)\b\s*[:=]")
_EXCEPTION_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning)\b")


def _opaque(namespace: str, value: Any) -> str:
    raw = str(value or "missing").encode("utf-8", errors="ignore")
    digest = hashlib.sha256(namespace.encode("ascii") + b":" + raw).hexdigest()[:16]
    return f"{namespace}:{digest}"


def command_category(command: Any) -> str:
    """Return a fixed low-cardinality command category without retaining args."""
    text = str(command or "").lstrip()
    if text.startswith("/"):
        text = text[1:]
    name = text.split(maxsplit=1)[0].lower()[:32] if text else ""
    name = {"q": "queue", "tasks": "agents", "reset": "new"}.get(name, name)
    return name if name in _COMMAND_CATEGORIES else "other"


def redact_summary(value: Any) -> str:
    """Return only allowlisted diagnostic classes and redaction markers.

    Free-form stderr/exception text can contain prompts or chat content without
    a reliable label, so it is never copied through. This intentionally favors
    privacy over prose-rich diagnostics.
    """
    text = " ".join(str(value or "").split())[: MAX_SUMMARY * 4]
    parts: list[str] = []
    if match := _EXCEPTION_RE.search(text):
        parts.append(match.group(0)[:64])
    if _SECRET_RE.search(text) or _KEY_RE.search(text):
        parts.append("secret_redacted")
    if _PATH_RE.search(text):
        parts.append("path_redacted")
    if _URL_RE.search(text):
        parts.append("url_redacted")
    if _CONTENT_RE.search(text):
        parts.append("content_redacted")
    lower = text.lower()
    for phrase, label in (
        ("timed out", "timeout"),
        ("timeout", "timeout"),
        ("closed pipe", "pipe_closed"),
        ("broken pipe", "pipe_closed"),
        ("connection reset", "connection_reset"),
        ("permission denied", "permission_denied"),
        ("address already in use", "address_in_use"),
    ):
        if phrase in lower and label not in parts:
            parts.append(label)
    return ";".join(parts)[:MAX_SUMMARY] or "diagnostic_redacted"


def depth_bucket(value: Any) -> str:
    try:
        depth = max(0, int(value))
    except (TypeError, ValueError):
        return "unknown"
    if depth == 0:
        return "empty"
    if depth == 1:
        return "one"
    if depth <= 5:
        return "two_to_five"
    if depth <= 20:
        return "six_to_twenty"
    return "over_twenty"


def age_bucket(value: Any) -> str:
    try:
        age = max(0.0, float(value))
    except (TypeError, ValueError):
        return "unknown"
    if age == 0:
        return "fresh"
    if age < 5:
        return "under_5s"
    if age < 30:
        return "5s_to_30s"
    if age < 120:
        return "30s_to_2m"
    return "over_2m"


def latency_bucket(value: Any) -> str:
    try:
        latency = max(0.0, float(value))
    except (TypeError, ValueError):
        return "unknown"
    if latency < 0.1:
        return "under_100ms"
    if latency < 1:
        return "100ms_to_1s"
    if latency < 5:
        return "1s_to_5s"
    if latency < 30:
        return "5s_to_30s"
    return "over_30s"


class SlashTelemetry:
    """Build and emit bounded local operational events without affecting work."""

    _restart_lock = threading.Lock()
    _restart_windows: dict[str, deque[float]] = {}
    _restart_loops: set[str] = set()

    def __init__(
        self,
        sink: Callable[[str], Any],
        *,
        session_key: Any,
        profile_home: Any,
        parent_pid: Any,
        launcher: Any,
        instance_seed: Any,
        clock: Callable[[], float] = time.time,
        dedup_window_seconds: float = 60.0,
        burst_limit: int = 50,
        cardinality_limit: int = 128,
    ) -> None:
        self._sink = sink
        self._clock = clock
        self._dedup_window = max(0.0, float(dedup_window_seconds))
        self._burst_limit = max(1, int(burst_limit))
        self._cardinality_limit = max(1, int(cardinality_limit))
        self._dedup: dict[str, float] = {}
        self._burst: deque[float] = deque()
        self._cardinality: set[str] = set()
        self._suppressed_count = 0
        self._dropped_count = 0
        self._cardinality_dropped_count = 0
        self._queue_health = "healthy"
        self._refs = {
            "session_ref": _opaque("session", session_key),
            "profile_ref": _opaque("profile", profile_home),
            "parent_ref": _opaque("parent", parent_pid),
            "launcher_ref": _opaque("launcher", launcher),
            "instance_ref": _opaque("instance", instance_seed),
        }

    def emit(
        self,
        event: str,
        *,
        state: str,
        reason: str,
        command: Any = None,
        summary: Any = None,
        dedup_key: Any = None,
        _bypass_limits: bool = False,
        **fields: Any,
    ) -> dict[str, Any] | None:
        """Fail-open boundary: telemetry must never affect worker behavior."""
        try:
            return self._emit(
                event,
                state=state,
                reason=reason,
                command=command,
                summary=summary,
                dedup_key=dedup_key,
                _bypass_limits=_bypass_limits,
                **fields,
            )
        except Exception:
            return None

    def _emit(
        self,
        event: str,
        *,
        state: str,
        reason: str,
        command: Any = None,
        summary: Any = None,
        dedup_key: Any = None,
        _bypass_limits: bool = False,
        **fields: Any,
    ) -> dict[str, Any] | None:
        now = self._clock()
        if not _bypass_limits:
            while self._burst and now - self._burst[0] >= 60.0:
                self._burst.popleft()
            if dedup_key is not None:
                dedup_ref = _opaque("dedup", dedup_key)
                previous = self._dedup.get(dedup_ref)
                if previous is not None and now - previous < self._dedup_window:
                    self._suppressed_count += 1
                    return None
                self._dedup[dedup_ref] = now
                if len(self._dedup) > self._cardinality_limit * 4:
                    cutoff = now - self._dedup_window
                    self._dedup = {key: ts for key, ts in self._dedup.items() if ts >= cutoff}
            if len(self._burst) >= self._burst_limit:
                self._dropped_count += 1
                return None
            self._burst.append(now)
            for key in tuple(fields):
                if not key.endswith("_ref") or not fields[key]:
                    continue
                candidate = _opaque(key[:-4] or "ref", fields[key])
                if candidate not in self._cardinality and len(self._cardinality) >= self._cardinality_limit:
                    fields[key] = "overflow"
                    self._cardinality_dropped_count += 1
                else:
                    self._cardinality.add(candidate)

        event_name = re.sub(r"[^a-z0-9_.-]", "_", str(event).lower())[:64]
        safe_state = state if state in _ALLOWED_STATES else "failed"
        safe_reason = reason if reason in _ALLOWED_REASONS else "unknown"
        if safe_state in {"failed", "timed_out", "crashed", "abandoned"}:
            severity = "error"
        elif safe_state in {"degraded", "stalled"}:
            severity = "warning"
        else:
            severity = "info"
        fingerprint_seed = ":".join(
            (event_name, safe_state, safe_reason, self._refs["session_ref"])
        )
        fingerprint = hashlib.sha256(fingerprint_seed.encode("ascii")).hexdigest()[:16]
        record: dict[str, Any] = {
            "schema": SCHEMA,
            "event": event_name,
            "state": safe_state,
            "reason": safe_reason,
            "severity": severity,
            "fingerprint": f"slash_worker:{fingerprint}",
            "observed_at_unix_ms": int(now * 1000),
            "command_category": command_category(command),
            **self._refs,
        }
        if summary:
            record["summary"] = redact_summary(summary)
        integer_fields = {
            "exit_code", "suppressed_count", "dropped_count",
            "cardinality_dropped_count", "restart_count",
        }
        for key, value in fields.items():
            if key.endswith("_bucket") and isinstance(value, str):
                record[key[:48]] = re.sub(r"[^a-z0-9_.-]", "_", value.lower())[:32]
            elif key in integer_fields:
                try:
                    record[key] = max(-2**31, min(2**31 - 1, int(value)))
                except (TypeError, ValueError):
                    continue
            elif key.endswith("_ref") and value:
                record[key[:48]] = _opaque(key[:-4] or "ref", value)

        wire = PREFIX + json.dumps(record, sort_keys=True, separators=(",", ":"))
        if len(wire.encode("utf-8")) > MAX_WIRE_BYTES:
            record.pop("summary", None)
            wire = PREFIX + json.dumps(record, sort_keys=True, separators=(",", ":"))
        try:
            self._sink(wire)
        except Exception:
            return None
        return record

    @classmethod
    def reset_restart_state_for_tests(cls) -> None:
        with cls._restart_lock:
            cls._restart_windows.clear()
            cls._restart_loops.clear()

    def record_restart_attempt(
        self, *, window_seconds: float = 300.0, threshold: int = 4
    ) -> list[dict[str, Any]]:
        """Record a bounded restart window and emit one loop transition edge."""
        now = self._clock()
        session_ref = self._refs["session_ref"]
        with self._restart_lock:
            history = self._restart_windows.setdefault(session_ref, deque())
            while history and now - history[0] > max(1.0, window_seconds):
                history.popleft()
            history.append(now)
            count = len(history)
            loop_edge = count >= max(2, threshold) and session_ref not in self._restart_loops
            if loop_edge:
                self._restart_loops.add(session_ref)
        emitted: list[dict[str, Any]] = []
        attempt = self.emit(
            "worker_restart_attempt",
            state="starting",
            reason="restart_attempt",
            restart_count=count,
        )
        if attempt:
            emitted.append(attempt)
        if loop_edge:
            loop = self.emit(
                "worker_restart_loop",
                state="degraded",
                reason="restart_loop",
                restart_count=count,
            )
            if loop:
                emitted.append(loop)
        return emitted

    def record_ready_recovery(self) -> dict[str, Any] | None:
        """Resolve a previously emitted restart-loop edge on confirmed readiness."""
        session_ref = self._refs["session_ref"]
        with self._restart_lock:
            if session_ref not in self._restart_loops:
                return None
            self._restart_loops.remove(session_ref)
            self._restart_windows.pop(session_ref, None)
        return self.emit(
            "worker_restart_recovered",
            state="recovered",
            reason="restart_recovered",
        )

    def observe_queue(self, *, depth: Any, oldest_age_seconds: Any) -> dict[str, Any] | None:
        """Emit only queue pressure and healthy recovery transitions."""
        try:
            depth_value = max(0, int(depth))
        except (TypeError, ValueError):
            depth_value = 0
        try:
            age_value = max(0.0, float(oldest_age_seconds))
        except (TypeError, ValueError):
            age_value = 0.0
        next_health = (
            "stalled"
            if age_value >= 120.0
            else "degraded"
            if depth_value >= 6 or age_value >= 30.0
            else "healthy"
        )
        if next_health == self._queue_health:
            return None
        previous_health = self._queue_health
        self._queue_health = next_health
        if next_health == "stalled":
            return self.emit(
                "queue_stalled",
                state="stalled",
                reason="queue_stall",
                queue_depth_bucket=depth_bucket(depth_value),
                queue_age_bucket=age_bucket(age_value),
            )
        if next_health == "degraded":
            return self.emit(
                "queue_degraded",
                state="degraded",
                reason="queue_pressure",
                queue_depth_bucket=depth_bucket(depth_value),
                queue_age_bucket=age_bucket(age_value),
            )
        if previous_health != "healthy":
            return self.emit(
                "queue_recovered",
                state="recovered",
                reason="queue_recovered",
                queue_depth_bucket=depth_bucket(depth_value),
                queue_age_bucket=age_bucket(age_value),
            )
        return None

    def record_command_rejected(
        self, *, command: Any, command_ref: Any, reason: str = "invalid_request"
    ) -> dict[str, Any] | None:
        """Record a bounded worker-local rejection without retaining input."""
        return self.emit(
            "command_rejected",
            state="rejected",
            reason=reason,
            command=command,
            command_ref=command_ref,
        )

    def record_duplicate(
        self, *, command: Any, command_ref: Any
    ) -> dict[str, Any] | None:
        """Record replay suppression without duplicating command content."""
        return self.emit(
            "command_deduplicated",
            state="deduplicated",
            reason="duplicate_request",
            command=command,
            command_ref=command_ref,
            dedup_key=command_ref,
        )

    def record_delivery_handoff_failure(
        self,
        *,
        command: Any,
        command_ref: Any,
        gateway_event_ref: Any = None,
    ) -> dict[str, Any] | None:
        """Own only the worker-local response/handoff edge, never channel delivery."""
        return self.emit(
            "command_response_failed",
            state="failed",
            reason="delivery_handoff_failure",
            command=command,
            command_ref=command_ref,
            gateway_event_ref=gateway_event_ref,
        )

    def flush_suppressed(self) -> dict[str, Any] | None:
        """Emit one coalesced health record for suppressed/dropped events."""
        if not (self._suppressed_count or self._dropped_count or self._cardinality_dropped_count):
            return None
        counts = {
            "suppressed_count": self._suppressed_count,
            "dropped_count": self._dropped_count,
            "cardinality_dropped_count": self._cardinality_dropped_count,
        }
        self._suppressed_count = 0
        self._dropped_count = 0
        self._cardinality_dropped_count = 0
        return self.emit(
            "telemetry_suppressed",
            state="suppressed",
            reason="telemetry_backpressure",
            _bypass_limits=True,
            **counts,
        )
