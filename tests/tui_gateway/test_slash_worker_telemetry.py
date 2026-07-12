from __future__ import annotations

import json


def test_lifecycle_event_is_bounded_pseudonymous_and_never_captures_content():
    from tui_gateway.slash_telemetry import SlashTelemetry

    emitted: list[str] = []
    telemetry = SlashTelemetry(
        emitted.append,
        session_key="session-private-123",
        profile_home=r"C:\Users\alice\AppData\Local\hermes-desktop",
        parent_pid=4242,
        launcher="C:/secret/location/python.exe",
        instance_seed="pid-9001-create-123.4",
        clock=lambda: 100.0,
    )
    event = telemetry.emit(
        "worker_starting",
        state="starting",
        reason="spawn_requested",
        command="/queue private prompt --token placeholder-secret",
        summary="Authorization: Bearer placeholder-secret C:\\Users\\alice\\private.txt",
    )
    assert event is not None
    assert event["schema"] == "hermes.dashboard.slash_worker.v1"
    assert event["event"] == "worker_starting"
    assert event["state"] == "starting"
    assert event["reason"] == "spawn_requested"
    assert event["severity"] == "info"
    assert event["fingerprint"].startswith("slash_worker:")
    assert event["command_category"] == "queue"
    for key, prefix in (
        ("session_ref", "session:"),
        ("profile_ref", "profile:"),
        ("parent_ref", "parent:"),
        ("launcher_ref", "launcher:"),
        ("instance_ref", "instance:"),
    ):
        assert event[key].startswith(prefix)
    wire = emitted[0]
    assert wire.startswith("HERMES_TELEMETRY ")
    assert json.loads(wire.removeprefix("HERMES_TELEMETRY ")) == event
    assert len(wire) <= 2048
    for prohibited in (
        "session-private-123", "private prompt", "placeholder-secret", "Users",
        "alice", "private.txt", "python.exe",
    ):
        assert prohibited not in wire


def test_queue_and_latency_metrics_use_fixed_buckets_only():
    from tui_gateway.slash_telemetry import age_bucket, depth_bucket, latency_bucket

    assert [depth_bucket(n) for n in (0, 1, 3, 9, 99)] == [
        "empty", "one", "two_to_five", "six_to_twenty", "over_twenty",
    ]
    assert [age_bucket(n) for n in (0, 2, 20, 90, 999)] == [
        "fresh", "under_5s", "5s_to_30s", "30s_to_2m", "over_2m",
    ]
    assert [latency_bucket(n) for n in (0.01, 0.4, 3, 20, 100)] == [
        "under_100ms", "100ms_to_1s", "1s_to_5s", "5s_to_30s", "over_30s",
    ]


def test_duplicate_burst_and_cardinality_suppression_are_counted_and_fail_safe():
    from tui_gateway.slash_telemetry import SlashTelemetry

    now = [100.0]
    lines: list[str] = []
    telemetry = SlashTelemetry(
        lines.append,
        session_key="session-a",
        profile_home="profile-a",
        parent_pid=42,
        launcher="python",
        instance_seed="instance-a",
        clock=lambda: now[0],
        dedup_window_seconds=10,
        burst_limit=2,
        cardinality_limit=1,
    )
    assert telemetry.emit(
        "command_failed", state="failed", reason="command_failed",
        command="/status private", dedup_key="same-request", command_ref="request-one",
    )
    assert telemetry.emit(
        "command_failed", state="failed", reason="command_failed",
        command="/status other", dedup_key="same-request", command_ref="request-one",
    ) is None
    assert telemetry.emit(
        "command_started", state="started", reason="command_started",
        command="/help", dedup_key="second", command_ref="request-two",
    )
    assert telemetry.emit(
        "command_started", state="started", reason="command_started",
        command="/help", dedup_key="third", command_ref="request-three",
    ) is None
    now[0] += 61
    health = telemetry.flush_suppressed()
    assert health is not None
    assert health["event"] == "telemetry_suppressed"
    assert health["state"] == "suppressed"
    assert health["suppressed_count"] == 1
    assert health["dropped_count"] >= 1
    assert health["cardinality_dropped_count"] >= 1

    def broken_sink(_line: str) -> None:
        raise OSError("disk full")

    broken = SlashTelemetry(
        broken_sink,
        session_key="session-a",
        profile_home="profile-a",
        parent_pid=42,
        launcher="python",
        instance_seed="instance-b",
    )
    assert broken.emit("worker_ready", state="ready", reason="child_ready") is None

    broken_clock = SlashTelemetry(
        lines.append,
        session_key="session-a",
        profile_home="profile-a",
        parent_pid=42,
        launcher="python",
        instance_seed="instance-c",
        clock=lambda: (_ for _ in ()).throw(RuntimeError("clock failed")),
    )
    assert broken_clock.emit("worker_ready", state="ready", reason="child_ready") is None


def test_parent_worker_emits_receipt_dispatch_completion_and_graceful_shutdown(monkeypatch):
    from tui_gateway import server

    telemetry_lines: list[str] = []

    class _Input:
        def write(self, _value):
            return None

        def flush(self):
            return None

        def close(self):
            return None

    class _Stream(list):
        def close(self):
            return None

    class _Proc:
        pid = 9001

        def __init__(self):
            self.stdin = _Input()
            self.stdout = _Stream(
                [
                    json.dumps({"telemetry": "initialized"}) + "\n",
                    json.dumps({"telemetry": "ready"}) + "\n",
                    json.dumps({"id": 1, "ok": True, "output": "private response"}) + "\n",
                ]
            )
            self.stderr = _Stream([r"token=hidden C:\Users\alice\secret.txt" + "\n"])
            self._returncode = None

        def poll(self):
            return self._returncode

        def terminate(self):
            self._returncode = 0

        def wait(self, timeout=None):
            return self._returncode or 0

        def kill(self):
            self._returncode = -9

    class _Thread:
        def __init__(self, target, daemon=True):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **kw: _Proc())
    monkeypatch.setattr(server.threading, "Thread", _Thread)
    monkeypatch.setattr(server.logger, "info", lambda _fmt, line: telemetry_lines.append(line))
    monkeypatch.setenv("HERMES_HOME", r"C:\Users\alice\AppData\Local\hermes-desktop")

    worker = server._SlashWorker("session-private", "model-private")
    assert worker.run("/status private argument", request_ref="rpc-private-1") == "private response"
    assert worker.run("/status private argument", request_ref="rpc-private-1") == "private response"
    worker.close()

    records = [json.loads(line.removeprefix("HERMES_TELEMETRY ")) for line in telemetry_lines]
    events = [record["event"] for record in records]
    assert events == [
        "worker_starting", "worker_initialized", "worker_ready", "command_received",
        "command_dispatched", "command_started", "command_completed",
        "command_deduplicated", "worker_stopping", "worker_stopped",
    ]
    completed = records[events.index("command_completed")]
    assert completed["command_category"] == "status"
    assert completed["queue_depth_bucket"] == "empty"
    assert "latency_bucket" in completed
    blob = json.dumps(records)
    for prohibited in (
        "session-private", "model-private", "private argument", "private response",
        "hidden", "Users", "alice", "secret.txt",
    ):
        assert prohibited not in blob
    assert worker.stderr_tail == ["secret_redacted;path_redacted"]


def test_child_protocol_reports_initializing_ready_and_prelogger_failure(monkeypatch):
    import io
    import sys

    from tui_gateway import slash_worker

    class _CLI:
        def __init__(self, **_kwargs):
            self.console = None

        def process_command(self, _command):
            print("safe output")

    monkeypatch.setattr(slash_worker, "HermesCLI", _CLI)
    monkeypatch.setattr(slash_worker, "_start_parent_death_watchdog", lambda *_: None)
    monkeypatch.setattr(slash_worker.psutil, "Process", lambda _pid: type("P", (), {"create_time": lambda self: 1.0})())
    monkeypatch.setattr(sys, "argv", ["slash_worker", "--session-key", "private-session"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"id": 1, "command": "/status private"}) + "\n"))
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    assert slash_worker.main() == 0
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert records[0] == {"telemetry": "initialized"}
    assert records[1] == {"telemetry": "ready"}
    assert records[2]["id"] == 1
    assert records[2]["ok"] is True

    class _BrokenCLI:
        def __init__(self, **_kwargs):
            raise RuntimeError("token=hidden C:\\Users\\alice\\secret.txt")

    monkeypatch.setattr(slash_worker, "HermesCLI", _BrokenCLI)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    failed = io.StringIO()
    monkeypatch.setattr(sys, "stdout", failed)
    assert slash_worker.main() == 1
    failed_records = [json.loads(line) for line in failed.getvalue().splitlines()]
    assert failed_records == [
        {"telemetry": "initialized"},
        {
        "telemetry": "bootstrap_failed",
        "reason": "initialization_failure",
        },
    ]
    for prohibited in ("hidden", "Users", "alice", "secret.txt", "private-session"):
        assert prohibited not in failed.getvalue()


def test_restart_loop_queue_recovery_and_delivery_boundary_are_explicit():
    from tui_gateway.slash_telemetry import SlashTelemetry

    now = [0.0]
    lines: list[str] = []
    telemetry = SlashTelemetry(
        lines.append,
        session_key="private-session",
        profile_home="private-profile",
        parent_pid=99,
        launcher="private-launcher",
        instance_seed="private-instance",
        clock=lambda: now[0],
    )
    telemetry.reset_restart_state_for_tests()
    for attempt in range(4):
        now[0] = float(attempt * 30)
        telemetry.record_restart_attempt(window_seconds=120, threshold=4)
    events = [json.loads(line.removeprefix("HERMES_TELEMETRY ")) for line in lines]
    assert [event["event"] for event in events].count("worker_restart_attempt") == 4
    assert [event["event"] for event in events].count("worker_restart_loop") == 1
    assert events[-1]["restart_count"] == 4

    now[0] = 150.0
    assert telemetry.record_ready_recovery()["event"] == "worker_restart_recovered"
    assert telemetry.observe_queue(depth=9, oldest_age_seconds=45)["event"] == "queue_degraded"
    assert telemetry.observe_queue(depth=12, oldest_age_seconds=80) is None
    assert telemetry.observe_queue(depth=0, oldest_age_seconds=0)["event"] == "queue_recovered"

    stalled = telemetry.observe_queue(depth=1, oldest_age_seconds=180)
    assert stalled["event"] == "queue_stalled"
    assert stalled["state"] == "stalled"
    assert telemetry.observe_queue(depth=0, oldest_age_seconds=0)["event"] == "queue_recovered"

    rejected = telemetry.record_command_rejected(
        command="/status private", command_ref="private-command", reason="invalid_request"
    )
    duplicate = telemetry.record_duplicate(
        command="/status private", command_ref="private-command"
    )
    assert rejected["state"] == "rejected"
    assert rejected["reason"] == "invalid_request"
    assert duplicate["state"] == "deduplicated"
    assert duplicate["reason"] == "duplicate_request"

    delivery = telemetry.record_delivery_handoff_failure(
        command="/status private",
        command_ref="private-command",
        gateway_event_ref="private-gateway-event",
    )
    assert delivery["event"] == "command_response_failed"
    assert delivery["reason"] == "delivery_handoff_failure"
    assert delivery["gateway_event_ref"].startswith("gateway_event:")
    blob = json.dumps(delivery)
    for prohibited in ("private", "telegram.sent", "telegram.send_failed", "channel.delivery"):
        assert prohibited not in blob


def test_diagnostics_and_reused_or_missing_process_ids_stay_safe_and_distinct():
    from tui_gateway.slash_telemetry import SlashTelemetry, redact_summary

    summary = redact_summary(
        "RuntimeError: prompt='launch banana' message='private chat' "
        "Authorization: Bearer placeholder-secret C:\\Users\\alice\\secret.txt"
    )
    assert summary == "RuntimeError;secret_redacted;path_redacted;content_redacted"
    for prohibited in ("banana", "private chat", "placeholder-secret", "Users", "alice"):
        assert prohibited not in summary

    common = {
        "sink": lambda _line: None,
        "session_key": None,
        "profile_home": None,
        "parent_pid": 4242,
        "launcher": "python",
    }
    first = SlashTelemetry(instance_seed="pid:7:create:1", **common).emit(
        "worker_starting", state="starting", reason="spawn_requested"
    )
    reused = SlashTelemetry(instance_seed="pid:7:create:2", **common).emit(
        "worker_starting", state="starting", reason="spawn_requested"
    )
    assert first["instance_ref"] != reused["instance_ref"]
    assert first["session_ref"].startswith("session:")
    assert first["profile_ref"].startswith("profile:")
