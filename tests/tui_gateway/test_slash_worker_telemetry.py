from __future__ import annotations

import json


def test_json_rpc_id_is_correlation_only_and_never_caches_response():
    from tui_gateway import server

    writes: list[dict] = []

    class _Input:
        def write(self, value):
            writes.append(json.loads(value))

        def flush(self):
            return None

    worker = object.__new__(server._SlashWorker)
    worker._lock = __import__("threading").Lock()
    worker._queue_lock = __import__("threading").Lock()
    worker._seq = 0
    worker._request_seq = 0
    worker._queued_at = {}
    worker._request_correlations = {}
    worker.proc = type("P", (), {"stdin": _Input(), "poll": lambda self: None})()
    worker.stdout_queue = __import__("queue").Queue()
    worker.stderr_tail = []
    worker.telemetry = type(
        "T",
        (),
        {
            "emit": lambda *a, **kw: None,
            "observe_queue": lambda *a, **kw: None,
            "record_correlation_reuse": lambda *a, **kw: None,
        },
    )()
    worker.stdout_queue.put({"id": 1, "ok": True, "output": "RESULT-A"})
    worker.stdout_queue.put({"id": 2, "ok": True, "output": "RESULT-B"})

    assert worker.run("/status first", request_ref=7) == "RESULT-A"
    assert worker.run("/status second", request_ref=7) == "RESULT-B"
    assert [item["command"] for item in writes] == ["/status first", "/status second"]
    assert not hasattr(worker, "_request_results")
    assert all(len(key) <= 32 and len(value[0]) <= 32 for key, value in worker._request_correlations.items())


def test_stdin_write_failure_emits_terminal_command_failure():
    import queue
    import threading

    import pytest

    from tui_gateway import server

    events: list[tuple[str, dict]] = []

    class _Input:
        def write(self, _value):
            raise BrokenPipeError("private response body")

    worker = object.__new__(server._SlashWorker)
    worker._lock = threading.Lock()
    worker._queue_lock = threading.Lock()
    worker._seq = 0
    worker._request_seq = 0
    worker._queued_at = {}
    worker._request_correlations = {}
    worker.proc = type("P", (), {"stdin": _Input(), "poll": lambda self: None})()
    worker.stdout_queue = queue.Queue()
    worker.stderr_tail = []
    worker.telemetry = type(
        "T",
        (),
        {
            "emit": lambda _self, event, **fields: events.append((event, fields)),
            "observe_queue": lambda *a, **kw: None,
            "record_correlation_reuse": lambda *a, **kw: None,
        },
    )()

    with pytest.raises(BrokenPipeError):
        worker.run("/status secret", request_ref=1)

    assert events[-1][0] == "command_failed"
    assert events[-1][1]["reason"] == "pipe_closed"
    assert events[-1][1]["state"] == "failed"


def test_close_never_emits_stopped_until_exit_is_confirmed():
    from tui_gateway import server

    events: list[str] = []

    class _Stream:
        def close(self):
            return None

    class _LiveProc:
        stdin = stdout = stderr = _Stream()

        def poll(self):
            return None

        def terminate(self):
            raise OSError("terminate failed")

        def kill(self):
            raise OSError("kill failed")

        def wait(self, timeout=None):
            raise TimeoutError("still live")

    worker = object.__new__(server._SlashWorker)
    worker._closed = False
    worker.proc = _LiveProc()
    worker.telemetry = type(
        "T",
        (),
        {
            "emit": lambda _self, event, **_fields: events.append(event),
            "flush_suppressed": lambda _self: None,
        },
    )()

    worker.close()

    assert "worker_shutdown_failed" in events
    assert "worker_stopped" not in events


def test_idle_stdout_eof_emits_worker_crash_with_actual_exit_status():
    import queue

    from tui_gateway import server

    events: list[tuple[str, dict]] = []
    worker = object.__new__(server._SlashWorker)
    worker._closed = False
    worker.proc = type("P", (), {"stdout": [], "poll": lambda self: 23})()
    worker.stdout_queue = queue.Queue()
    worker.telemetry = type(
        "T",
        (),
        {"emit": lambda _self, event, **fields: events.append((event, fields))},
    )()

    worker._drain_stdout()

    assert events == [
        (
            "worker_crashed",
            {"state": "crashed", "reason": "abrupt_exit", "exit_code": 23},
        )
    ]
    assert worker.stdout_queue.get_nowait() is None


def test_parent_propagates_child_bootstrap_exit_status():
    import queue

    from tui_gateway import server

    events: list[tuple[str, dict]] = []
    worker = object.__new__(server._SlashWorker)
    worker._closed = False
    worker.proc = type(
        "P",
        (),
        {
            "stdout": [json.dumps({"telemetry": "bootstrap_failed", "exit_code": 1})],
            "poll": lambda self: 1,
        },
    )()
    worker.stdout_queue = queue.Queue()
    worker.telemetry = type(
        "T",
        (),
        {"emit": lambda _self, event, **fields: events.append((event, fields))},
    )()

    worker._drain_stdout()

    assert events[0] == (
        "worker_bootstrap_failed",
        {"state": "failed", "reason": "initialization_failure", "exit_code": 1},
    )


def test_restart_and_dedup_state_are_strictly_count_and_ttl_bounded():
    from tui_gateway.slash_telemetry import SlashTelemetry

    now = [0.0]
    SlashTelemetry.reset_restart_state_for_tests()
    for index in range(1000):
        telemetry = SlashTelemetry(
            lambda _line: None,
            session_key=f"session-{index}",
            profile_home="profile",
            parent_pid=1,
            launcher="python",
            instance_seed=index,
            clock=lambda: now[0],
            burst_limit=1,
            cardinality_limit=8,
            dedup_window_seconds=10,
        )
        telemetry.record_restart_attempt(window_seconds=10)
    assert len(SlashTelemetry._restart_windows) <= 128
    assert len(SlashTelemetry._restart_loops) <= 128

    for index in range(10_000):
        telemetry.emit(
            "command_started",
            state="started",
            reason="command_started",
            dedup_key=f"request-{index}",
        )
    assert len(telemetry._dedup) <= 32

    now[0] = 61.0
    telemetry.emit(
        "command_started",
        state="started",
        reason="command_started",
        dedup_key="fresh",
        command_ref="fresh-ref",
    )
    assert len(telemetry._dedup) == 1
    assert len(telemetry._cardinality) == 1


def test_retained_diagnostics_are_count_and_ttl_bounded():
    from collections import deque

    from tui_gateway import server

    worker = object.__new__(server._SlashWorker)
    worker.stderr_tail = []
    worker._stderr_diagnostics = deque()

    for index in range(100):
        worker._retain_stderr_diagnostic(f"RuntimeError item-{index}", now=float(index))
    assert len(worker.stderr_tail) == 80

    worker._retain_stderr_diagnostic("RuntimeError fresh", now=500.0)
    assert worker.stderr_tail == ["RuntimeError"]


def test_recovery_state_and_counters_advance_only_after_durable_sink_delivery():
    import json

    from tui_gateway.slash_telemetry import PREFIX, SlashTelemetry

    lines: list[str] = []
    fail = [True]

    def sink(line: str) -> None:
        if fail[0]:
            raise OSError("sink unavailable")
        lines.append(line)

    telemetry = SlashTelemetry(
        sink,
        session_key="session",
        profile_home="profile",
        parent_pid=1,
        launcher="python",
        instance_seed="instance",
    )
    assert telemetry.observe_queue(depth=9, oldest_age_seconds=45) is None
    assert telemetry._queue_health == "healthy"

    fail[0] = False
    degraded = telemetry.observe_queue(depth=9, oldest_age_seconds=45)
    assert degraded is not None
    recovered = telemetry.observe_queue(depth=0, oldest_age_seconds=0)
    assert recovered is not None
    assert degraded["fingerprint"] == recovered["fingerprint"]

    telemetry._suppressed_count = 3
    fail[0] = True
    assert telemetry.flush_suppressed() is None
    assert telemetry._suppressed_count == 3
    fail[0] = False
    assert telemetry.flush_suppressed()["suppressed_count"] == 3
    assert json.loads(lines[-1].removeprefix(PREFIX))["suppressed_count"] == 3


def test_restart_incident_is_durable_and_links_to_recovery_fingerprint():
    from tui_gateway.slash_telemetry import SlashTelemetry

    lines: list[str] = []
    fail = [True]

    def sink(line: str) -> None:
        if fail[0]:
            raise OSError("sink unavailable")
        lines.append(line)

    telemetry = SlashTelemetry(
        sink,
        session_key="session",
        profile_home="profile",
        parent_pid=1,
        launcher="python",
        instance_seed="instance",
        clock=lambda: 100.0,
    )
    telemetry.reset_restart_state_for_tests()
    telemetry.record_restart_attempt(threshold=2)
    telemetry.record_restart_attempt(threshold=2)
    assert telemetry.record_ready_recovery() is None

    fail[0] = False
    incident = telemetry.record_restart_attempt(threshold=2)[-1]
    recovery = telemetry.record_ready_recovery()

    assert incident["event"] == "worker_restart_loop"
    assert recovery is not None
    assert incident["fingerprint"] == recovery["fingerprint"]


def test_queue_health_ages_on_deadlines_without_new_arrivals():
    import threading

    from tui_gateway import server

    observations: list[tuple[int, float]] = []
    worker = object.__new__(server._SlashWorker)
    worker._queue_lock = threading.Lock()
    worker._queued_at = {1: 100.0, 2: 120.0}
    worker.telemetry = type(
        "T",
        (),
        {
            "observe_queue": lambda _self, **fields: observations.append(
                (fields["depth"], fields["oldest_age_seconds"])
            )
        },
    )()

    worker._observe_queue_health(now=221.0)
    assert observations == [(2, 121.0)]

    worker._queued_at.pop(1)
    worker._observe_queue_health(now=221.0)
    assert observations[-1] == (1, 101.0)


def test_dispatch_reports_oldest_remaining_queued_item(monkeypatch):
    import queue
    import threading

    from tui_gateway import server

    events: list[tuple[str, dict]] = []

    class _Input:
        def write(self, _value):
            return None

        def flush(self):
            return None

    worker = object.__new__(server._SlashWorker)
    worker._lock = threading.Lock()
    worker._queue_lock = threading.Lock()
    worker._seq = 0
    worker._request_seq = 0
    worker._queued_at = {99: 0.0}
    worker._request_correlations = {}
    worker.proc = type("P", (), {"stdin": _Input(), "poll": lambda self: None})()
    worker.stdout_queue = queue.Queue()
    worker.stdout_queue.put({"id": 1, "ok": True, "output": "ok"})
    worker.stderr_tail = []
    worker.telemetry = type(
        "T",
        (),
        {
            "emit": lambda _self, event, **fields: events.append((event, fields)),
            "observe_queue": lambda *_args, **_kwargs: None,
            "record_correlation_reuse": lambda *_args, **_kwargs: None,
        },
    )()
    monkeypatch.setattr(server.time, "monotonic", lambda: 100.0)

    assert worker.run("/status") == "ok"

    dispatched = next(fields for event, fields in events if event == "command_dispatched")
    assert dispatched["queue_depth_bucket"] == "one"
    assert dispatched["queue_age_bucket"] == "30s_to_2m"


def test_planned_worker_replacement_does_not_count_as_restart_attempt(monkeypatch):
    from tui_gateway import server

    attempts: list[str] = []

    class _Telemetry:
        def record_restart_attempt(self):
            attempts.append("restart")

    class _Worker:
        telemetry = _Telemetry()

        def __init__(self, *_args, **_kwargs):
            return None

        def close(self):
            return None

    monkeypatch.setattr(server, "_SlashWorker", _Worker)
    monkeypatch.setattr(server, "_attach_worker", lambda _sid, session, worker: session.update(slash_worker=worker))
    session = {
        "session_key": "session",
        "slash_worker": _Worker(),
        "agent": type("A", (), {"model": "model"})(),
    }

    server._restart_slash_worker("sid", session)

    assert attempts == []


def test_telemetry_constructor_failure_cannot_block_worker_spawn(monkeypatch):
    from tui_gateway import server, slash_telemetry

    spawned: list[bool] = []

    class _Stream(list):
        def close(self):
            return None

    class _Proc:
        pid = 7
        stdin = _Stream()
        stdout = _Stream()
        stderr = _Stream()

        def poll(self):
            return 0

    class _Thread:
        def __init__(self, **_kwargs):
            return None

        def start(self):
            return None

    def popen(*_args, **_kwargs):
        spawned.append(True)
        return _Proc()

    monkeypatch.setattr(
        slash_telemetry,
        "SlashTelemetry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("telemetry failed")),
    )
    monkeypatch.setattr(server.subprocess, "Popen", popen)
    monkeypatch.setattr(server.threading, "Thread", _Thread)

    worker = server._SlashWorker("session", "model")

    assert spawned == [True]
    assert worker.proc.pid == 7


def test_telemetry_clock_failure_cannot_block_recovery_worker_spawn(monkeypatch):
    from tui_gateway import server, slash_telemetry

    spawned: list[bool] = []

    class _Stream(list):
        def close(self):
            return None

    class _Proc:
        pid = 8
        stdin = _Stream()
        stdout = _Stream()
        stderr = _Stream()

        def poll(self):
            return 0

    class _Thread:
        def __init__(self, **_kwargs):
            return None

        def start(self):
            return None

    real_telemetry = slash_telemetry.SlashTelemetry

    def broken_clock_telemetry(*args, **kwargs):
        kwargs["clock"] = lambda: (_ for _ in ()).throw(RuntimeError("clock failed"))
        return real_telemetry(*args, **kwargs)

    monkeypatch.setattr(slash_telemetry, "SlashTelemetry", broken_clock_telemetry)
    monkeypatch.setattr(
        server.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (spawned.append(True) or _Proc()),
    )
    monkeypatch.setattr(server.threading, "Thread", _Thread)

    worker = server._SlashWorker("session", "model", recovery_restart=True)

    assert spawned
    assert worker.proc.pid == 8


def test_production_empty_command_rejection_is_wired_without_delivery_duplication(monkeypatch):
    from tui_gateway import server

    rejected: list[dict] = []
    telemetry = type(
        "T",
        (),
        {
            "record_command_rejected": lambda _self, **fields: rejected.append(fields),
        },
    )()
    session = {"slash_worker": type("W", (), {"telemetry": telemetry})()}
    monkeypatch.setattr(server, "_sess", lambda _params, _rid: (session, None))

    response = server._methods["slash.exec"](9, {"command": "", "session_id": "sid"})

    assert response["error"]["code"] == 4004
    assert rejected == [{"command": "", "command_ref": 9, "reason": "invalid_request"}]


def test_response_handoff_failure_is_worker_local_and_does_not_duplicate_lifecycle(monkeypatch):
    from tui_gateway import server

    failures: list[dict] = []

    class _Telemetry:
        def record_delivery_handoff_failure(self, **fields):
            failures.append(fields)

    class _Worker:
        telemetry = _Telemetry()
        closed = False

        def run(self, _command):
            return "ok"

        def close(self):
            self.closed = True

    worker = _Worker()
    session = {"slash_worker": worker, "session_key": "session"}
    monkeypatch.setattr(server, "_sess", lambda _params, _rid: (session, None))
    monkeypatch.setattr(
        server,
        "_mirror_slash_side_effects",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("handoff failed")),
    )

    response = server._methods["slash.exec"](11, {"command": "/status", "session_id": "sid"})

    assert response["error"]["code"] == 5030
    assert failures == [
        {"command": "/status", "command_ref": 11, "gateway_event_ref": "slash.exec"}
    ]
    assert worker.closed is False
    assert session["slash_worker"] is worker
    assert "slash_worker_recovery_pending" not in session


def test_real_gui_logger_path_receives_private_bounded_telemetry(tmp_path):
    import hermes_logging
    from tui_gateway import server
    from tui_gateway.slash_telemetry import SlashTelemetry

    hermes_logging.setup_logging(hermes_home=tmp_path, mode="gui", force=True)
    telemetry = SlashTelemetry(
        lambda line: server.logger.info("%s", line),
        session_key="private-session",
        profile_home="C:/Users/alice/private",
        parent_pid=1,
        launcher="C:/private/python.exe",
        instance_seed="private-instance",
    )
    telemetry.emit(
        "command_failed",
        state="failed",
        reason="command_failed",
        command="/status private prompt",
        summary="token=placeholder-secret C:/Users/alice/private.txt",
    )
    hermes_logging.flush_log_queue()

    content = (tmp_path / "logs" / "gui.log").read_text(encoding="utf-8")
    assert "HERMES_TELEMETRY" in content
    assert "command_failed" in content
    for prohibited in ("private-session", "private prompt", "placeholder-secret", "alice"):
        assert prohibited not in content


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
    writes: list[dict] = []

    class _Input:
        def write(self, value):
            writes.append(json.loads(value))
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
                    json.dumps({"id": 2, "ok": True, "output": "second response"}) + "\n",
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
            if self.target.__name__ != "_monitor_queue_health":
                self.target()

    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **kw: _Proc())
    monkeypatch.setattr(server.threading, "Thread", _Thread)
    monkeypatch.setattr(server.logger, "info", lambda _fmt, line: telemetry_lines.append(line))
    monkeypatch.setenv("HERMES_HOME", r"C:\Users\alice\AppData\Local\hermes-desktop")

    worker = server._SlashWorker("session-private", "model-private")
    assert worker.run("/status private argument", request_ref="rpc-private-1") == "private response"
    assert worker.run("/status private argument", request_ref="rpc-private-1") == "second response"
    worker.close()

    assert [write["id"] for write in writes] == [1, 2]

    records = [json.loads(line.removeprefix("HERMES_TELEMETRY ")) for line in telemetry_lines]
    events = [record["event"] for record in records]
    assert events == [
        "worker_starting", "worker_initialized", "worker_ready", "worker_crashed",
        "command_received",
        "command_dispatched", "command_started", "command_completed",
        "command_correlation_reused", "command_received", "command_dispatched",
        "command_started", "command_completed", "worker_stopping", "worker_stopped",
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
        "exit_code": 1,
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
