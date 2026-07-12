# Dashboard slash-worker operational events

The native dashboard owns a persistent `tui_gateway.slash_worker` subprocess for
slash commands. Its operational events are emitted as one-line JSON records to
the dashboard's normal GUI log using the stable `HERMES_TELEMETRY ` marker and
the `hermes.dashboard.slash_worker.v1` schema. This keeps the records on the
existing host-log path used by local operators and log collectors; telemetry
failure is fail-open and never changes command or shutdown behavior.

## Privacy and cardinality contract

Records contain only:

- fixed lifecycle, state, reason, severity, command-category, and bucket enums;
- bounded counters and exit codes;
- stable fingerprints; and
- SHA-256-derived opaque references for the profile, launcher, parent process,
  worker instance, session, command, and an optional gateway event.

They never contain command arguments, prompts, slash-command text, message or
response content, model/API tokens, credentials, authorization headers,
usernames, home paths, or raw URLs. Free-form exception and stderr text passes
through an allowlist-only summarizer before it reaches a record or the retained
stderr tail. Every wire record is bounded to 2048 bytes. Dynamic references
have a fixed cardinality budget; duplicate events, bursts, and over-budget
references are suppressed or collapsed, with bounded suppression counters.

## State transitions

The schema covers worker starting, initialization, readiness, graceful or forced
shutdown, shutdown failure, abrupt exit, and restart-loop recovery. Command
states cover accepted, rejected, deduplicated, dispatched, started, completed,
failed, timed out, and abandoned. Queue health uses fixed depth, oldest-age, and
latency buckets and emits edge-triggered pressure, stall, and recovery records.

`fingerprint` is stable for the event/state/reason/session tuple. Consumers
should open an incident on degraded, stalled, failed, timed-out, abandoned, or
crashed states and resolve it on the corresponding recovered/ready/stopped
transition. Restart-loop and queue-health transitions are emitted only on state
edges to avoid alert churn.

## Ownership boundary

These records own only dashboard/slash-worker semantics:

- They do **not** duplicate generic process-supervisor lifecycle events. Parent,
  launcher, and instance references are correlation fields only.
- They do **not** claim channel or gateway delivery. A worker-local response or
  delivery-handoff failure may reference an opaque gateway event ID, while the
  gateway/channel event remains authoritative for actual delivery.
- They do **not** instrument Desktop shell, WebUI, MCP, or Telegram domain
  events.

Collectors must parse only lines with the exact marker and schema, preserve the
bounded record as structured details, and must not enrich it with raw process
arguments, environment values, message text, or filesystem paths.
