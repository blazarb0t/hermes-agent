"""Same-tick, privacy-safe gateway dispatcher health classification."""

import logging

import pytest

from hermes_cli.kanban_db import DispatchResult
from gateway.kanban_watchers import _dispatch_no_spawn_diagnostic
from gateway.kanban_watchers import GatewayKanbanWatchersMixin


def classify(result: DispatchResult):
    return _dispatch_no_spawn_diagnostic([("private-board-slug", result)])


def test_per_profile_cap_is_benign_and_counted():
    result = DispatchResult(ready_count=2)
    result.skipped_per_profile_capped = [
        ("secret-1", "profile", 3),
        ("secret-2", "profile", 3),
    ]
    assert classify(result) == ("benign", "ready=2 profile_capped=2")


def test_nonspawnable_is_benign_and_counted():
    result = DispatchResult(ready_count=1, skipped_nonspawnable=["secret-task"])
    assert classify(result) == ("benign", "ready=1 nonspawnable=1")


def test_respawn_guard_reasons_are_stable_counts():
    result = DispatchResult(ready_count=2)
    result.respawn_guarded = [("secret-1", "recent_success"), ("secret-2", "active_pr")]
    assert classify(result) == (
        "benign",
        "ready=2 guard_recent_success=1 guard_active_pr=1",
    )


def test_locked_and_memory_pressure_are_benign():
    locked = DispatchResult(skipped_locked=True)
    pressure = DispatchResult(ready_count=4, memory_pressure="critical")
    assert _dispatch_no_spawn_diagnostic([("a", locked), ("b", pressure)]) == (
        "benign",
        "ready=4 locked=1 memory_critical=1",
    )


def test_capacity_is_benign_and_counted():
    result = DispatchResult(ready_count=5, skipped_capacity=5)
    assert classify(result) == ("benign", "ready=5 capacity=5")


def test_unexplained_ready_work_is_actionable():
    assert classify(DispatchResult(ready_count=3)) == (
        "actionable",
        "ready=3 unexplained=3",
    )


def test_unassigned_and_breaker_are_actionable():
    result = DispatchResult(ready_count=2, skipped_unassigned=["private"])
    result.auto_blocked = ["also-private"]
    assert classify(result) == ("actionable", "ready=2 unassigned=1 circuit_breaker=1")


def test_auth_guard_takes_precedence_over_benign_causes():
    result = DispatchResult(ready_count=2, skipped_nonspawnable=["private"])
    result.respawn_guarded = [("secret", "blocker_auth")]
    assert classify(result) == (
        "actionable",
        "ready=2 nonspawnable=1 guard_blocker_auth=1",
    )


def test_unexplained_board_takes_precedence_over_benign_other_board():
    benign = DispatchResult(ready_count=1, skipped_nonspawnable=["private"])
    unexplained = DispatchResult(ready_count=2)
    assert _dispatch_no_spawn_diagnostic([("a", benign), ("b", unexplained)]) == (
        "actionable",
        "ready=3 nonspawnable=1 unexplained=2",
    )


def test_summary_never_contains_sensitive_dispatch_values():
    result = DispatchResult(
        ready_count=1,
        skipped_per_profile_capped=[("task-secret", "profile-secret", 9)],
    )
    state, summary = _dispatch_no_spawn_diagnostic([("board-secret", result)])
    assert state == "benign"
    assert "secret" not in summary


@pytest.mark.asyncio
async def test_watcher_warning_is_driven_by_same_tick_dispatch_result(
    monkeypatch,
    caplog,
    tmp_path,
):
    """Kill test: discarding DispatchResult plumbing makes this stay greenless."""
    from hermes_cli import config, kanban_db as kb
    import gateway.kanban_watchers as watchers

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
                "auto_decompose": False,
            }
        },
    )
    monkeypatch.setattr(
        watchers, "_acquire_singleton_lock", lambda _path: (object(), "held")
    )
    monkeypatch.setattr(kb, "kanban_home", lambda: tmp_path)
    monkeypatch.setattr(
        kb, "kanban_db_path", lambda _slug=None, **_kw: tmp_path / "board.db"
    )
    monkeypatch.setattr(kb, "list_boards", lambda **_kw: [{"slug": "private-board"}])
    monkeypatch.setattr(kb, "resolve_max_in_progress", lambda value: value)
    monkeypatch.setattr(kb, "reap_worker_zombies", lambda: [])

    class Conn:
        def close(self):
            pass

    monkeypatch.setattr(kb, "connect", lambda **_kw: Conn())
    monkeypatch.setattr(
        kb, "dispatch_once", lambda *_a, **_kw: DispatchResult(ready_count=1)
    )

    class Runner(GatewayKanbanWatchersMixin):
        _running = True

        def _release_kanban_dispatcher_lock(self):
            pass

    runner = Runner()
    sleeps = 0

    async def bounded_sleep(_seconds):
        nonlocal sleeps
        sleeps += 1
        # One initial delay plus six health-window ticks.
        if sleeps >= 7:
            runner._running = False

    monkeypatch.setattr(watchers.asyncio, "sleep", bounded_sleep)
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await runner._kanban_dispatcher_watcher()

    warnings = [r.message for r in caplog.records if "dispatcher stuck" in r.message]
    assert len(warnings) == 1
    assert "ready=1 unexplained=1" in warnings[0]


@pytest.mark.asyncio
async def test_benign_tick_resets_watcher_actionable_streak(
    monkeypatch,
    caplog,
    tmp_path,
):
    """A benign deferral breaks the streak; a fresh full window still warns."""
    from hermes_cli import config, kanban_db as kb
    import gateway.kanban_watchers as watchers

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
                "auto_decompose": False,
            }
        },
    )
    monkeypatch.setattr(
        watchers, "_acquire_singleton_lock", lambda _path: (object(), "held")
    )
    monkeypatch.setattr(kb, "kanban_home", lambda: tmp_path)
    monkeypatch.setattr(
        kb, "kanban_db_path", lambda _slug=None, **_kw: tmp_path / "board.db"
    )
    monkeypatch.setattr(kb, "list_boards", lambda **_kw: [{"slug": "board"}])
    monkeypatch.setattr(kb, "resolve_max_in_progress", lambda value: value)
    monkeypatch.setattr(kb, "reap_worker_zombies", lambda: [])

    class Conn:
        def close(self):
            pass

    monkeypatch.setattr(kb, "connect", lambda **_kw: Conn())
    def actionable():
        return DispatchResult(ready_count=1)

    benign = DispatchResult(ready_count=1, skipped_capacity=1)
    tick_results = [*(actionable() for _ in range(5)), benign,
                    *(actionable() for _ in range(6))]

    def dispatch_once(*_args, **_kwargs):
        return tick_results.pop(0)

    monkeypatch.setattr(kb, "dispatch_once", dispatch_once)

    class Runner(GatewayKanbanWatchersMixin):
        _running = True

        def _release_kanban_dispatcher_lock(self):
            pass

    runner = Runner()
    sleeps = 0

    async def bounded_sleep(_seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps >= 13:  # initial delay plus twelve dispatch ticks
            runner._running = False

    monkeypatch.setattr(watchers.asyncio, "sleep", bounded_sleep)
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await runner._kanban_dispatcher_watcher()

    warnings = [r.message for r in caplog.records if "dispatcher stuck" in r.message]
    assert len(warnings) == 1
    assert "6 consecutive actionable ticks" in warnings[0]
