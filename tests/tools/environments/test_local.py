"""Regression tests for the ``_resolve_safe_cwd`` self-heal fix.

The parent incident is secops kanban t_9db280e4: a kanban handoff card
was assigned the referring card's scratch workspace path, the referring
card completed on its own timeline, and the receiving session's cwd
disappeared from under it.  The old ``_resolve_safe_cwd`` walked up the
parent chain and returned the shared ``.../workspaces/`` directory,
which every card on the board shares — silently converting any
subsequent relative ``write_file`` into a cross-card leak.  Only a
WARNING was emitted (see the referring session's ``errors.log``), which
no downstream agent had any way of surfacing.

The fix has three observable behaviours, one test each:

- **Self-heal.** A missing cwd whose parent still exists is recreated in
  place; the returned path equals the input path and ``os.path.isdir``
  is true after the call.  This is the anti-leak path.
- **Un-recreatable fallback.** When ``os.makedirs`` fails (permissions,
  ``/dev/null/…``, broken mount), the walk-up ancestor fallback still
  fires so ``subprocess.Popen`` doesn't crash with ``FileNotFoundError``
  before bash starts — the regression protection for issue #17558 stays
  intact.  The fallback is logged at **ERROR**, not WARNING, so
  downstream tooling can surface it.
- **#17558 stays fixed.** The classic self-deletion pattern (a tool call
  ``rm -rf``'d its own working directory) still returns a directory
  that exists on disk, so the subsequent ``Popen(..., cwd=...)``
  succeeds.  Under the new behaviour that directory is the ORIGINAL
  cwd (self-heal), not an ancestor — which is a strengthening of the
  guarantee, not a regression.
"""

import logging
import os

import pytest

from tools.environments.local import _resolve_safe_cwd


class TestSelfHealHappyPath:
    """A missing cwd under an existing parent is re-materialised in place."""

    def test_recreates_missing_path_in_place(self, tmp_path):
        # Simulate a kanban scratch workspace whose completion cleanup
        # removed the directory out from under a still-running sibling
        # session.  Parent tmp_path is intact (like a real
        # ``.../workspaces/`` root); ``missing`` is the receiver's cwd.
        missing = tmp_path / "t_child_scratch"
        assert not missing.exists()

        result = _resolve_safe_cwd(str(missing))

        # The exact path is returned — self-heal, not walk-up-to-parent.
        assert result == str(missing)
        assert os.path.isdir(result), (
            "expected _resolve_safe_cwd to recreate the missing directory "
            "in place so relative writes stay isolated to this session; "
            "walking up to the shared parent would leak across cards "
            "(secops t_9db280e4)."
        )

    def test_self_heal_returns_input_not_an_ancestor(self, tmp_path):
        """Explicit anti-regression: the returned value must NOT be a
        parent ancestor even when that ancestor exists and is writable.
        The whole point of the fix is that walking up to the shared
        parent is the leak."""
        missing = tmp_path / "deep" / "nested" / "scratch"

        result = _resolve_safe_cwd(str(missing))

        assert result == str(missing)
        assert result != str(tmp_path)
        assert result != str(missing.parent)


class TestUnrecreatableFallback:
    """When ``os.makedirs`` raises, the walk-up ancestor fallback fires and
    logs at ERROR so downstream log analysis can surface the situation."""

    def test_returns_ancestor_when_makedirs_raises(
        self, tmp_path, monkeypatch, caplog,
    ):
        # A path whose parent exists but which cannot be recreated (we
        # simulate the failure by monkeypatching ``os.makedirs``).
        missing = tmp_path / "unrecreatable_scratch"

        real_makedirs = os.makedirs

        def raising_makedirs(path, *args, **kwargs):
            if os.path.abspath(path) == os.path.abspath(str(missing)):
                raise PermissionError("simulated: cannot recreate cwd")
            return real_makedirs(path, *args, **kwargs)

        monkeypatch.setattr(os, "makedirs", raising_makedirs)

        with caplog.at_level(logging.ERROR, logger="tools.environments.local"):
            result = _resolve_safe_cwd(str(missing))

        # Popen would receive this — must be a real existing directory.
        assert result == str(tmp_path)
        assert os.path.isdir(result)

        # An ERROR-level log record surfaces the fallback.  Downstream
        # tooling grepping ``errors.log`` for cross-card workspace races
        # relies on this level.
        fallback_records = [
            rec for rec in caplog.records
            if "un-recreatable" in rec.message
            and "falling back to ancestor" in rec.message
        ]
        assert fallback_records, (
            "expected an ERROR log describing the un-recreatable "
            "fallback; nothing matched. Records: "
            + repr([rec.message for rec in caplog.records])
        )
        assert all(rec.levelname == "ERROR" for rec in fallback_records), (
            "the un-recreatable fallback MUST log at ERROR, not WARNING — "
            "the WARNING-level fallback was the bug this card fixes "
            "(secops t_9db280e4)."
        )

    def test_self_heal_success_also_logs_at_error(
        self, tmp_path, caplog,
    ):
        """The self-heal happy path is also observable in the log at
        ERROR (with a distinct message).  Silent recovery is exactly the
        anti-pattern this card removes — downstream log analysis must be
        able to spot repeated races even when the session recovered."""
        missing = tmp_path / "scratch_to_heal"

        with caplog.at_level(logging.ERROR, logger="tools.environments.local"):
            result = _resolve_safe_cwd(str(missing))

        assert result == str(missing)
        assert os.path.isdir(result)

        heal_records = [
            rec for rec in caplog.records
            if "recreated in place" in rec.message
        ]
        assert heal_records, (
            "expected an ERROR log describing the self-heal recovery; "
            "nothing matched. Records: "
            + repr([rec.message for rec in caplog.records])
        )
        assert all(rec.levelname == "ERROR" for rec in heal_records)


class TestIssue17558RegressionStaysFixed:
    """The classic ``rm -rf`` self-deletion pattern still yields a live
    directory for ``Popen(..., cwd=...)``.  This is the guarantee issue
    #17558 originally added; the self-heal path preserves it (and in
    fact strengthens it — the returned directory is now the ORIGINAL
    cwd, not an ancestor)."""

    def test_deleted_own_cwd_still_returns_live_directory(self, tmp_path):
        wedged = tmp_path / "wedge_repro"
        wedged.mkdir()
        # Emulate #17558: the tool call ran ``rm -rf $PWD``.
        wedged.rmdir()
        assert not wedged.exists()

        result = _resolve_safe_cwd(str(wedged))

        # Whatever the resolver returns, Popen must be able to spawn
        # bash there — that's the whole guarantee.
        assert os.path.isdir(result), (
            "issue #17558 regression: _resolve_safe_cwd must return a "
            "directory that exists on disk so subprocess.Popen doesn't "
            "raise FileNotFoundError before bash starts."
        )

    def test_deleted_own_cwd_returns_self_healed_original_path(self, tmp_path):
        """Stronger form of the #17558 guarantee under the new
        behaviour: the returned directory is the ORIGINAL wedged path
        (self-heal), not an ancestor.  Prevents a same-session
        follow-up write from silently landing under a different
        directory than the tool call expected."""
        wedged = tmp_path / "wedge_repro"
        wedged.mkdir()
        wedged.rmdir()

        result = _resolve_safe_cwd(str(wedged))

        assert result == str(wedged)
        assert result != str(tmp_path)


class TestNoOpWhenCwdIsFine:
    """The happy path must not log spurious ERROR records; a healthy
    cwd is returned unchanged with zero log traffic."""

    def test_returns_input_unchanged_and_silent(self, tmp_path, caplog):
        cwd = str(tmp_path)

        with caplog.at_level(logging.WARNING, logger="tools.environments.local"):
            result = _resolve_safe_cwd(cwd)

        assert result == cwd
        assert not caplog.records, (
            "healthy cwd path must be silent — noise on every terminal "
            "call would swamp real races. Records: "
            + repr([(rec.levelname, rec.message) for rec in caplog.records])
        )


# Belt-and-braces: assert the helper is importable at the location the
# card body pins it to.  If a future refactor moves it, this test will
# fail loudly rather than a downstream tool silently importing a stub.
def test_helper_location_stable():
    from tools.environments import local as local_mod

    assert local_mod._resolve_safe_cwd is _resolve_safe_cwd
