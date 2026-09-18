"""One name for a good ending, across the writer, the store and the console.

No Postgres: these are the naming facts that the DB-backed integrator tests
skip past when POSTGRES_HOST is unset, which is exactly how the mismatch
survived.
"""

import pathlib

from wisefood_mcp.stores import APPROVABLE_FROM, STATUSES


SERVICE = (
    pathlib.Path(__file__).parent.parent / "src" / "integrator" / "service.py"
).read_text()


def test_imported_is_the_good_ending_and_integrated_is_not_a_status():
    assert "imported" in STATUSES
    assert "integrated" not in STATUSES


def test_the_writer_uses_the_name_the_store_declares():
    # It wrote "integrated", which is in neither STATUSES nor the console's
    # stageOf(), so a successful integration rendered as "dropped" with no
    # status label at all.
    assert 'status="imported" if outcome["status"] == "succeeded"' in SERVICE
    assert 'status="integrated"' not in SERVICE


def test_a_failed_proposal_is_approvable_so_a_retry_has_a_route():
    # start_integration requires `approved`; a finished run leaves `failed`.
    # Without this the proposal is stuck between the two refusals.
    assert "failed" in APPROVABLE_FROM
    for terminal in ("rejected", "imported"):
        assert terminal not in APPROVABLE_FROM


def test_the_approval_gate_is_checked_before_the_writes_switch():
    # The 403 a curator actually sees should name the real blocker. Status is
    # checked first, so "not been approved" never means "writes are off".
    approved_at = SERVICE.index('raise PermissionError("this proposal has not been approved")')
    writes_at = SERVICE.index("catalog writes are switched off")
    assert approved_at < writes_at
