"""CLI --detections flag: validation + wiring through to run_question."""
from __future__ import annotations

import pytest

from core.runner.__main__ import main as single_main


def test_detections_without_fixtures_errors_cleanly(capsys):
    """--detections is only valid with --fixtures; argparse exits 2 with a clear message."""
    with pytest.raises(SystemExit) as ei:
        single_main(["How many stools are in the room?", "--detections", "labels.json"])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "--detections requires --fixtures" in err
