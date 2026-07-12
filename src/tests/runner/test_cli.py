"""CLI main() smoke tests for both entry points."""
from __future__ import annotations

import pytest

from core.runner.__main__ import main as single_main
from core.runner.battery import main as battery_main


def test_single_cli_smoke(capsys):
    rc = single_main(["How many chairs are near the table?"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "qtype" in out
    assert "answer" in out
    assert "published: True" in out


@pytest.mark.slow
def test_single_cli_verbose_dumps_flight_log(capsys):
    rc = single_main(["Find the vase on the table", "--verbose"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "flight log:" in out


@pytest.mark.slow
def test_battery_cli_smoke(capsys, tmp_path):
    rc = battery_main(["--scenes", "arabic_room", "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "battery:" in out
    assert (tmp_path / "battery_report.md").exists()
    assert (tmp_path / "battery_results.json").exists()
