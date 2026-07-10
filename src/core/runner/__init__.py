"""core.runner — the developer cockpit.

Drives one question (or the whole training battery) through the full pipeline
(:class:`~core.fsm.controller.QuestionController` wired by
:func:`core.heads.build_callables` over a live :class:`SceneIndex`) on the injected
*simulated* clock, then reports.

* :func:`core.runner.single.run_question` — one question, one io, structured
  :class:`RunResult`.
* :mod:`core.runner.battery` — the 75 training questions against per-scene
  synthetic scenes, scored for STRUCTURAL health, emitting a markdown + json report.

Everything runs offline (regex parse tier / no network) and deterministically.
"""
from core.runner.single import RunResult, run_question

__all__ = ["RunResult", "run_question"]
