"""core.heads — answer heads + the factory wiring them into the QuestionController.

The factory (:func:`core.heads.factory.build_callables`) turns a live SceneIndex + nav
stack into the ``{parse, explore, verify, probe}`` callables the FSM injects, implementing
the architecture §4 per-type strategies (numerical / object-reference / instruction).
"""
from core.heads.factory import HeadState, build_callables

__all__ = ["HeadState", "build_callables"]
