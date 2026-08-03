"""RemoteDetector: the dev-only GDINO offload seam (issue #86) — wire contract, dual-pass
tick scheduling, prompt-refresh compatibility, and failure/backoff containment against an
in-process fake HTTP server standing in for tools/cluster/gdino_server.py."""
from __future__ import annotations

import http.server
import json
import threading
import time

import numpy as np
import pytest

from core.perception.detector import GDINO_BACKOFF_CAP_S, GDINO_BACKOFF_DEGRADE_N, refresh_prompt
from core.perception.remote_detector import (
    DEFAULT_REMOTE_DETECTOR_TIMEOUT_S,
    ENV_REMOTE_DETECTOR_TIMEOUT_S,
    ENV_REMOTE_DETECTOR_URL,
    RemoteDetector,
)


def _fake_encode_fn(arr) -> bytes:
    return b"JPEGBYTES"


def _tiles(n=1):
    return [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(n)]


class _FakeClock:
    """Manually-advanced monotonic clock stand-in (mirrors test_detector.py's)."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> float:
        self._t += float(dt)
        return self._t


class _FakeGdinoServer:
    """In-process stand-in for tools/cluster/gdino_server.py.

    Records every request's (path, parsed JSON body) and replies with a configurable canned
    response. ``response`` is a callable ``(body: dict) -> (status, payload_dict)`` so a test
    can vary the reply per request (e.g. distinct boxes per caption).
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []
        self.response = lambda body: (200, {"per_tile": [[] for _ in body.get("tiles", [])]})
        self.sleep_s = 0.0
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - stdlib method name
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                body = json.loads(raw) if raw else {}
                server.requests.append((self.path, body))
                if server.sleep_s:
                    time.sleep(server.sleep_s)
                status, payload = server.response(body)
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, fmt, *args):  # silence stdout during tests
                pass

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture()
def fake_server():
    server = _FakeGdinoServer()
    yield server
    server.stop()


def _detector(fake_server, **kwargs) -> RemoteDetector:
    kwargs.setdefault("encode_fn", _fake_encode_fn)
    kwargs.setdefault("timeout_s", 2.0)
    return RemoteDetector(fake_server.url, **kwargs)


# ------------------------------------------------------------------ construction


def test_url_required(monkeypatch):
    monkeypatch.delenv(ENV_REMOTE_DETECTOR_URL, raising=False)
    with pytest.raises(RuntimeError) as exc:
        RemoteDetector()
    assert ENV_REMOTE_DETECTOR_URL in str(exc.value)


def test_url_from_env(monkeypatch):
    monkeypatch.setenv(ENV_REMOTE_DETECTOR_URL, "http://127.0.0.1:9999")
    assert RemoteDetector(encode_fn=_fake_encode_fn).url == "http://127.0.0.1:9999"


def test_timeout_precedence(monkeypatch):
    monkeypatch.delenv(ENV_REMOTE_DETECTOR_TIMEOUT_S, raising=False)
    det = RemoteDetector("http://x", encode_fn=_fake_encode_fn)
    assert det.timeout_s == DEFAULT_REMOTE_DETECTOR_TIMEOUT_S
    monkeypatch.setenv(ENV_REMOTE_DETECTOR_TIMEOUT_S, "3.5")
    assert RemoteDetector("http://x", encode_fn=_fake_encode_fn).timeout_s == 3.5
    assert RemoteDetector("http://x", timeout_s=7.0, encode_fn=_fake_encode_fn).timeout_s == 7.0


# ------------------------------------------------------------------ wire round-trip


def test_wire_round_trip_routes_detections_by_tile(fake_server):
    def response(body):
        return 200, {
            "per_tile": [
                [{"bbox_xyxy": [1.0, 2.0, 3.0, 4.0], "label": "teapot", "score": 0.9}],
                [{"bbox_xyxy": [5.0, 6.0, 7.0, 8.0], "label": "table", "score": 0.5}],
            ]
        }

    fake_server.response = response
    det = _detector(fake_server, vocab_pass_cadence=1)
    det.question_prompt = "teapot ."
    out = det(_tiles(2))

    assert [d.label for d in out[0]] == ["teapot"]
    assert out[0][0].bbox_xyxy == (1.0, 2.0, 3.0, 4.0)
    assert out[0][0].score == pytest.approx(0.9)
    assert out[0][0].tile_id == 0
    assert [d.label for d in out[1]] == ["table"]
    assert out[1][0].tile_id == 1

    assert len(fake_server.requests) == 1
    path, body = fake_server.requests[0]
    assert path == "/detect"
    assert body["tiles"] == ["SlBFR0JZVEVT", "SlBFR0JZVEVT"]  # base64("JPEGBYTES") x2
    assert body["caption"] == "teapot ."
    assert body["box_threshold"] == pytest.approx(det.question_box_threshold)
    assert body["text_threshold"] == pytest.approx(det.text_threshold)


# ------------------------------------------------------------------ dual-pass cadence parity


def test_dual_pass_cadence_parity(fake_server):
    det = _detector(fake_server, vocab_pass_cadence=3)
    det.prompt = "table ."
    det.question_prompt = "teapot ."

    for _ in range(6):
        det(_tiles(1))

    # ticks 0 and 3 -> question + vocab (2 POSTs each); ticks 1,2,4,5 -> question only.
    assert len(fake_server.requests) == 2 + 1 + 1 + 2 + 1 + 1

    captions_per_tick: list[list[str]] = []
    idx = 0
    per_tick_counts = [2, 1, 1, 2, 1, 1]
    for count in per_tick_counts:
        captions_per_tick.append([fake_server.requests[idx + i][1]["caption"] for i in range(count)])
        idx += count

    assert captions_per_tick[0] == ["teapot .", "table ."]  # tick 0: question then vocab
    assert captions_per_tick[1] == ["teapot ."]
    assert captions_per_tick[2] == ["teapot ."]
    assert captions_per_tick[3] == ["teapot .", "table ."]  # tick 3: question then vocab
    assert captions_per_tick[4] == ["teapot ."]
    assert captions_per_tick[5] == ["teapot ."]

    # box_threshold values match the pass they belong to.
    tick0_thresholds = [fake_server.requests[i][1]["box_threshold"] for i in range(2)]
    assert tick0_thresholds == [pytest.approx(det.question_box_threshold), pytest.approx(det.box_threshold)]


# ------------------------------------------------------------------ #91: split box thresholds


def test_marginal_score_admitted_for_question_noun_cut_for_vocab_noun_remote(fake_server):
    """Issue #91 synthetic case, remote path (#134's lesson: both paths must agree). The
    fake server enforces the threshold it was sent — exactly like the real gdino_server.py
    (which forwards ``box_threshold`` straight into
    ``GroundingDinoDetector.run_caption_pass`` -> ``_decode_batch_item``'s
    ``max_logits > box_threshold`` filter) — so this exercises the same admit/cut
    semantics as the local-path test, not just that the two request bodies carry
    different threshold values (already covered by test_dual_pass_cadence_parity)."""
    RAW_SCORE = 0.18

    def response(body):
        if RAW_SCORE <= body["box_threshold"]:
            return 200, {"per_tile": [[] for _ in body["tiles"]]}
        label = "candle holder" if body["caption"] == "candle holder ." else "chair"
        return 200, {
            "per_tile": [[{"bbox_xyxy": [0.0, 0.0, 1.0, 1.0], "label": label, "score": RAW_SCORE}]]
        }

    fake_server.response = response
    det = _detector(fake_server, vocab_pass_cadence=1, question_box_threshold=0.15)
    assert det.box_threshold == 0.35  # vocab default, cuts the 0.18 score
    det.question_prompt = "candle holder ."
    det.prompt = "chair ."

    out = det(_tiles(1))

    labels = [d.label for d in out[0]]
    assert "candle holder" in labels  # question-pass threshold (0.15) admits 0.18
    assert "chair" not in labels  # vocab-pass threshold (0.35) cuts the identical 0.18 score


# ------------------------------------------------------------------ refresh_prompt compat


def test_refresh_prompt_updates_remote_detector_in_place(fake_server):
    det = _detector(fake_server)
    assert det.prompt == "" and det.question_prompt == ""
    new_prompt = refresh_prompt(det, ["teapot"], ["table"])
    assert new_prompt == "teapot . table ."
    assert det.prompt == "teapot . table ."
    assert det.question_prompt == "teapot ."


# ------------------------------------------------------------------ empty prompts


def test_empty_prompts_make_zero_requests_and_zero_encode_calls(fake_server):
    calls = {"n": 0}

    def counting_encode(arr) -> bytes:
        calls["n"] += 1
        return b"X"

    det = RemoteDetector(fake_server.url, encode_fn=counting_encode, timeout_s=2.0)
    assert det.prompt == "" and det.question_prompt == ""
    out = det(_tiles(3))
    assert out == [[], [], []]
    assert calls["n"] == 0
    assert fake_server.requests == []


# ------------------------------------------------------------------ failure containment


def test_server_500_returns_empty_no_raise(fake_server):
    fake_server.response = lambda body: (500, {"error": "boom"})
    det = _detector(fake_server)
    det.question_prompt = "teapot ."
    out = det(_tiles(2))
    assert out == [[], []]
    assert det._consecutive_failures == 1


def test_connection_refused_returns_empty_no_raise():
    # Point at a closed local port — nothing is listening.
    det = RemoteDetector("http://127.0.0.1:1", encode_fn=_fake_encode_fn, timeout_s=1.0)
    det.question_prompt = "teapot ."
    out = det(_tiles(2))
    assert out == [[], []]
    assert det._consecutive_failures == 1


def test_timeout_returns_empty_no_raise(fake_server):
    fake_server.sleep_s = 0.3
    det = _detector(fake_server, timeout_s=0.05)
    det.question_prompt = "teapot ."
    out = det(_tiles(1))
    assert out == [[]]
    assert det._consecutive_failures == 1


def test_backoff_schedule_and_no_request_during_cooldown(fake_server):
    fake_server.response = lambda body: (500, {"error": "boom"})
    clock = _FakeClock()
    det = _detector(fake_server, clock=clock)
    det.question_prompt = "teapot ."
    tiles = _tiles(1)

    det(tiles)
    assert len(fake_server.requests) == 1  # attempt 1 (fails)
    assert det._consecutive_failures == 1

    clock.advance(0.5)  # < 1s backoff
    det(tiles)
    assert len(fake_server.requests) == 1  # still cooling down, no request sent

    clock.advance(0.5)  # at the 1s boundary
    det(tiles)
    assert len(fake_server.requests) == 2  # cooldown elapsed -> retried (and failed again)
    assert det._consecutive_failures == 2

    clock.advance(1.9)  # < 2s backoff
    det(tiles)
    assert len(fake_server.requests) == 2

    clock.advance(0.1)  # at the 2s boundary
    det(tiles)
    assert len(fake_server.requests) == 3
    assert det._consecutive_failures == 3


def test_backoff_degrades_at_n_and_caps(fake_server, caplog):
    fake_server.response = lambda body: (500, {"error": "boom"})
    clock = _FakeClock()
    det = _detector(fake_server, clock=clock)
    det.question_prompt = "teapot ."
    tiles = _tiles(1)

    with caplog.at_level("WARNING", logger="core.perception.remote_detector"):
        for _ in range(GDINO_BACKOFF_DEGRADE_N):
            det(tiles)
            wait = det._next_retry_at - clock()
            if wait > 0:
                clock.advance(wait)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(warnings) == GDINO_BACKOFF_DEGRADE_N - 1
    assert len(errors) == 1
    assert "degraded" in errors[0].message
    assert det._consecutive_failures == GDINO_BACKOFF_DEGRADE_N

    caplog.clear()
    det(tiles)  # still within the (not-yet-advanced) cooldown -> no request, no new log
    assert det._next_retry_at - clock() == pytest.approx(GDINO_BACKOFF_CAP_S)
    assert caplog.records == []


# ------------------------------------------------------------------ issue #131 / #134: cross-tile NMS parity


def test_same_label_near_duplicates_collapse_through_remote_path(fake_server):
    """Ported from #131 (see test_detector.py's
    test_same_tile_duplicate_boxes_collapse_to_the_higher_score): the question pass and
    the vocab pass can each return their own near-identical box for the same physical
    object. RemoteDetector.__call__ must run the union through
    suppress_cross_tile_duplicates before returning, exactly like
    GroundingDinoDetector.__call__ — only the stronger box should survive."""

    def response(body):
        if body["caption"] == "teapot .":
            return 200, {
                "per_tile": [[{"bbox_xyxy": [100.0, 200.0, 180.0, 300.0], "label": "chair", "score": 0.55}]]
            }
        return 200, {
            "per_tile": [[{"bbox_xyxy": [102.0, 198.0, 178.0, 302.0], "label": "chair", "score": 0.31}]]
        }

    fake_server.response = response
    det = _detector(fake_server, vocab_pass_cadence=1)
    det.question_prompt = "teapot ."
    det.prompt = "chair ."

    out = det(_tiles(1))

    assert sum(len(t) for t in out) == 1
    assert out[0][0].score == pytest.approx(0.55)


def test_distinct_same_class_objects_survive_through_remote_path(fake_server):
    """Ported from #131 (see test_detector.py's test_distinct_bearings_both_survive):
    two clearly separate same-label boxes must not be merged by the ported
    suppression — both must reach the caller."""

    def response(body):
        return 200, {
            "per_tile": [
                [
                    {"bbox_xyxy": [10.0, 300.0, 60.0, 400.0], "label": "chair", "score": 0.5},
                    {"bbox_xyxy": [400.0, 300.0, 460.0, 400.0], "label": "chair", "score": 0.45},
                ]
            ]
        }

    fake_server.response = response
    det = _detector(fake_server)
    det.question_prompt = "chair ."

    out = det(_tiles(1))

    assert sorted(d.score for d in out[0]) == [pytest.approx(0.45), pytest.approx(0.5)]


# ------------------------------------------------------------------ recovery


def test_recovery_resets_backoff(fake_server):
    fake_server.response = lambda body: (500, {"error": "boom"})
    clock = _FakeClock()
    det = _detector(fake_server, clock=clock)
    det.question_prompt = "teapot ."
    tiles = _tiles(1)

    det(tiles)
    clock.advance(1.0)
    det(tiles)
    assert det._consecutive_failures == 2

    clock.advance(2.0)  # past the second cooldown
    fake_server.response = lambda body: (200, {"per_tile": [[{"bbox_xyxy": [0, 0, 1, 1], "label": "teapot", "score": 0.7}]]})
    out = det(tiles)
    assert [d.label for d in out[0]] == ["teapot"]
    assert det._consecutive_failures == 0
    assert det._next_retry_at == 0.0

    # Next failure waits the base delay again, not the capped/degraded one.
    fake_server.response = lambda body: (500, {"error": "boom"})
    n_before = len(fake_server.requests)
    det(tiles)
    assert len(fake_server.requests) == n_before + 1
    assert det._consecutive_failures == 1
    assert det._next_retry_at - clock() == pytest.approx(1.0)  # GDINO_BACKOFF_BASE_S


# ------------------------------------------------------------------ #172: caption-label guard


def test_remote_detector_drops_a_server_detection_still_carrying_the_caption_marker(fake_server):
    """Issue #172 defence-in-depth (#134's lesson: fix both paths). The server-side fix
    (detector.py) should mean this never happens, but if an unpatched/old server ever
    replies with a label that still contains the caption join marker, the client must
    drop it rather than trust and forward it -- other, legitimate detections in the
    same response still come through."""

    def response(body):
        return 200, {
            "per_tile": [
                [
                    {
                        "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
                        "label": "magazine . ottoman . potted plant . dressing table .",
                        "score": 0.19,
                    },
                    {"bbox_xyxy": [5.0, 6.0, 7.0, 8.0], "label": "ottoman", "score": 0.4},
                ],
            ]
        }

    fake_server.response = response
    det = _detector(fake_server, vocab_pass_cadence=1)
    det.question_prompt = "magazine . ottoman . potted plant . dressing table ."

    out = det(_tiles(1))

    assert [d.label for d in out[0]] == ["ottoman"]
