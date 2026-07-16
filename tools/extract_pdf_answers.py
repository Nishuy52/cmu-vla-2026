"""Extract ground-truth numerical answers from the per-scene questions.pdf files.

The upstream challenge repo ships a rendered ``questions.pdf`` per training
scene whose text layer contains the numerical answer verbatim
("Response: Print N in terminal"). Object-reference and instruction-following
answers are embedded images (bounding-box / trajectory renders) and are NOT
extracted here — they need visual transcription.

Usage (from the repo root)::

    python -m tools.extract_pdf_answers [--questions-root upstream/CMU-VLN-Challenge-2026/questions] [--out docs/gt_answers_numerical.json]

Output JSON: per-scene ``{question, answer}`` plus a provenance block
(source paths, extraction pattern, date). Offline dev tool — not part of the
scored pipeline.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from pathlib import Path

from pypdf import PdfReader

_ANSWER_RE = re.compile(r"Response:\s*Print\s*(\d+)\s*in\s*terminal", re.I)
_QUESTION_RE = re.compile(r"Question\s*1:\s*(.*?)\s*Response:", re.I)
_WS_RE = re.compile(r"\s+")


def extract(questions_root: Path) -> dict:
    scenes: dict[str, dict] = {}
    for pdf in sorted(questions_root.glob("*/questions.pdf")):
        scene = pdf.parent.name
        text = _WS_RE.sub(" ", " ".join((p.extract_text() or "") for p in PdfReader(str(pdf)).pages))
        m_ans = _ANSWER_RE.search(text)
        m_q = _QUESTION_RE.search(text)
        scenes[scene] = {
            # the text layer squeezes inter-word spaces; keep the raw form so
            # the value is verifiable against the PDF, not prettified
            "question_raw": m_q.group(1) if m_q else None,
            "answer": int(m_ans.group(1)) if m_ans else None,
        }
    return scenes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions-root", type=Path, default=Path("upstream/CMU-VLN-Challenge-2026/questions"))
    ap.add_argument("--out", type=Path, default=Path("docs/gt_answers_numerical.json"))
    args = ap.parse_args()

    scenes = extract(args.questions_root)
    missing = [s for s, v in scenes.items() if v["answer"] is None]
    payload = {
        "provenance": {
            "source": "per-scene questions.pdf text layer, upstream challenge repo",
            "pattern": _ANSWER_RE.pattern,
            "extracted": _dt.date.today().isoformat(),
            "regenerate": "python -m tools.extract_pdf_answers (from repo root)",
            "note": "OR/IF answers are embedded images in the same PDFs; not extracted here",
        },
        "scenes": scenes,
    }
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(scenes)} scenes; missing answers: {missing or 'none'})")


if __name__ == "__main__":
    main()
