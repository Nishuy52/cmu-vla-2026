"""API-tier prompt templates for checkpoint-1 parsing (question text -> Plan JSON).

Provider-agnostic: exposes plain message dicts and a ChatFn injection point.
No API keys, no network calls, no provider SDK imports live in this module.

In-context examples: one numerical + one object_reference + two instruction_following
(the corridor and avoid constructs both live in instruction_following, and both are
penalty-scored, so each gets its own worked example).
"""
from __future__ import annotations

import json
from typing import Callable

#: A chat completion function: list of {'role': ..., 'content': ...} messages -> reply text.
#: Adapters for concrete providers (API 1, API 2, local VLM) are injected by the caller.
ChatFn = Callable[[list[dict[str, str]]], str]

# --------------------------------------------------------------------- JSON schema
# Hand-derived from core/plan_schema.py (Plan/TargetSpec/Clause/Anchor/RouteLeg/AvoidSpec).

PLAN_JSON_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Plan",
    "type": "object",
    "required": ["qtype", "question_raw"],
    "properties": {
        "qtype": {"enum": ["numerical", "object_reference", "instruction_following"]},
        "question_raw": {"type": "string"},
        "target": {
            "oneOf": [{"$ref": "#/definitions/target"}, {"type": "null"}],
            "description": "Required for numerical & object_reference; null for instruction_following.",
        },
        "route": {
            "type": "array",
            "items": {"$ref": "#/definitions/leg"},
            "description": "Ordered legs; required non-empty for instruction_following, "
            "empty otherwise. The final leg MUST be kind 'goto'.",
        },
        "avoid": {"type": "array", "items": {"$ref": "#/definitions/avoid"}},
        "notes": {
            "type": "string",
            "description": "Escape hatch: anything you could not encode in the schema.",
        },
        "parse_tier": {"type": "string"},
    },
    "definitions": {
        "anchor": {
            "type": "object",
            "required": ["noun"],
            "properties": {
                "noun": {
                    "type": "string",
                    "description": "Canonical lowercase-singular noun ('refrigerator', 'potted plant').",
                },
                "raw": {"type": "string", "description": "Surface form as written, typos kept."},
                "attributes": {"type": "array", "items": {"type": "string"}},
                "disambiguator": {
                    "oneOf": [{"$ref": "#/definitions/clause"}, {"type": "null"}],
                    "description": "Nested constraint picking WHICH instance ('the table closest to the door').",
                },
            },
        },
        "clause": {
            "type": "object",
            "required": ["pred", "anchors"],
            "properties": {
                "pred": {
                    "enum": [
                        "on", "in", "near", "next_to", "between", "above",
                        "under", "closest_to", "farthest_from", "with",
                    ]
                },
                "anchors": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/anchor"},
                    "description": "Exactly 2 anchors for 'between', exactly 1 otherwise.",
                },
                "negated": {"type": "boolean"},
            },
        },
        "target": {
            "type": "object",
            "required": ["noun"],
            "properties": {
                "noun": {"type": "string"},
                "raw": {"type": "string"},
                "attributes": {"type": "array", "items": {"type": "string"}},
                "clauses": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/clause"},
                    "description": "ALL clauses must hold (AND).",
                },
            },
        },
        "leg": {
            "type": "object",
            "required": ["kind", "anchors"],
            "properties": {
                "kind": {"enum": ["goto", "via_near", "corridor_between"]},
                "anchors": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/anchor"},
                    "description": "Exactly 2 anchors for corridor_between, exactly 1 otherwise.",
                },
            },
        },
        "avoid": {
            "type": "object",
            "description": "Set exactly ONE of 'between' (forbidden corridor) or 'near' (forbidden disc).",
            "properties": {
                "between": {
                    "oneOf": [
                        {"type": "array", "items": {"$ref": "#/definitions/anchor"}},
                        {"type": "null"},
                    ]
                },
                "near": {"oneOf": [{"$ref": "#/definitions/anchor"}, {"type": "null"}]},
            },
        },
    },
}

# --------------------------------------------------------------------- in-context examples


def _a(noun: str, raw: str = "", attributes: list | None = None, disambiguator: dict | None = None) -> dict:
    """Build an example anchor dict."""
    return {
        "noun": noun,
        "raw": raw or noun,
        "attributes": attributes or [],
        "disambiguator": disambiguator,
    }


def _c(pred: str, anchors: list[dict], negated: bool = False) -> dict:
    """Build an example clause dict."""
    return {"pred": pred, "anchors": anchors, "negated": negated}


EXAMPLES: list[tuple[str, dict]] = [
    (
        "How many chairs are near the table with a vase on it?",
        {
            "qtype": "numerical",
            "question_raw": "How many chairs are near the table with a vase on it?",
            "target": {
                "noun": "chair",
                "raw": "chairs",
                "attributes": [],
                "clauses": [
                    _c("near", [_a("table", disambiguator=_c("with", [_a("vase")]))])
                ],
            },
            "route": [],
            "avoid": [],
            "notes": "indefinite article on 'vase'",
            "parse_tier": "api",
        },
    ),
    (
        "Find the wall lamp that is between a door frame and a window.",
        {
            "qtype": "object_reference",
            "question_raw": "Find the wall lamp that is between a door frame and a window.",
            "target": {
                "noun": "wall lamp",
                "raw": "wall lamp",
                "attributes": [],
                "clauses": [_c("between", [_a("door frame"), _a("window")])],
            },
            "route": [],
            "avoid": [],
            "notes": "indefinite articles: any door frame / window pair",
            "parse_tier": "api",
        },
    ),
    (
        "First, go to the potted plant furthest from the hookah, then take the path "
        "between the two columns, and stop at the tray on the table.",
        {
            "qtype": "instruction_following",
            "question_raw": "First, go to the potted plant furthest from the hookah, then "
            "take the path between the two columns, and stop at the tray on the table.",
            "target": None,
            "route": [
                {
                    "kind": "goto",
                    "anchors": [
                        _a("potted plant", disambiguator=_c("farthest_from", [_a("hookah")]))
                    ],
                },
                {
                    "kind": "corridor_between",
                    "anchors": [_a("column", raw="columns"), _a("column", raw="columns")],
                },
                {
                    "kind": "goto",
                    "anchors": [_a("tray", disambiguator=_c("on", [_a("table")]))],
                },
            ],
            "avoid": [],
            "notes": "'the two columns' expands to a corridor between the column pair",
            "parse_tier": "api",
        },
    ),
    (
        "First, go to the chair near the window, then stop at the soccer ball near the "
        "couch, avoiding the path between the TV and the tea table.",
        {
            "qtype": "instruction_following",
            "question_raw": "First, go to the chair near the window, then stop at the soccer "
            "ball near the couch, avoiding the path between the TV and the tea table.",
            "target": None,
            "route": [
                {"kind": "goto", "anchors": [_a("chair", disambiguator=_c("near", [_a("window")]))]},
                {
                    "kind": "goto",
                    "anchors": [_a("soccer ball", disambiguator=_c("near", [_a("couch")]))],
                },
            ],
            "avoid": [
                {"between": [_a("tv", raw="tv"), _a("tea table")], "near": None}
            ],
            "notes": "avoid corridor applies to the WHOLE traversal, not one leg",
            "parse_tier": "api",
        },
    ),
]

# --------------------------------------------------------------------- prompt templates

_RULES = """\
You are the question parser of an indoor vision-language navigation robot. Convert ONE
natural-language question into a Plan JSON object conforming EXACTLY to the JSON schema
below. Output ONLY the JSON object — no markdown fences, no commentary, no extra keys.

Rules:
1. qtype: counting questions ("How many...", "Count...") -> "numerical"; locate questions
   ("Find the...", or a bare noun phrase) -> "object_reference"; movement commands
   ("Go to...", "take the path...", "stop at...") -> "instruction_following".
2. numerical/object_reference: fill "target", leave "route" empty.
   instruction_following: fill "route" (ordered!), leave "target" null.
3. Nouns are canonical lowercase singular ("refrigerator", "potted plant"); keep the
   surface form (typos, plurals, e.g. "refridgerator") in "raw". Colors/sizes/materials
   go in "attributes", not in "noun".
4. A relation that picks WHICH instance of an anchor ("the table closest to the door")
   goes in that anchor's "disambiguator", nesting as deep as needed.
5. "with X on it" / "that has X on it" -> pred "with". "with X above it" -> pred "under"
   (the subject sits under X).
6. instruction_following legs, in the order the text gives them:
   - "go to / go near / stop at / stop by / then to" -> kind "goto"
   - "take the path near X" / "pass by X" -> kind "via_near" (1 anchor)
   - "take the path between X and Y" / "go between X and Y" -> kind "corridor_between"
     (exactly 2 anchors; "the two columns" means two instances of "column")
   The route MUST end with a "goto" leg (the terminal stop).
   NOTE: "the vase between the TV and the door" after "stop at" is a goto leg whose
   anchor has a "between" disambiguator — NOT a corridor leg. Corridors are only
   "take the path between" / "go between" phrasings.
7. "avoiding the path between X and Y" / "avoid the path near X" -> an entry in "avoid"
   (forbidden for the whole traversal). Never encode avoids as route legs.
8. All relations are object-to-object (allocentric). Superlatives ("closest to",
   "farthest from"/"furthest from") map to closest_to / farthest_from.
9. If something cannot be encoded, choose the nearest predicate and explain in "notes".
   Never invent new predicates, kinds, or keys.
10. Set "question_raw" to the question verbatim and "parse_tier" to "api".

JSON schema:
{schema}

Examples:
{examples}"""

USER_PROMPT_TEMPLATE = """\
Question: {question}

Respond with ONLY the Plan JSON object."""

REPAIR_PROMPT_TEMPLATE = """\
Your previous output for this question failed validation.

Question: {question}

Your previous output:
{previous_output}

Validation errors:
{errors}

Return the corrected Plan JSON object. Output ONLY the JSON object — no markdown fences,
no commentary. Fix every listed error while preserving the parts that were correct."""


def _render_examples() -> str:
    """Render the in-context examples as Question/JSON blocks."""
    blocks = []
    for q, plan in EXAMPLES:
        blocks.append(f"Question: {q}\n{json.dumps(plan, separators=(',', ': '))}")
    return "\n\n".join(blocks)


def _strip_schema(obj):
    """Recursively drop cost-only keys ($schema/title/description) from the JSON schema.

    OR-F10: the rendered schema dominates the CP1 token spend. The parse contract is fully
    conveyed by the structure (required keys + enums) and the in-context examples, so the
    human-readable ``description`` / ``$schema`` / ``title`` fields are dropped from the
    wire copy. Combined with a no-indent, compact-separator dump this brings the system
    prompt under the ~2k-token budget without changing what the model is asked to produce.
    """
    if isinstance(obj, dict):
        return {
            k: _strip_schema(v)
            for k, v in obj.items()
            if k not in ("$schema", "title", "description")
        }
    if isinstance(obj, list):
        return [_strip_schema(v) for v in obj]
    return obj


#: Compact wire copy of the schema (descriptions stripped, no indent) — see _strip_schema.
COMPACT_PLAN_JSON_SCHEMA: dict = _strip_schema(PLAN_JSON_SCHEMA)

SYSTEM_PROMPT: str = _RULES.format(
    schema=json.dumps(COMPACT_PLAN_JSON_SCHEMA, indent=None, separators=(",", ":")),
    examples=_render_examples(),
)


def build_parse_messages(question: str) -> list[dict[str, str]]:
    """Build the chat messages for a first parse attempt of one question."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=question)},
    ]


def build_repair_messages(
    question: str, previous_output: str, errors: list[str]
) -> list[dict[str, str]]:
    """Build the chat messages for the one repair round after a failed validation."""
    err_text = "\n".join(f"- {e}" for e in errors) or "- output was not parseable JSON"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": REPAIR_PROMPT_TEMPLATE.format(
                question=question,
                previous_output=previous_output or "(empty)",
                errors=err_text,
            ),
        },
    ]
