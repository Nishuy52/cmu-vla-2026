# docker/ai_module — DEPRECATED (superseded by docker/ai_module_fork/)

**This directory is retired (H7 / red-team finding F4). Use
[`docker/ai_module_fork/`](../ai_module_fork/README.md) instead.**

## Why

The `Dockerfile` here was written for a **repo-root build context** (`COPY src/` resolving against
the repo root). The challenge submission is a **fork** in which only the `ai_module/` directory may
be modified, and the upstream compose builds the `ai_module` service from **context `../ai_module`
using `ai_module/docker/Dockerfile`** (`docs/upstream_notes.md` §5B, gotcha 12). A repo-root-context
Dockerfile cannot slot into that shape — the submitted fork would fail `docker compose up --build`.

The fork-shaped packaging (Dockerfile written for an `ai_module/` context, plus `sync_to_fork.ps1`
/ `sync_to_fork.sh` that stage our `src/` payload into a fork checkout) now lives in
[`docker/ai_module_fork/`](../ai_module_fork/README.md).

## Status of the files here

- `Dockerfile` — **do not build.** Kept as a historical record of the old repo-root approach;
  marked DEPRECATED in-file.
- `README.md` — this stub.

Definition of packaging done is now: **`docker compose up --build` from a clean clone of the fork**
succeeds (with `src/` synced into `ai_module/src/` via the sync script). See
[`docker/ai_module_fork/README.md`](../ai_module_fork/README.md) for the exact commands.
