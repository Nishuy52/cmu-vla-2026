# sync_to_fork.ps1 — stage our repo's payload into the challenge fork's ai_module/ tree (H7).
#
# Windows-side twin of sync_to_fork.sh. The submission fork may modify ONLY ai_module/
# (upstream_notes.md gotcha 12). Our code lives in THIS repo under src/,
# docker/ai_module_fork/docker/Dockerfile, and docker/ai_module_fork/ai_module/launch_with_llm.sh.
# This script copies all three into the fork's ai_module/ directory in the exact shape the fork
# build expects:
#
#     <fork>\ai_module\docker\Dockerfile      <- docker\ai_module_fork\docker\Dockerfile (this repo)
#     <fork>\ai_module\launch_with_llm.sh     <- docker\ai_module_fork\ai_module\launch_with_llm.sh (this repo)
#     <fork>\ai_module\src\core\              <- src\core\        (this repo)
#     <fork>\ai_module\src\ros_adapter\       <- src\ros_adapter\ (this repo)
#
# NOTE (Phase 3): docker/ai_module_fork/docker/compose.yml and compose_gpu.yml are reference
# copies only, NOT synced here (they live outside ai_module/ in the fork tree) — apply manually
# per docs/ubuntu_setup.md §7a step 2.
#
# src\tests\ is DELIBERATELY EXCLUDED: the runtime image only needs `core` + `ros_adapter`, so
# tests only bloat the size-budgeted eval image (Simply NUC i9, upstream_notes.md §5). Run the
# test suite from THIS repo on Ubuntu (ubuntu_setup.md §3), not from inside the fork/image.
#
# NOTE: the actual `docker compose up --build` validation is Ubuntu-gated. On Windows this only
# stages files; it does not (and must not) attempt a build.
#
# Usage:
#     .\docker\ai_module_fork\sync_to_fork.ps1 -ForkRoot <path-to-fork-checkout>
# where <path-to-fork-checkout> is the root of the cloned challenge fork (the dir CONTAINING
# ai_module\).
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ForkRoot
)
$ErrorActionPreference = 'Stop'

# Resolve this repo's root from the script location (…\docker\ai_module_fork\sync_to_fork.ps1).
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir '..\..')).Path
$AiModule  = Join-Path $ForkRoot 'ai_module'

if (-not (Test-Path -LiteralPath $AiModule -PathType Container)) {
    throw "error: $AiModule does not exist — is '$ForkRoot' the fork root (must contain ai_module\)?"
}

Write-Host "repo   : $RepoRoot"
Write-Host "fork   : $ForkRoot"
Write-Host "target : $AiModule"

# 1) Dockerfile -> ai_module\docker\Dockerfile
$dockerDst = Join-Path $AiModule 'docker'
New-Item -ItemType Directory -Force -Path $dockerDst | Out-Null
Copy-Item -LiteralPath (Join-Path $ScriptDir 'docker\Dockerfile') -Destination (Join-Path $dockerDst 'Dockerfile') -Force
Write-Host "synced : ai_module\docker\Dockerfile"

# 2) launch_with_llm.sh -> ai_module\launch_with_llm.sh (Phase 3 in-image LLM launch wrapper)
Copy-Item -LiteralPath (Join-Path $ScriptDir 'ai_module\launch_with_llm.sh') -Destination (Join-Path $AiModule 'launch_with_llm.sh') -Force
Write-Host "synced : ai_module\launch_with_llm.sh"

# 3) src\ (core + ros_adapter, NO tests) -> ai_module\src\
#    Rebuild the fork's src\ from scratch so it is an exact mirror of ours minus tests/caches.
$srcDst = Join-Path $AiModule 'src'
if (Test-Path -LiteralPath $srcDst) { Remove-Item -LiteralPath $srcDst -Recurse -Force }
New-Item -ItemType Directory -Force -Path $srcDst | Out-Null

foreach ($d in @('core', 'ros_adapter')) {
    $srcPath = Join-Path $RepoRoot "src\$d"
    Copy-Item -LiteralPath $srcPath -Destination $srcDst -Recurse -Force
}
# Prune caches that Copy-Item brought along.
Get-ChildItem -LiteralPath $srcDst -Recurse -Directory -Force |
    Where-Object { $_.Name -in @('__pycache__', '.pytest_cache') } |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }
Get-ChildItem -LiteralPath $srcDst -Recurse -File -Force -Filter '*.pyc' |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue }
Write-Host "synced : ai_module\src\  (core + ros_adapter; tests excluded)"

Write-Host "done. Next (on Ubuntu): cd '$ForkRoot/docker' && docker compose up --build"
