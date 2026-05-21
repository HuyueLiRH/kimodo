#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <prior-run-folder> [port]" >&2
  exit 2
fi

RUN_FOLDER="$1"
PORT="${2:-7861}"
HOST="${KIMODO_PRIOR_VIEWER_HOST:-127.0.0.1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="${KIMODO_PRIOR_VIEWER_WORKSPACE:-$PWD}"
VENV_DIR="${WORKSPACE_ROOT}/.venvs/kimodo-prior-viewer"
READY_MARKER="${VENV_DIR}/.kimodo-prior-viewer-ready"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      if ! "${candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
        continue
      fi
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "Could not find Python 3.10+ for the local Kimodo prior viewer venv." >&2
  echo "Set PYTHON_BIN=/path/to/python3.10-or-newer and rerun this script." >&2
  exit 2
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  mkdir -p "$(dirname "${VENV_DIR}")"
  # Keep the install isolated by using python -m venv under the current workspace.
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

if ! "${VENV_DIR}/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Existing viewer venv was created with Python < 3.10. Move ${VENV_DIR} aside and rerun." >&2
  exit 2
fi

if [[ ! -f "${READY_MARKER}" ]]; then
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
  SKIP_MOTION_CORRECTION_IN_SETUP=1 "${VENV_DIR}/bin/python" -m pip install -e "${REPO_ROOT}[demo]"
  touch "${READY_MARKER}"
fi

echo "Kimodo prior viewer: http://${HOST}:${PORT}/"
exec "${VENV_DIR}/bin/kimodo_prior_viewer" --run-folder "${RUN_FOLDER}" --host "${HOST}" --port "${PORT}"
