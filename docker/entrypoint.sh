#!/usr/bin/env bash
# BraTS-METS 2026 Task 1 container entrypoint. Licensed under Apache-2.0.
set -euo pipefail

echo "[entrypoint] $(date -u '+%Y-%m-%d %H:%M:%S UTC') starting"

: "${BRATS_INPUT_DIR:=/input}"
: "${BRATS_OUTPUT_DIR:=/output}"

if [[ ! -d "${BRATS_INPUT_DIR}" ]]; then
  echo "[entrypoint] FATAL: input directory ${BRATS_INPUT_DIR} not mounted" >&2
  exit 1
fi
mkdir -p "${BRATS_OUTPUT_DIR}"

# Recreate scratch dirs in case /tmp is a fresh tmpfs at runtime.
mkdir -p "${nnUNet_raw:-/tmp/nnUNet_raw}" \
         "${nnUNet_preprocessed:-/tmp/nnUNet_preprocessed}" \
         "${BRATS_WORK_DIR:-/tmp/brats_work}"

# Guard: the challenge forbids writing into /input.
if [[ -w "${BRATS_INPUT_DIR}" ]]; then
  echo "[entrypoint] note: ${BRATS_INPUT_DIR} is writable; this pipeline never writes there."
fi

exec python /opt/app/src/predict.py "$@"
