#!/bin/bash

set -e

if ! command -v python &> /dev/null; then
  echo "error: python not found"
  exit 127
fi

SRC_DIR=deepneumo
PYPATH="${PWD}"/venv/bin

"${PYPATH}"/python "${SRC_DIR}"/cli.py do-split-in-three