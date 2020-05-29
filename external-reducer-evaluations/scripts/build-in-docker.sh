#!/usr/bin/env bash

set -e -u -x

ROOT=$(realpath $(dirname "$0")/..)
export HOME=/tmp/home

export PYTHONPATH="$ROOT"/hypothesis-csmith

if ! python -m reducereval check 2>/dev/null ; then
    pip install -e "$ROOT"
fi

python -m reducereval build
