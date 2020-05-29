#!/usr/bin/env bash

set -e -u -x

ROOT=$(realpath $(dirname "$0")/..)

make docker
docker run --network=host -u$UID -v"$ROOT:/reducer-eval" -w/reducer-eval -it  reducer-eval bash
