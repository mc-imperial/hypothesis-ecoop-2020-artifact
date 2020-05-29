#!/usr/bin/env bash

set -e -u -x

ROOT=$(realpath $(dirname "$0")/..)

make docker
docker run --network=host -u$UID -v"$ROOT:/reducer-eval" -w/reducer-eval -t  reducer-eval /reducer-eval/scripts/build-in-docker.sh
