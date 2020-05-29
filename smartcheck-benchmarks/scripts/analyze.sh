#!/usr/bin/env sh

set -e -x -u

docker run -v$(pwd):/shrink-evaluations -it shrink-evaluations-hypothesis \
    sh -c "cd /shrink-evaluations && python3 -u evaluation.py analyze $@" | sed 's/\r?\n/\n/' | tee data/analysis.jsons
