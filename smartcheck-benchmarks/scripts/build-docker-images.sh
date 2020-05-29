#!/usr/bin/env sh

set -e -x -u

for f in common quickcheck smartcheck hypothesis ; do
  docker build -t shrink-evaluations-$f -f ./Dockerfile.$f .
done
