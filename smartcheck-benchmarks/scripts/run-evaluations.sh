#!/usr/bin/env bash

set -e -x -u

docker run -u$UID -v$(pwd):/shrink-evaluations -w/shrink-evaluations -it shrink-evaluations-hypothesis \
    sh -c 'python3 evaluation.py build --target=Hypothesis'

docker run -u$UID -v$(pwd):/shrink-evaluations -w/shrink-evaluations -it shrink-evaluations-quickcheck \
    sh -c 'python3 evaluation.py build --target=QuickCheckCustom'

docker run -u$UID -v$(pwd):/shrink-evaluations -w/shrink-evaluations -it shrink-evaluations-quickcheck \
    sh -c 'python3 evaluation.py build --target=QuickCheck'

docker run -u$UID -v$(pwd):/shrink-evaluations -w/shrink-evaluations -it shrink-evaluations-smartcheck \
    sh -c 'python3 evaluation.py build --target=SmartCheck'
