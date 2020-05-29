#!/usr/bin/env sh

set -e -u -x

apt-get install -y python3-dev python3-pip

# C-Reduce deps
apt-get install -y \
  libexporter-lite-perl libfile-which-perl libgetopt-tabular-perl \
  libregexp-common-perl flex build-essential zlib1g-dev \
  libncurses5-dev libxml2-dev

# Misc utilities
apt-get install -y libtool m4 time atool wget

python3 -m pip install --upgrade setuptools pip virtualenv
