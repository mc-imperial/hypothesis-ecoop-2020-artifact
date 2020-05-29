#!/usr/bin/env sh

set -e -u -x

cpan -i Exporter::Lite File::Which Getopt::Tabular Regexp::Common

cd /tmp
git clone https://github.com/csmith-project/creduce.git
cd creduce
git checkout creduce-2.10.0
./configure
make install
