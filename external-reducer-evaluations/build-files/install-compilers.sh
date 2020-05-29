#!/usr/bin/env bash

# Trimmed down and specialised version of matt godbolt's compiler explorer
# install script at https://github.com/mattgodbolt/compiler-explorer-image/blob/master/update_compilers/install_cpp_compilers.sh

set -e -u -x

S3BUCKET=compiler-explorer
SUBDIR=opt
S3URL=https://s3.amazonaws.com/${S3BUCKET}/${SUBDIR}

mkdir -p /opt/compiler-explorer

cd /opt/compiler-explorer

for version in \
    4.1.2 \
    4.4.7 \
    4.5.3 \
    4.6.4 \
    4.7.{1,2,3,4} \
    4.8.{1,2,3,4,5} \
    4.9.{0,1,2,3,4} \
    5.{1,2,3,4,5}.0 \
    6.{1,2,3,4}.0 \
    7.{1,2,3,4}.0 \
    8.{1,2,3}.0 \
; do
    if [[ ! -d gcc-${version} ]]; then
        compiler=gcc-${version}.tar.xz
        curl ${S3URL}/$compiler | tar Jxf -
    fi
done

for version in \
    3.9.1 \
    4.0.0 \
    4.0.1 \
    5.0.0 \
    6.0.0 \
    7.0.0 \
; do
    if [[ ! -d clang-${version} ]]; then
        compiler=clang-${version}.tar.xz
        curl ${S3URL}/$compiler | tar Jxf -
    fi
done
