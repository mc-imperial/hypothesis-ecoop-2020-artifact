#!/usr/bin/env sh

set -e -u -x

# Install LLVM + clang binary packages for C-Reduce
cd /tmp
wget http://releases.llvm.org/8.0.0/clang+llvm-8.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
aunpack clang+llvm-8.0.0*.tar.xz
cp -R clang+llvm-8.0.0*/* /usr/local
