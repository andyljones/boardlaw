#!/bin/bash

mkdir build
cd build
cmake ../
make -j4

# Less-than-ideal to copy it into a subdir of /bin,
# but the DATADIR is hardcoded to the same dir. So
# if we want any portability at all, subdir-and-
# softlink it is
mkdir $PREFIX/bin/mohex
cp src/mohex/mohex $PREFIX/bin/mohex/
cp -r ../share $PREFIX/bin/mohex/