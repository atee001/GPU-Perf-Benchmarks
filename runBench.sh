#!/bin/bash

make clean
make
./build/perf 1 20
# ./perf 2 20
