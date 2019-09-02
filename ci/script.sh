#!/bin/bash

set -euxo pipefail

mkdir build
cd build
cmake ..
make

# First run unit tests normally, to see if any test fails.
make test

# If the tests succeed, run them once more to see if there
# are any memory errors or leaks.
ctest \
  --overwrite MemoryCheckCommandOptions="--leak-check=full --error-exitcode=1" \
  -T memcheck