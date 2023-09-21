#!/bin/bash

set -e

g++ -fsanitize=address -std=c++20 -fcoroutines -o gt_deadlock_detect_test -O2 -g3 -Wall -Wextra graphtoy.cpp gt_deadlock_detect_test.cpp

./gt_deadlock_detect_test
