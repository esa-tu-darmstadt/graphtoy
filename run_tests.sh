#!/bin/bash

set -e

g++ -fsanitize=address -std=c++20 -fcoroutines -o gt_test_app -O2 -g3 -Wall -Wextra graphtoy.cpp graphtoy_test_app.cpp gt_buffer_tests.cpp

./gt_test_app
