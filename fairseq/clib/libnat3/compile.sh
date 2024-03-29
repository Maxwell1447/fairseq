#!/bin/bash

FLAGS="-Wall"

g++ $FLAGS edit_dist.cpp -o edit_dist -lpthread

./edit_dist
