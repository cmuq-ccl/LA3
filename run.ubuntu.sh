#!/bin/shbash
# Ubuntu run script
# export PATH=/usr/local/openmpi/bin:$PATH

mpirun -np 2 bin/graph_analytics/pr data/graph_analytics/g1_8_8_13.bin 8
