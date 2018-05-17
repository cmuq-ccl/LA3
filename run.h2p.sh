#!/bin/bash
# h2p load modules script

module purge
module load openmpi
#module load intel
module load python/anaconda3.5-4.2.0
module load boost/1.62.0
module load gcc/6.3.0

mpirun -np 2 bin/graph_analytics/pr data/graph_analytics/g1_8_8_13.bin 8
