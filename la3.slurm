#!/bin/bash
#SBATCH --job-name=la3
#SBATCH --output=la3.out
#SBATCH --error=la3.err
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --nodes=4
# #SBATCH --sockets-per-node=2
# #SBATCH --cores-per-socket=8
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --cluster=mpi
#SBATCH --partition=opa



# Set this to whatever MAX_NUM_THREADS
# -> is set to
# Three day max wall time
# -> choose what you need




# any module loads would go here

# I noticed that you left running processes
# -> on the nodes, make sure they die
#kill_distrograph_and_revolver () {
#   pkill --uid=$SLURM_JOB_USER distrograph
#   pkill --uid=$SLURM_JOB_USER revolver.sh
#}
#trap kill_distrograph_and_revolver EXIT

echo "SLURM_JOB_ID="$SLURM_JOB_ID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_CORES_NODES"=$SLURM_CPUS_PER_TASK
echo "SLURM_TASKS"=$SLURM_NTASKS
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR


echo "************************************************"



module purge
module load python/anaconda3.5-4.2.0
module load boost/1.62.0
module load gcc/6.3.0
module load openmpi
#module load intel

export SLURM_CPU_BIND="none"
#export I_MPI_FABRICS_LIST="ofi,tmi,dapl,tcp,ofa"
#export I_MPI_FALLBACK=1
export I_MPI_FABRICS=shm:ofa


export I_MPI_DEBUG=2
export LD_LIBRARY_PATH=/ihome/rmelhem/moh18/boost/boost_1_67_0/stage/lib:/ihome/rmelhem/moh18/boost:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
NP=$SLURM_NTASKS
#./hello.omp

time mpirun -np ${NP} bin/pr /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat26.bin 67108864
#time mpirun -np 32 bin/pr /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat27.bin 134217728


#time mpirun -np ${NP} bin/pr /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat26.bin 67108864 20
#time mpirun -np ${NP} bin/sssp /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat26.bin 67108864 5
#time mpirun -np ${NP} bin/bfs /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat26.bin 67108864 5
#time mpirun -np ${NP} bin/cc /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat26.bin 67108864
#time mpirun -np ${NP} bin/tc /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat26.bin 67108864

#time mpirun -np ${NP} bin/pr /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat27.bin 134217728 20
#time mpirun -np ${NP} bin/sssp /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat27.bin 134217728 5
#time mpirun -np ${NP} bin/bfs /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat27.bin 134217728 5
#time mpirun -np ${NP} bin/cc /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat27.bin 134217728
#time mpirun -np ${NP} bin/tc /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat27.bin 134217728

#time mpirun -np ${NP} bin/pr /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat28.bin 268435456 20
#time mpirun -np ${NP} bin/sssp /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat28.bin 268435456 5
#time mpirun -np ${NP} bin/bfs /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat28.bin 268435456 5
#time mpirun -np ${NP} bin/cc /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat28.bin 268435456
#time mpirun -np ${NP} bin/tc /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat28.bin 268435456



#time mpirun -np 32 bin/pr /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat27.bin 134217728
#squeue -u moh18 &> la3.out
#ifconfig &> la3.out
#make run app=pr np=4 args="data/g1_8_8_13.bin 8" &> la3.out
#HYDRA_TOPO_DEBUG=1 mpirun -np 2 -ppn 1 bin/pr data/g1_8_8_13.bin 8 &> la3.out
#(time srun --nodes=4 --ntasks-per-node=8 --ntasks=32 bin/pr data/livejournal.bin 4847571) >> la3.out 2>&1
#(time mpirun -np 8 bin/pr data/livejournal.bin 4847571) >> la3.out 2>&1
#(time mpirun -np 32  bin/pr /zfs1/cs3580_2017F/moh18/graph500/rmat/rmat26.bin 67108864) >> la3.out 2>&1

