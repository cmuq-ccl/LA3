# Usage:
#  make
#  make run app=<app_name> [np=<num_procs> (default=1)]
#                          [tpp=<threads_per_proc> (default=2)]
#                          [mf=<machine_file> (default=./machinefile)]
#                          args="<input_arguments>"

#app ?=
np ?= 1
tpp ?= 2
mf ?= ./machinefile
#args ?=

# C++ and MPI Compilers (minimum -std=c++14)
CXX = g++
MPI_CXX = mpicxx

# Compiler Flags:
# For debugging, enable DEBUG and disable OPTIMIZE
DNWARN = -Wno-comment -Wno-literal-suffix -Wno-sign-compare \
         -Wno-unused-function -Wno-unused-variable -Wno-reorder
THREADED = -fopenmp -D_GLIBCXX_PARALLEL
#DEBUG = -g -fsanitize=undefined,address -lasan -lubsan
OPTIMIZE = -DNDEBUG -O3 -flto -fwhole-program -march=native

SRC_UTILS = $(wildcard src/utils/*.cpp)

.PHONY: all tools apps

all: clean tools apps

clean:
	rm -rf bin

apps: degree pr sssp bfs cc tc tfidf tfidf_batch graphsim

tools:
	@mkdir -p bin
	$(CXX) -std=c++14 $(DNWARN) $(OPTIMIZE) $(DEBUG) src/tools/mtx2bin.cpp -o bin/mtx2bin
	$(MPI_CXX) -std=c++14 $(DNWARN) $(OPTIMIZE) $(DEBUG) src/tools/cw2bin.cpp \
			-lboost_serialization -lboost_mpi -o bin/cw2bin

run: #$(app)
	export OMP_NUM_THREADS=$(tpp); \
	mpirun -np $(np) -machinefile $(mf) bin/$(app) $(args)

.DEFAULT:
	@mkdir -p bin
	$(MPI_CXX) -std=c++14 $(DNWARN) $(THREADED) $(OPTIMIZE) $(DEBUG) \
		$(SRC_UTILS) src/apps/$@.cpp -lboost_serialization -I src -o bin/$@
