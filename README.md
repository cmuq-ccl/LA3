# LA3: Distributed Big-Graph Analytics
LA3: A Scalable Link- and Locality-Aware Linear Algebra-Based Graph Analytics System

Dependencies:
- GNU g++ >= 5 (for C++14 support)
- MPICH v3
- Boost Serialization

Compile all:
- `make`

Compile an app:
- `make <app_name>`

Run an app:
- `make run ...`
  - `app=<app_name>`
  - `[np=<num_procs> (default=1)]`
  - `[tpp=<threads_per_proc> (default=2)]`
  - `[mf=<machine_file> (default=./machinefile)]`
  - `args="<input_arguments>"`

MPI + OpenMP:
  - In general, we launch:
    - total_num_vcpus/2 MPI ranks
    - two OpenMP threads per rank (default)
  - For example, for a 10-node cluster with 8 vcpus per node:
    - `np=40 tpp=2`
  - Some apps (e.g., TF-IDF) may perform better with one OpenMP thread per rank:
    - `np=40 tpp=1`

Examples:
- Run Pagerank on test graph G1 (8 vertices) (until convergence):
  - `make run app=pr np=4 args="data/g1_8_8_13.bin 8"`
- Run Pagerank on test graph G1 (8 vertices) (20 iterations):
  - `make run app=pr np=4 args="data/g1_8_8_13.bin 8 20"`
- Run SSSP on test graph G1 (8 vertices) (root vertex id = 5):
  - `make run app=sssp np=4 args="data/g1_8_8_13.w.bin 8 5"`
- Run BFS on test graph G1 (8 vertices) (root vertex id = 5):
  - `make run app=bfs np=4 args="data/g1_8_8_13.bin 8 5"`
- Run TC on test graph G1 (8 vertices):
  - `make run app=tc np=4 args="data/g1_8_8_13.bin 8"`
- Run TF-IDF on test graph G2 (bipartite: 4 docs, 7 terms) without blind feedback:
  (k=3, 10 random queries, 2 terms per query):
  - `make run app=tfidf np=4 args="data/g2_4_7_10.w.bp.bin 4 7 3 0 10 2"`
- Run TF-IDF on test graph G2 (bipartite: 4 docs, 7 terms) with blind feedback:
  (k=3, r=2, 10 random queries, 2 terms per query):
  - `make run app=tfidf np=4 args="data/g2_4_7_10.w.bp.bin 4 7 3 2 10 2"`

Converting a dataset from Matrix Market format to LA3 binary format:
- `bin/mtx2bin <filepath_in> <filepath_out> [-hi[o]] [-wi] [-wo{i|d}]`
  - `[-hi[o]] .... read in first non-comment line as header [and write [o]ut]`
  - `[-wi] ....... read in edge weights (must be int or double)`
  - `[-wo{i|d}] .. write out edge weights ({i}nt/{d}ouble) (rand [1,128] if none given)`
- Examples:
  - LiveJournal (text pairs to binary pairs):
    - `bin/mtx2bin soc-LiveJournal1.txt livejournal.bin`
  - LiveJournal (text pairs to binary triples): 
    - `bin/mtx2bin soc-LiveJournal1.txt livejournal.w.bin -woi`
    
