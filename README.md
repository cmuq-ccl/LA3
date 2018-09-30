# LA3: Distributed Big-Graph Analytics
LA3: A Scalable Link- and Locality-Aware Linear Algebra-Based Graph Analytics System

Dependencies:
- GNU g++ >= 5 (for C++14 support)
- MPICH v3
- Boost Serialization

Compile all:
- `make`

Compile an individual app:
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

Examples:
- Graph Analytics:
  - Run Pagerank on test graph G1 (8 vertices) (until convergence):
    - `make run app=pr np=4 args="data/graph_analytics/g1_8_8_13.bin 8"`
  - Run Pagerank on test graph G1 (8 vertices) (20 iterations):
    - `make run app=pr np=4 args="data/graph_analytics/g1_8_8_13.bin 8 20"`
  - Run SSSP on (weighted) test graph G1 (8 vertices) (root vertex id = 5):
    - `make run app=sssp np=4 args="data/graph_analytics/g1_8_8_13.w.bin 8 5"`
  - Run BFS on test graph G1 (8 vertices) (root vertex id = 5):
    - `make run app=bfs np=4 args="data/graph_analytics/g1_8_8_13.bin 8 5"`
  - Run TC on test graph G1 (8 vertices):
    - `make run app=tc np=4 args="data/graph_analytics/g1_8_8_13.bin 8"`
- Information Retrieval:
  - Run BM25 w/ top-k (k=3) on test graph G2 (bipartite: 7 docs, 7 terms, 17 edges):
    - `dd=data/information_retrieval; g=g2_7_7_17; ... `
    - `make ir run app=bm25 np=4 args="$dd/$g.w.bp.bin $dd/$g.term.labels 7 7 $dd/$g.queries 3"`
  - Run Language Modeling (LM) w/ top-k (k=3) on test graph G2:
    - `dd=data/information_retrieval; g=g2_7_7_17; ...`
    - `make ir run app=lm np=4 args="$dd/$g.w.bp.bin $dd/$g.term.labels 7 7 $dd/$g.queries 3"`
- Graph Simulation:
  - Run Graph Simulation on test graph G3 (13 vertices) and query graph Q1 (4 vertices):
    - `dd=data/graph_simulation; g=g3_13_13_20; q=q1_4_4_6; ...`
    - `make gs run app=graphsim np=4 args="$dd/$g.labels $dd/$g.bin $dd/$q.labels $dd/$q.mtx"`
    
Converting a dataset from Matrix Market format to LA3 binary format:
- `bin/tools/mtx2bin <filepath_in> <filepath_out> [-hi[o]] [-wi] [-wo{i|d}]`
  - `[-hi[o]] .... read in first non-comment line as header [and write [o]ut]`
  - `[-wi] ....... read in edge weights (must be int or double)`
  - `[-wo{i|d}] .. write out edge weights ({i}nt/{d}ouble) (rand [1,128] if none given)`
- Examples:
  - LiveJournal (text pairs to binary pairs):
    - `bin/tools/mtx2bin soc-LiveJournal1.txt livejournal.bin`
  - LiveJournal (text pairs to binary triples): 
    - `bin/tools/mtx2bin soc-LiveJournal1.txt livejournal.w.bin -woi`
    
Converting a dataset from LA3 binary format to Matrix Market format:
- `bin/tools/bin2mtx <filepath_in> <filepath_out> [-hi[o]] [-wi] [-wo{i|d}]`
  - `[-hi[o]] .... read in header [and write [o]ut]`
  - `[-wi] ....... read in edge weights (must be int)`
  - `[-wo[r]] .... write out edge weights (int) (by default 1, or rand [1,128] if [r])`
- Examples:
  - LiveJournal (binary pairs to text pairs):
    - `bin/tools/bin2mtx livejournal.bin soc-LiveJournal1.txt`
  - LiveJournal (binary triples to text triples): 
    - `bin/tools/bin2mtx soc-LiveJournal1.txt livejournal.w.bin -wi -wo`
    
