# Implementing BFS and PageRank with SYCL

This repository contains code written for a final project in CS380: Advanced Topics in Compilers at UT Austin. SYCL (https://www.khronos.org/sycl/) is a C++ framework for GPU computing, based on OpenCL. It is cross-platform compatible and significantly easier to use than architecture-specific languages such as CUDA. The goal of this project was to write SYCL applications for two standard graph algorithms: Breadth First Search and PageRank. We aimed to match CUDA performance on these tasks.

## Building on Tuxedo
All of our experiments were done on the Oden Institute's Tuxedo machine (at UT Austin). To build the executables on Tuxedo, run the following commands in the source repo.

    source load-compute-cpp
    ./build_all.sh

## Running the executables

The above script will generate four executables. For all scripts the first argument is always the input graph.
Both the Galois binary format (.gr) and an edge-list text format (.txt) are accepted.

### BFS 

For all BFS implementations the starting node defaults to 0.

bfs_op - a baseline SCYL implementation of BFS that does not use a worklist

bfs_local_backup - our best SYCL implementation of BFS - uses local worklists for each work group that are combined to form a global worklist at the end of each iteration.

### PageRank

power_iteration_pr - Contains two implementations of power iteration for PageRank (adaptive and scalar). Adaptive precomputes node blocks with roughly equal workloads, while scalar naively processes each node with a single thread. The algorithm defaults to adaptive, but can be specified explicitly with a second argument (e.g ./power_iteration_pr graph.gr scalar).

worklist_pr - Our optimized worklist implementation of push-based PageRank. In order to implement push-based PageRank, we had to store the residuals in integers with a fixed scaling factor (as SCYL doesn't support atomic floating point operations, or atomic 64-bit integers). This led to some small round-off error.
