#!/bin/bash

for target in bfs_local_backup bfs_op worklist_pr power_iteration_pr; do
	echo "Building $target"
	./simple-build.sh $target sycl_csr_graph.cpp
done
