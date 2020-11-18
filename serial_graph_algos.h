#ifndef SERIAL_GRAPH_ALGO_H
#define SERIAL_GRAPH_ALGO_H

#include "sycl_csr_graph.h"

int serialPageRank(SYCL_CSR_Graph* g, float** result, float alpha, float epsilon);
int serialBFS(SYCL_CSR_Graph* g, int** result, int sourceNode);

#define MAX_PAGERANK_ITERS 200

#endif
