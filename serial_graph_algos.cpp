#include "serial_graph_algos.h"
#include "stdlib.h"
#include <iostream>

using namespace std;

// Contains serial (single-threaded) implementations of BFS and PageRank
// For use in debugging


// Compute the pageRank weights for a CSR graph using power iteration
// returns number of iterations, and sets *result to a pointer to an array of the converged weights
// if no convergence, an error is output and we halt.

int serialPageRank(SYCL_CSR_Graph* g, float** result, float alpha, float epsilon) {
    if (0 > alpha || alpha > 1 || epsilon <= 0) {
        cout << "Expected alpha to be between 0 and 1, and epsilon to be strictly positive for PageRank!" << endl;
        return 0;
    }

    float * res = (float*)malloc(g->numNodes * sizeof(float));
    float * temp = (float*)malloc(g->numNodes * sizeof(float));

    int niters = 0;
    int i;
    float initialProb = 1. - alpha;

    for (i = 0; i < g->numNodes; i++) res[i] = initialProb;

    while (true) {
        if (niters == MAX_PAGERANK_ITERS) {
            cout << "Maximum iterations (" << MAX_PAGERANK_ITERS << ") reached without convergence" << endl;
            break;
        }

        niters++;
        for (i = 0; i < g->numNodes; i++) temp[i] = initialProb;
        for (int node = g->start(); node < g->end(); node++) {
            int degree = g->getOutDegree(node);
            if (!degree) continue;
            //The delta that is pushed to other nodes
            float pushDelta = alpha * res[node] / degree;
            for (int edge = g->getEdgeStart(node); edge < g->getEdgeEnd(node); edge++) {
                int dstNode = g->getEdgeDst(edge);
                temp[dstNode] += pushDelta;
            }
        }

        float maxDelta = 0.0;
        for (i = 0; i < g->numNodes; i++) {
            float delta = (temp[i]>res[i]) ? temp[i] - res[i] : res[i] - temp[i];
            if (delta > maxDelta) maxDelta = delta;
        }

        // swap the pointers so res points to the computed next iteration and temp points to the
        // now uneeded last iteration (which can be overwritten)
        float * intermediate = res;
        res = temp;
        temp = intermediate;

        //Convergence achieved
        if (maxDelta < epsilon) break;

    }

    free(temp);

    // Normalize the weights to be probabilities
    float one_norm = 0.0;
    for (i = 0; i < g->numNodes; i++) one_norm += res[i];
    for (i = 0; i < g->numNodes; i++) res[i] /= one_norm;

    // Pass the computed weights to the user
    *result = res;

    return niters;
}

int serialBFS(SYCL_CSR_Graph* g, int** result, int sourceNode) {
    return 0;
}

