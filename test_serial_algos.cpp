#include "sycl_csr_graph.h"
#include "serial_graph_algos.h"
#include <iostream>

int main(int argc, const char ** argv) {
    if (argc < 2) {
        cout << "Missing testcase filename!" << endl;
        return 1;
    }

    SYCL_CSR_Graph g;
    g.load(argv[1]);

    float * weights;
    int nits = serialPageRank(&g, &weights, 0.85, 0.0001);
    cout << "PageRank converged after " << nits << " iterations" << endl;
    cout << "PageRank weights: " << endl;
    for (int i = 0; i < g.numNodes; i++) {
        cout << weights[i] << " ";
        if ((i == 5) && (i < g.numNodes-5)) {
            cout << "... ";
            i = g.numNodes - 5;
        }
    }
    cout << endl;
}
