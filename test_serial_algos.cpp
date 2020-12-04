#include "sycl_csr_graph.h"
#include "serial_graph_algos.h"
#include "stats.h"
#include <iostream>

int main(int argc, const char ** argv) {
    if (argc < 2) {
        cout << "Missing testcase filename!" << endl;
        return 1;
    }

    Stats s;
    s.start();
    SYCL_CSR_Graph g;
    g.load(argv[1]);
    s.checkpoint("load");

    float * weights;
    int nits = serialPageRank(&g, &weights, 0.85, 0.000001);
    s.checkpoint("pagerank");
    s.stop();
    s.add_stat("iterations", nits);

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
    if (argc > 2) {
        s.json_dump(argv[2]);
    }
}
