#include "sycl_csr_graph.h"
#include <iostream>

using namespace std;

int main(int argc, const char ** argv) {
    if (argc < 2) {
        cout << "Missing testcase filename!" << endl;
        return 1;
    }

    SYCL_CSR_Graph g;
    g.loadTxt(argv[1]);
    cout << "Loaded graph with " << g.numNodes << " nodes and " << g.numEdges << " edges" << endl;
    g.printGraph();
}
