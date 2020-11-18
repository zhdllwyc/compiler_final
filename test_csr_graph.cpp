#include "sycl_csr_graph.h"
#include <iostream>
#include "sys/stat.h"
#include "sys/types.h"

using namespace std;

bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

int main(int argc, const char ** argv) {
    if (argc < 2) {
        cout << "Missing testcase filename!" << endl;
        return 1;
    }

    SYCL_CSR_Graph g;

    if (hasEnding(argv[1], ".txt")) {
        g.loadTxt(argv[1]);
    } else if (hasEnding(argv[1], ".gr")) {
        g.loadGalois(argv[1]);
    }

    cout << "Loaded graph with " << g.numNodes << " nodes and " << g.numEdges << " edges" << endl;
    g.printGraph();
}
