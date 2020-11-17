#ifndef SYCL_CSR_GRAPH_H
#define SYCL_CSR_GRAPH_H

#include <string>

using namespace std;

class SYCL_CSR_Graph {

public:

    int numNodes;
    int numEdges;

    SYCL_CSR_Graph() {
        this->numNodes = this->numEdges = 0;
        this->nodeDegree = this->nodePtr = this->data = nullptr;        
    }

    ~SYCL_CSR_Graph() {
        if (this->nodeDegree != nullptr) free(this->nodeDegree);
        if (this->nodePtr != nullptr) free(this->nodePtr);
        if (this->data != nullptr) free(this->data);
    }


    // Load the graph from a .txt file with a list of edges
    int loadTxt(string filename);

    // TODO:: figure out the format for the galois graphs and implement loading them
    //int loadGalois(string filename);
    
    // Print out the graph (for small graphs only)
    void printGraph();

    // Print out the degree, nodeptrs, and edge info for a single node
    void printNodeInfo(int node);

    int start() {
        return 0;
    }

    int end() {
        return numNodes;
    }

    // Get the outdegree for a given node
    int getOutDegree(int node) {
        return nodeDegree[node];
    }

    // Get the first edge for a 
    int getEdgeStart(int node) {
        return nodePtr[node];
    }

    int getEdgeEnd(int node) {
        return nodePtr[node+1];
    }

    int getEdgeDst(int edge) {
        return data[edge];
    }

protected:

    // Array containing the out degrees for each node
    int* nodeDegree;

    // Node i points to the nodes in data[nodePtr[i]:nodePtr[i+1]]
    // The total length of node_ptr is numNodes + 1;
    // We assume nodes are labeled with integers starting at 0
    int* nodePtr;

    // Contains the 'destinations' of the edges
    // For PageRank and BFS, we don't care about edge weights
    int* data;
};

#endif
