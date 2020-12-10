#ifndef SYCL_CSR_GRAPH_H
#define SYCL_CSR_GRAPH_H

#include <string>

#define MAX_PRINT_NODES 10

using namespace std;

class SYCL_CSR_Graph {

public:

    int numNodes;
    int numEdges;
    int max_outdegree;    

    SYCL_CSR_Graph() {
        this->numNodes = this->numEdges = this->max_outdegree= 0;
        this->nodeDegree = this->nodePtr = this->data = nullptr;        
    }

    ~SYCL_CSR_Graph() {
        if (this->nodeDegree != nullptr) free(this->nodeDegree);
        if (this->nodePtr != nullptr) free(this->nodePtr);
        if (this->data != nullptr) free(this->data);
    }

    // Load the graph, infering the type from the filename
    int load(string filename);

    // Load the graph from a .txt file with a list of edges
    int loadTxt(string filename);

    // Load a graph from the binary format used by the Galois library
    int loadGalois(string filename);
    
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

    // Flip the directions of the edges and return a new graph
    // This is needed for easy pull-based PageRank
    SYCL_CSR_Graph* flip();

    // Array containing the out degrees for each node
    int* nodeDegree;

    // Node i points to the nodes in data[nodePtr[i]:nodePtr[i+1]]
    // The total length of node_ptr is numNodes + 1;
    // We assume nodes are labeled with integers starting at 0
    int* nodePtr;

    // Contains the 'destinations' of the edges
    // For PageRank and BFS, we don't care about edge weights
    int* data;
    int* edge_start;
    int* edge_end;
protected:

    // Allocate memory for the graph
    // Must be called after the number of edges and nodes are known
    void allocateArrays();
};

#endif
