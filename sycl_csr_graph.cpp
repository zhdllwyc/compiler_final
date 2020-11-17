#include "sycl_csr_graph.h"
#include <fstream>
#include <iostream>
#include "stdlib.h"
#include <vector>
// For sorting edge lists
#include <algorithm>

void SYCL_CSR_Graph::printGraph() {
    for (int node = this->start(); node < this->end(); node++) {
        this->printNodeInfo(node);
    }
}

void SYCL_CSR_Graph::printNodeInfo(int node) {
    cout << "Node " << node << ": " << endl;
    cout << "\tDegree: " << this->getOutDegree(node) << ", nodePtr bounds: " << this->getEdgeStart(node) << " " << this->getEdgeEnd(node);
    cout << endl << "Neighbors (outgoing):";
    for (int j = this->getEdgeStart(node); j < this->getEdgeEnd(node); j++) cout << " " << this->getEdgeDst(j);
    cout << endl;
}

int SYCL_CSR_Graph::loadTxt(string filename) {
    ifstream infile(filename);

    if (!infile.is_open()) {
        cout << "Error parsing file " << filename << endl;
    }

    int n, m, i;

    if (!(infile >> n >> m)) {
        cout << "Expected first line to contain numNodes numEdges!" << endl;
        return 0;
    }
    
    this->numNodes = n;
    this->numEdges = m;

    vector<int> ** edgeLists = (vector<int>**)malloc(sizeof(vector<int>*) * n);
    //vector<vector<int>> edgeLists;

    for (i = 0; i < n; i++) {
        edgeLists[i] = new vector<int>();
    }

    int source, dest;
    while (infile >> source >> dest) {
        edgeLists[source]->push_back(dest);
    }

    this->nodeDegree = (int*)malloc(sizeof(int)*n);
    this->nodePtr = (int*)malloc(sizeof(int)*(n+1));
    this->data = (int*)malloc(sizeof(int)*m);

    this->nodePtr[0] = 0;

    for (i = 0; i < n; i++) {
        sort(edgeLists[i]->begin(), edgeLists[i]->end());
        int outDegree = this->nodeDegree[i] = edgeLists[i]->size();
        int offset = this->nodePtr[i];
        this->nodePtr[i+1] = offset + outDegree;

        for (int j = 0; j < outDegree; j++) {
            this->data[offset+j] = (*edgeLists[i])[j];
        }

        delete edgeLists[i];
    }


    free(edgeLists);

    return 1;
}

