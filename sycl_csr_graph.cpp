#include "sycl_csr_graph.h"
#include <fstream>
#include <iostream>
#include <vector>

// For sorting edge lists
#include <algorithm>

//for mmap
#include "sys/stat.h"
#include "sys/mman.h"
#include "sys/types.h"
#include "fcntl.h"
#include "limits.h"
#include "stdlib.h"
#include "unistd.h"

void SYCL_CSR_Graph::printGraph() {
    int skipped = 0;
    for (int node = this->start(); node < this->end(); node++) {
        if ((node >= MAX_PRINT_NODES/2) && (!skipped)) {
            cout << " . . ." << endl;
            int goto_node = this->end() - MAX_PRINT_NODES/2;
            if (goto_node > node) node = goto_node;
            skipped = 1;
        }
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

void SYCL_CSR_Graph::allocateArrays() {
    int n = this->numNodes;
    int m = this->numEdges;
    this->nodeDegree = (int*)malloc(sizeof(int)*n);
    this->nodePtr = (int*)malloc(sizeof(int)*(n+1));
    this->data = (int*)malloc(sizeof(int)*m);
    this->nodePtr[0] = 0;
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

    this->allocateArrays();

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


int SYCL_CSR_Graph::loadGalois(string filename) {
    int fd = open(filename.c_str(), O_RDONLY);

    if (fd == -1 ) {
        cout << "Error opening Galois graph file " << filename << endl;
        return 0;
    }

    struct stat statbuf;
    if (fstat(fd, &statbuf) < 0) {
        cout << "fstat error for file " << filename << endl;
        return 0;
    }

    void * base = mmap(nullptr, statbuf.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        cout << "mmap error for file " << filename << endl;
        return 0;
    }

    uint64_t * data = (uint64_t *)base;
    uint64_t galois_nodes, galois_edges, galois_version;
    galois_version = data[0];
    this->numNodes = galois_nodes = data[2];
    this->numEdges = galois_edges = data[3];
    if (galois_version < 1 || galois_version > 2) {
        cout << "Invalid galois graph version " << galois_version << endl;
        munmap(base, statbuf.st_size);
        close(fd);
        return 0;
    }

    cout << "Reading Galois graph version " << galois_version << ", numNodes " << galois_nodes << " numEdges " << galois_edges << endl;

    if (galois_edges > INT_MAX || galois_nodes > INT_MAX) {
        cout << "Graph too large to fit into 32-bit integers! Aborting." << endl;
        munmap(base, statbuf.st_size);
        close(fd);
        return 0;
    }

    this->allocateArrays();

    //Skip header
    data += 4;
    //We need to copy with a for loop because the types may not be the same
    size_t i;
    for (i = 0; i < this->numNodes; i++) this->nodePtr[i+1] = data[i];

    if (galois_version == 1) {
        uint32_t* edgeArray = (uint32_t *)(data + this->numNodes);
        for (i = 0; i < this->numEdges; i++) this->data[i] = edgeArray[i];
    } else if (galois_version == 2) {
        uint64_t* edgeArray = data + this->numNodes;
        for (i = 0; i < this->numEdges; i++) this->data[i] = edgeArray[i];
    }

    for (i = 0; i < this->numNodes; i++) {
        this->nodeDegree[i] = this->nodePtr[i+1] - this->nodePtr[i];
    }

    munmap(base, statbuf.st_size);
    close(fd);
    return 1;
}
