#include <iostream>
#include <cstdint>
#include <cassert>
#include <CL/sycl.hpp>
#include "sycl_csr_graph.h"

#define ALPHA 0.85
#define EPSILON 0.000001

// PageRank with power iteration in SYCL
// This code is intended as a benchmark to test other (hopefully faster) algos against

namespace sycl = cl::sycl;

void scalar_csr(SYCL_CSR_Graph * f)
{
    SYCL_CSR_Graph* g = f->flip();
    int* outDegree = f->nodeDegree;
 
    std::cout << "Loaded and flipped graph." << std::endl;

    // Figure out what the work group size is (and the number of threads per work-group)
    sycl::device device = sycl::default_selector{}.select_device();
    sycl::queue queue(device, [] (sycl::exception_list el) {
        for (auto ex : el) { std::rethrow_exception(ex); }
    } );

    auto wgroup_size = device.get_info<sycl::info::device::max_work_group_size>();

    std::cout << "Work-group size " << wgroup_size << std::endl;

    if (wgroup_size % 2 != 0) {
        throw "Work-group size has to be even!";
    }


    auto has_local_mem = device.is_host()
        || (device.get_info<sycl::info::device::local_mem_type>()
        != sycl::info::local_mem_type::none);
    auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();

    std::cout << "Local mem size " << local_mem_size << std::endl;

    if (!has_local_mem || local_mem_size < (wgroup_size * sizeof(int32_t)))
    {
       throw "Device doesn't have enough local memory!";
    }

    float * residuals = (float*)malloc(g->numNodes*sizeof(float));
    float * next_residuals = (float*)malloc(g->numNodes*sizeof(float));
    for (int i = 0; i < g->numNodes; i++) residuals[i] = 1-ALPHA;

    int n = g->numNodes;

    // Begin C scope to enable nice SYCL memory management
    {
        sycl::buffer<float, 1> res_buf(residuals, sycl::range<1>(n));
        sycl::buffer<float, 1> next_buf(next_residuals, sycl::range<1>(n));
        // Buffers to access graph data
        // Note we need to use the outDegree of the non-flipped graph
        sycl::buffer<int, 1> deg_buf(outDegree, sycl::range<1>(n));
        sycl::buffer<int, 1> nodePtr_buf(g->nodePtr, sycl::range<1>(n+1));
        sycl::buffer<int, 1> edgeDst_buf(g->data, sycl::range<1>(g->numEdges));

        for (int iter = 0; iter < 52; iter++) {
            queue.submit([&] (sycl::handler& cgh) {
                auto deg = deg_buf.get_access<sycl::access::mode::read>(cgh);
                auto nodePtr = nodePtr_buf.get_access<sycl::access::mode::read>(cgh);
                auto edgeDst = edgeDst_buf.get_access<sycl::access::mode::read>(cgh);
                auto res = res_buf.get_access<sycl::access::mode::read>(cgh);
                auto next = next_buf.get_access<sycl::access::mode::read_write>(cgh);

                cgh.parallel_for<class pagerank_iter>(
                    sycl::range<1>(n),
                    [=] (sycl::item<1> item) {
                        size_t id = item.get_linear_id();
                        float sum = 0.0;
                        for (auto i = nodePtr[id]; i < nodePtr[id+1]; i++) {
                            auto src = edgeDst[i];
                            sum += res[src]/deg[src];
                        }
                        next[id] = ALPHA * sum + (1-ALPHA);
                    }
                );

            });
            queue.wait_and_throw();
            auto tmp = res_buf;
            res_buf = next_buf;
            next_buf = tmp;
        }
    }

    int i;
    float one_norm = 0.0;
    // normalize
    for (i = 0; i < n; i++) {
        one_norm += next_residuals[i];
    }


    cout << "PageRank weights: " << endl;
    for (i = 0; i < n; i++) {
        cout << next_residuals[i]/one_norm << " ";
        if ((i == 5) && (i < n-5)) {
            cout << "... ";
            i = n - 5;
        }
    }

    free(residuals);
    free(next_residuals);
    delete g;
}

// block size is the max size per work-group
// typically a power of two
int computeRowBlocks(int * row_lengths, int num_rows, int num_nonzero, int block_size, int** rowBlocks)
{
    int max_blocks = (num_nonzero+block_size-1) / block_size + 1;
    int * p = (int*)malloc(sizeof(int)*max_blocks);

    int size = 0;
    int block_num = 0;
    p[0] = 0;
    for (int i = 0; i < num_rows; i++) {
        // Fat row - we put it in its own block
        if (row_lengths[i] >= block_size) {
            if (size) block_num++;
            p[block_num] = i;
            block_num++;
            size = 0;
            continue;
        }
        else if (size + row_lengths[i] > block_size) {
            // If we had a block accruing before-hand, advance to a new one
            if (size) block_num++;
            // Reset the size and start a new block
            size = row_lengths[i];
            p[block_num] = i;
        }
        else size += row_lengths[i];
    }

    if (size) block_num++;
    p[block_num] = num_rows;

    // return the pointer
    *rowBlocks = p;
    if (block_num >= max_blocks) {
        std::cout << "Error in block computation, block pointer went out of bounds!" << std::endl;
    }
    // Return the total number of blocks
    return block_num;
}

// Compute power iteration for PageRank using the Adaptive CSR algorithm for sparse matrix-vector multiplication
// Inspired by https://medium.com/analytics-vidhya/sparse-matrix-vector-multiplication-with-cuda-42d191878e8f
void adaptive_csr(SYCL_CSR_Graph * f)
{
    SYCL_CSR_Graph* g = f->flip();
    int* outDegree = f->nodeDegree;
 
    std::cout << "Loaded and flipped graph." << std::endl;

    // Figure out what the work group size is (and the number of threads per work-group)
    sycl::device device = sycl::gpu_selector{}.select_device();
    sycl::queue queue(device, [] (sycl::exception_list el) {
        for (auto ex : el) { std::rethrow_exception(ex); }
    } );

    auto wgroup_size = device.get_info<sycl::info::device::max_work_group_size>();
    std::cout << "Running on "
            << device.get_info<sycl::info::device::name>()
            << "\n";

    std::cout << "Work-group size " << wgroup_size << std::endl;

    if (wgroup_size % 2 != 0) {
        throw "Work-group size has to be even!";
    }


    auto has_local_mem = device.is_host()
        || (device.get_info<sycl::info::device::local_mem_type>()
        != sycl::info::local_mem_type::none);
    auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();

    std::cout << "Local mem size " << local_mem_size << std::endl;

    int block_size = (int)2*wgroup_size;

    if (!has_local_mem || local_mem_size < (block_size * sizeof(float)))
    {
       throw "Device doesn't have enough local memory!";
    }

    float * residuals = (float*)malloc(g->numNodes*sizeof(float));
    float * next_residuals = (float*)malloc(g->numNodes*sizeof(float));
    for (int i = 0; i < g->numNodes; i++) residuals[i] = 1-ALPHA;

    int n = g->numNodes;
    int m = g->numEdges;
    int * rowBlocks;
    // Try using twice the work group size for the number of nonzeros
    int num_blocks = computeRowBlocks(g->nodeDegree, n, m, block_size, &rowBlocks);
    std::cout << "Number of blocks " << num_blocks << std::endl;

    // Begin C scope to enable nice SYCL memory management
    {
        sycl::buffer<float, 1> res_buf(residuals, sycl::range<1>(n));
        sycl::buffer<float, 1> next_buf(next_residuals, sycl::range<1>(n));
        // Buffers to access graph data
        // Note we need to use the outDegree of the non-flipped graph
        sycl::buffer<int, 1> deg_buf(outDegree, sycl::range<1>(n));
        sycl::buffer<int, 1> nodePtr_buf(g->nodePtr, sycl::range<1>(n+1));
        sycl::buffer<int, 1> edgeDst_buf(g->data, sycl::range<1>(m));
        sycl::buffer<int, 1> rowBlocks_buf(rowBlocks, sycl::range<1>(num_blocks));

        for (int iter = 0; iter < 52; iter++) {
            queue.submit([&] (sycl::handler& cgh) {
                auto deg = deg_buf.get_access<sycl::access::mode::read>(cgh);
                auto nodePtr = nodePtr_buf.get_access<sycl::access::mode::read>(cgh);
                auto edgeDst = edgeDst_buf.get_access<sycl::access::mode::read>(cgh);
                auto res = res_buf.get_access<sycl::access::mode::read>(cgh);
                auto rowBlocks = rowBlocks_buf.get_access<sycl::access::mode::read>(cgh);
                auto next = next_buf.get_access<sycl::access::mode::read_write>(cgh);

                sycl::accessor
                    <float,
                     1,
                     sycl::access::mode::read_write,
                     sycl::access::target::local>
                    local_mem(sycl::range<1>(block_size), cgh);
                cgh.parallel_for<class pagerank_adaptive_iter>(
                    sycl::nd_range<1>(num_blocks*wgroup_size, wgroup_size),
                    [=] (sycl::nd_item<1> item) {
                        size_t local_id = item.get_local_linear_id();
                        size_t global_id = item.get_global_linear_id();
                        size_t group_id = item.get_group_linear_id();
                        if (group_id >= num_blocks) return;
                        int first_node = rowBlocks[group_id];
                        int last_node = rowBlocks[group_id+1];
                        // more than one row in the block
                        // we need to apply CSR-stream
                        if (first_node - last_node > 1) {
                        
                        }
                        // We are dealing with a single row
                        else {
                           auto row_start = nodePtr[first_node];
                           auto row_end = nodePtr[first_node+1];
                           // do the sum
                        }
                        /*float sum = 0.0;
                        for (auto i = nodePtr[id]; i < nodePtr[id+1]; i++) {
                            auto src = edgeDst[i];
                            sum += res[src]/deg[src];
                        }
                        next[id] = ALPHA * sum + (1-ALPHA);*/
                    }
                );

            });
            queue.wait_and_throw();
            auto tmp = res_buf;
            res_buf = next_buf;
            next_buf = tmp;
            break;
        }
    }

    int i;
    float one_norm = 0.0;
    // normalize
    for (i = 0; i < n; i++) {
        one_norm += next_residuals[i];
    }


    cout << "PageRank weights: " << endl;
    for (i = 0; i < n; i++) {
        cout << next_residuals[i]/one_norm << " ";
        if ((i == 5) && (i < n-5)) {
            cout << "... ";
            i = n - 5;
        }
    }

    free(residuals);
    free(next_residuals);
    free(rowBlocks);
    delete g;
}

int main (int argc, char** argv)
{

    if (argc < 2) {
        std::cout << "Missing input graph filename." << std::endl;
        return 1;
    }

    SYCL_CSR_Graph* f = new SYCL_CSR_Graph();
    f->load(argv[1]);
    //scalar_csr(f);
    adaptive_csr(f);

    delete f;
    return 0;
}


