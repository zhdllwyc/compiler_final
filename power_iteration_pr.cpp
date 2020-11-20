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


int main (int argc, char** argv)
{

    if (argc < 2) {
        std::cout << "Missing input graph filename." << std::endl;
        return 1;
    }

    SYCL_CSR_Graph* f = new SYCL_CSR_Graph();
    f->load(argv[1]);
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
    delete f;
    return 0;
}


