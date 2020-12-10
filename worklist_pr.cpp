#include <iostream>
#include <cstdint>
#include <cassert>
#include <vector>
#include <CL/sycl.hpp>
#include "sycl_csr_graph.h"
#include "stats.h"

#define ALPHA 0.85
#define EPSILON 0.000001

#define CPU_WLIST_CROSSOVER 10000

#define MAX_WGROUP_BLOCKS 100

#define SCALE_FACTOR 0x100000
#define MAX_ITER 200

// My attempt at push-based PageRank (with a worklist)

namespace sycl = cl::sycl;

typedef struct {
    int src;
    int offset;
} extra_point;

Stats stats;

void normalize_weights(float * weights, int n) {
    float norm = 0.0;
    int i;
    for (i = 0; i < n; i++) {
        norm += weights[i];
    }
    for (i = 0; i < n; i++) weights[i] /= norm;
}

template<typename T> void print_array(T* arr, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << " ";
        if ((i == 5) && (i < size-5)) {
            std::cout << "... ";
            i = size - 5;
        }
    }
    std::cout << endl;
}

void check_dups(int * arr, int size, int max_el) {
    int * marks = (int*)malloc(max_el*sizeof(int));
    memset(marks, 0, max_el*sizeof(int));
    int total = 0;
    for (int i = 0; i < size; i++) {
        if (marks[arr[i]] && arr[i]) {
            if (total == 0) std::cout << "Element " << arr[i] << " occurs at positions " << marks[arr[i]]-1 << " and " << i << std::endl;
            total++;
        }
        marks[arr[i]] = i+1;
    }
    std::cout << "Total duplicates " << total << std::endl;
    free(marks);
}

// Exploratory code to determine what is possible with SYCL atomics
void atomic_experiments(sycl::device device, sycl::queue queue)
{
    int n = 1024*256;
    int * arr = (int*)malloc(n*sizeof(int));
    memset(arr, 0, n*sizeof(int));

    {
        sycl::buffer<int, 1> buf(arr, sycl::range<1>(n));

        queue.submit([&] (sycl::handler& cgh) {
            auto acc = buf.get_access<sycl::access::mode::atomic>(cgh);

            cgh.parallel_for<class atomic_test_iter>(
                sycl::range<1>(n),
                [=] (sycl::item<1> item) {
                    size_t id = item.get_linear_id();
                    sycl::atomic_fetch_add(acc[(id*id) % n], 1);
                }
            );

        });
        queue.wait_and_throw();
    }

    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }

    std::cout << "Expected sum: " << n << ", actual sum " << sum << std::endl;
}


// When the worklist size drops below 100, we switch to the CPU to avoid overhead from launching kernels
int serial_push_pagerank(SYCL_CSR_Graph * g, int * in_wl, int * out_wl, float* weights, unsigned int * residuals, int wl_size, int iter)
{
    std::cout << "Switching to CPU, starting WL size " << wl_size << std::endl;
    int cpu_rounds = 0;
    while (iter < MAX_ITER && wl_size > 0) {
        int new_wl_size = 0;
        for (int i = 0; i < wl_size; i++) {
            int node = in_wl[i];
            float old_res = residuals[node];
            residuals[node] = 0;
            weights[node] += old_res/(float)SCALE_FACTOR;
            unsigned int update = old_res * ALPHA / (float)(g->nodeDegree[node]);
            for (int edge = g->nodePtr[node]; edge < g->nodePtr[node+1]; edge++) {
                int dst = g->data[edge];
                unsigned int dst_res = residuals[dst];
                residuals[dst] += update;
                if (dst_res < SCALE_FACTOR*EPSILON && dst_res + update >= SCALE_FACTOR*EPSILON) {
                    out_wl[new_wl_size++] = dst;
                }
            }
        }

        wl_size = new_wl_size;
        int * tmp = in_wl;
        in_wl = out_wl;
        out_wl = tmp;
        iter++;
        cpu_rounds++;
        std::cout << "Completed " << cpu_rounds << " on the CPU, " << iter << " rounds total." << std::endl;
        std::cout << "Out worklist size " << wl_size << std::endl; 
    }
    return cpu_rounds;
}

void push_based_pagerank(SYCL_CSR_Graph * g, sycl::device device, sycl::queue queue)
{
    auto wgroup_size = device.get_info<sycl::info::device::max_work_group_size>();
    int n = g->numNodes;

    // Initialize worklists
    int max_wl_size = 1.1*n;
    int * in_wl = (int*)malloc(max_wl_size*sizeof(int));
    int * out_wl = (int*)malloc(max_wl_size*sizeof(int));

    int heap_block_size = wgroup_size;
    // The 2 is a safety factor
    // The maximum bloat that can occur is one extra block for each work group
    // And we need at least ceil(n/heap_block_size) blocks for the initial worklist
    int heap_size = heap_block_size * (n/heap_block_size + n/wgroup_size + 2);
    //int heap_size = heap_block_size * (n/heap_block_size) + 1;
    // The idea is that we store all  
    int * heap = (int*)malloc(heap_size*sizeof(int));
    // If a work-group needs extra blocks, we mark the 'excess nodes' with their offsets
    extra_point * extra_mask = (extra_point*)malloc(n*sizeof(extra_point));
    //int * dup_mask = (int*)malloc(n*sizeof(int));
    

    memset(out_wl, 0, max_wl_size * sizeof(int));
    memset(in_wl, 0, max_wl_size * sizeof(int));
    memset(extra_mask, 0, n*sizeof(extra_point));
    //memset(dup_mask, 0, n*sizeof(int));

    int i;
    for (i = 0; i < n; i++) {
        in_wl[i] = i;
    }

    // Initialize residuals and weights
    // We have to use longs for residuals because SYCL doesn't support atomic_add on floats :-(
    unsigned int * residuals = (unsigned int*)malloc(n*sizeof(unsigned int));
    memset(residuals, 0, n*sizeof(unsigned int));
    float * weights = (float*)malloc(n*sizeof(float));
    /*for (i = 0; i < n; i++) {
        weights[i] = 1-ALPHA;
        int degree = g->getOutDegree(i);
        for (int edge = g->getEdgeStart(i); edge < g->getEdgeEnd(i); edge++) {
            int dst = g->getEdgeDst(edge);
            residuals[dst] += (ALPHA*(1-ALPHA)/degree) * SCALE_FACTOR;
        }
    }*/
    stats.checkpoint("preprocessing");
    int counters[] = {0,0};

    int iter;
    int wl_size = n;

    // SYCL scope
    {

        // Buffers for residuals & weights
        sycl::buffer<unsigned int, 1> res_buf(residuals, sycl::range<1>(n));
        sycl::buffer<float, 1> weight_buf(weights, sycl::range<1>(n));
        // Buffers to access graph data
        //sycl::buffer<int, 1> deg_buf(g->nodeDegree, sycl::range<1>(n));
        sycl::buffer<int, 1> nodePtr_buf(g->nodePtr, sycl::range<1>(n+1));
        sycl::buffer<int, 1> edgeDst_buf(g->data, sycl::range<1>(g->numEdges));
        // Worklist related buffers
        sycl::buffer<int, 1> in_wl_buf(in_wl, sycl::range<1>(max_wl_size));
        sycl::buffer<int, 1> out_wl_buf(out_wl, sycl::range<1>(max_wl_size));
        sycl::buffer<int, 1> heap_buf(heap, sycl::range<1>(heap_size));
        sycl::buffer<extra_point, 1> extra_buf(extra_mask, sycl::range<1>(n));
        //sycl::buffer<int, 1> dup_buf(dup_mask, sycl::range<1>(n));
        
        queue.submit([&] (sycl::handler& cgh) {
            // create accessors
            // read-only
            //auto deg = deg_buf.get_access<sycl::access::mode::read>(cgh);
            auto nodePtr = nodePtr_buf.get_access<sycl::access::mode::read>(cgh);
            auto edgeDst = edgeDst_buf.get_access<sycl::access::mode::read>(cgh);
            // read-write
            auto weights = weight_buf.get_access<sycl::access::mode::read_write>(cgh);
            //atomic
            auto residuals = res_buf.get_access<sycl::access::mode::atomic>(cgh);
            cgh.parallel_for<class worklist_pr_init>(
                sycl::range<1>(n),
                [=] (sycl::item<1> item) {
                    size_t local_id = item.get_linear_id();
                    weights[local_id] = 1-ALPHA;
                    int start = nodePtr[local_id];
                    int end = nodePtr[local_id+1];
                    float denom = end-start;
                    unsigned int update = SCALE_FACTOR * (ALPHA)*(1-ALPHA) / (denom);
                    for (int j = start; j < end; j++) {
                        int dst = edgeDst[j];
                        residuals[dst].fetch_add(update);
                    }
                });
        });

        int max_its = 200;
        for (iter = 0; iter < max_its; iter++) {
        // Begin counter scope
        if (!wl_size || wl_size < CPU_WLIST_CROSSOVER) break;

        {
        auto n_wgroups = ((wl_size+wgroup_size-1) / wgroup_size);
        stats.add_datapoint("n_wgroups", n_wgroups);
        stats.add_datapoint("wl_size", wl_size);

        // Initially, each work group has its own block of memory
        counters[0] = n_wgroups;
        counters[1] = 0;
        // Buffer to track global heap counter
        sycl::buffer<int, 1> counter_buf(counters, sycl::range<1>(2));

        // Submit iteration
        queue.submit([&] (sycl::handler& cgh) {
            // create accessors
            // read-only
            //auto deg = deg_buf.get_access<sycl::access::mode::read>(cgh);
            auto nodePtr = nodePtr_buf.get_access<sycl::access::mode::read>(cgh);
            auto edgeDst = edgeDst_buf.get_access<sycl::access::mode::read>(cgh);
            // read-write
            auto weights = weight_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto in_wl = in_wl_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto out_wl = out_wl_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto heap = heap_buf.get_access<sycl::access::mode::read_write>(cgh);
            auto extra = extra_buf.get_access<sycl::access::mode::read_write>(cgh);
            //atomic
            auto residuals = res_buf.get_access<sycl::access::mode::atomic>(cgh);
            //auto dup_mask = dup_buf.get_access<sycl::access::mode::atomic>(cgh);

            // counters
            auto counter = counter_buf.get_access<sycl::access::mode::atomic>(cgh);

                sycl::accessor
                    <unsigned int,
                     1,
                     sycl::access::mode::atomic,
                     sycl::access::target::local>
                    local_counters(sycl::range<1>(2), cgh);
                sycl::accessor
                    <unsigned int,
                     1,
                     sycl::access::mode::read_write,
                     sycl::access::target::local>
                    local_nonatomic_counters(sycl::range<1>(3), cgh);

                sycl::accessor
                    <int,
                     1,
                     sycl::access::mode::read_write,
                     sycl::access::target::local>
                    local_block_pointers(sycl::range<1>(MAX_WGROUP_BLOCKS), cgh);


            cgh.parallel_for<class worklist_pr_iter>(
                sycl::nd_range<1>(n_wgroups*wgroup_size, wgroup_size),
                [=] (sycl::nd_item<1> item) {
                    size_t local_id = item.get_local_linear_id();
                    size_t global_id = item.get_global_linear_id();
                    size_t group_id = item.get_group_linear_id();
                    if (local_id == 0) {
                        sycl::atomic_store(local_counters[0], (unsigned int)0);
                        sycl::atomic_store(local_counters[1], (unsigned int)0);
                        local_block_pointers[0] = group_id;
                    }

                    item.barrier(sycl::access::fence_space::local_space);

                    int i = group_id*wgroup_size + local_id;
                    int have_work = (i < wl_size);
                    int have_extra = 0;
                    int node, start, end;
                    float denom;                    
                    if (have_work) {
                        node = in_wl[i];
                        start = nodePtr[node];
                        end = nodePtr[node+1];
                        denom = end-start;
                    }

                    if (have_work) {
                    unsigned int old_res = sycl::atomic_fetch_and(residuals[node], (unsigned int)0);

                    weights[node] += ((float)old_res) / SCALE_FACTOR;
                    unsigned int update = ((float)old_res * ALPHA) / denom;
                    if (update) {
                    for (auto j = start; j < end; j++) {
                        auto dst = edgeDst[j];
                        unsigned int old_other_res = residuals[dst].fetch_add(update);
                        if (old_other_res < SCALE_FACTOR*EPSILON && old_other_res + update >= SCALE_FACTOR*EPSILON) {
                            // Don't add to the worklist if we've already processed it
                            /*if (dup_mask[dst].fetch_max(iter+1) == iter+1) {
                                //dup_mask[0].fetch_add(1);
                                continue;
                            }*/
                            // Need to increment the local offset counter
                            unsigned int old_offset = sycl::atomic_fetch_add(local_counters[0], (unsigned int)1);
                            //dup_mask[1].fetch_add(1);
                            //unsigned int allocated_blocks = local_counters[1].load() + 1;

                            if (old_offset < heap_block_size) {
                                int block_pointer = group_id;
                                heap[block_pointer*heap_block_size + old_offset%heap_block_size] = dst;
                            }

                            // Allocate a new block of memory
                            else if (old_offset % heap_block_size == 0) {
                                int old_counter = sycl::atomic_fetch_add(counter[0], 1);
                                int old_allocated_blocks = local_counters[1].fetch_add(1);
                                heap[old_counter*heap_block_size] = dst;
                                local_block_pointers[old_offset/heap_block_size] = old_counter;
                            }

                            // We got here before allocation of another block suceeded.
                            // we save the offset in a global array, then do another loop after we get a chance to
                            // sync local memory
                            // we add one here so we can use 0 to mean empty
                            else {
                                extra_point tmp;
                                tmp.src = node;
                                tmp.offset = old_offset + 1;
                                extra[dst] = tmp;
                                have_extra = 1;
                            }
                        }
                    }
                    } // end if have update

                    } // end if                    

                    // sync threads
                    item.barrier(sycl::access::fence_space::global_and_local);
                    if (have_work && have_extra) {
                        for (auto j = start; j < end; j++) {
                            auto dst = edgeDst[j];
                            extra_point tmp = extra[dst];
                            // not only does this node need adding to a worklist, 
                            // it also has to have been put into extra_point by an update from THIS node
                            if (tmp.offset > 0 && tmp.src == node) {
                                auto offset = tmp.offset - 1;
                                int block_pointer = local_block_pointers[offset/heap_block_size];
                                heap[block_pointer*heap_block_size + offset%heap_block_size] = dst;
                                // Clear from extra
                                tmp.offset = tmp.src = 0;
                                extra[dst] = tmp;
                            }
                        }
                    }
                    // sync threads
                    item.barrier(sycl::access::fence_space::global_and_local);
                    // now we know that the heap and our local_block_pointers are in a consistent state
                    // we can begin writing to out_wl

                    // Step 1 - copy the local counter into non-atomic memory
                    if (local_id == 0) {
                        int local_wlist_size = local_nonatomic_counters[0] = (int)(local_counters[0].load());
                        local_nonatomic_counters[1] = local_counters[1].load();
                        // Reserve space on out_wl for the local worklist
                        local_nonatomic_counters[2] = sycl::atomic_fetch_add(counter[1], (int)local_wlist_size);
                    }

                    item.barrier(sycl::access::fence_space::global_and_local);
                    int start_offset = local_nonatomic_counters[2];
                    int local_wlist_size = local_nonatomic_counters[0];
                    int extra_allocated_blocks = local_nonatomic_counters[1];
                    // Step 2 - copy from the heap to out-wl
                    for (int k = local_id; k < local_wlist_size; k += wgroup_size) {
                        out_wl[start_offset + k] = heap[local_block_pointers[k/heap_block_size]*heap_block_size + k%heap_block_size];
                    }
                }
            );

        });

        /*try {
            queue.wait_and_throw();
        } catch (const cl::sycl::exception& e) {
            std::cout << e.what() << std::endl;
            break;
        }*/
        } // end counter scope to force copy back to device
        /*std::cout << "Completed " << iter+1 << " iterations" << std::endl;
        std::cout << "Heap counter " << counters[0] << std::endl;
        std::cout << "Out worklist size " << counters[1] << std::endl;*/
        stats.add_datapoint("heap_counter", counters[0]);
        wl_size = counters[1];
        auto tmp = in_wl_buf;
        in_wl_buf = out_wl_buf;
        out_wl_buf = tmp;
        }

    }

    // Finish off on CPU
    int cpu_iters = 0;
    if (wl_size) {
        // If we have completed an odd number of iterations, we need to switch the pointers
        if (iter % 2) {
            int * tmp = in_wl;
            in_wl = out_wl;
            out_wl = tmp;
        }
        cpu_iters = serial_push_pagerank(g, in_wl, out_wl, weights, residuals, wl_size, iter);
    }

    stats.add_stat("iterations", iter+cpu_iters);
    stats.add_stat("gpu_iterations", iter);
    stats.checkpoint("pagerank");
    std::cout << "Weights before normalization: ";
    print_array(weights, n);
    std::cout << "Weights after normalization: ";
    normalize_weights(weights, n);
    stats.checkpoint("normalize");
    stats.stop();
    print_array(weights, n);

    free(in_wl);
    free(residuals);
    free(weights);
    free(out_wl);
    free(heap);
    free(extra_mask);
//    free(dup_mask);
}


int main (int argc, char** argv)
{

    if (argc < 2) {
        std::cout << "Usage graphfile [outputstatsfile]" << std::endl;
        return 1;
    }

    stats.start();
    SYCL_CSR_Graph g;
    g.load(argv[1]);

    std::vector<cl::sycl::device> devices = sycl::device::get_devices();

    sycl::device device;
    int first = 1;

    for (const auto& dev : devices) {
        if (!first && dev.is_gpu()) {
            device = dev;
            break;
        }
        if (dev.is_gpu()) first = 0;
    }
//    return 0;

    std::cout << "Running on "
            << device.get_info<sycl::info::device::name>()
            << "\n";

    //sycl::device device = sycl::gpu_selector{}.select_device();
    sycl::queue queue(device, [] (sycl::exception_list el) {
        for (auto ex : el) { std::rethrow_exception(ex); }
    } );
    stats.checkpoint("load");

    try {
        push_based_pagerank(&g, device, queue);
    } catch (sycl::exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    if (argc > 2) stats.json_dump(argv[2]);
    return 0;
}
