#include <iostream>
#include <cstdint>
#include <cassert>
#include <CL/sycl.hpp>
#include "sycl_csr_graph.h"


namespace sycl = cl::sycl;


int main (int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "Missing input graph filename." << std::endl;
        return 1;
    }
    SYCL_CSR_Graph* g = new SYCL_CSR_Graph();
    g->load(argv[1]);
    int* outDegree = g->nodeDegree;

    int starting_node = std::stoi(argv[2]);

    std::cout << "Loaded graph." << std::endl;
    
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
    int n = g->numNodes;

    int * Q = (int*)malloc(g->numNodes*sizeof(int)); //enqueue and dequeue
    int * W = (int*)malloc(g->numNodes*sizeof(int));//visited node
    int * dequeued_index = (int*)malloc(1*sizeof(int)); // the index of next dequeued vertex from Q
    int * enqueued_index = (int*)malloc(1*sizeof(int)); // the index of next dequeued vertex from Q


    for (int i = 0; i < g->numNodes; i++) W[i] = 0;
    W[starting_node]=1;
    Q[0]=starting_node;
    dequeued_index[0] = 0;
    enqueued_index[0] = 0;
   

    while(dequeued_index[0]!=n){
    {
        sycl::buffer<int, 1> Q_buf(Q, sycl::range<1>(n));
        sycl::buffer<int, 1> W_buf(W, sycl::range<1>(n));
        sycl::buffer<int, 1> deg_buf(outDegree, sycl::range<1>(n));
        sycl::buffer<int, 1> nodePtr_buf(g->nodePtr, sycl::range<1>(n+1));
        sycl::buffer<int, 1> edgeDst_buf(g->data, sycl::range<1>(g->numEdges));
        sycl::buffer<int, 1> dequeued_index_buf(dequeued_index, sycl::range<1>(1));
        sycl::buffer<int, 1> enqueued_index_buf(enqueued_index, sycl::range<1>(1));

            queue.submit([&] (sycl::handler& cgh) {
                auto Q_submit = Q_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto W_submit = W_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto deg = deg_buf.get_access<sycl::access::mode::read>(cgh);
                auto nodePtr = nodePtr_buf.get_access<sycl::access::mode::read>(cgh);
                auto edgeDst = edgeDst_buf.get_access<sycl::access::mode::read>(cgh);
                auto dequeuedIndex = dequeued_index_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto enqueuedIndex = enqueued_index_buf.get_access<sycl::access::mode::read_write>(cgh);


                cgh.parallel_for<class bfs_noOP>(
                 sycl::range<1>(1), [=](sycl::id<1> index){
                   
                    int dequeue_id = dequeuedIndex[index]; //get the current dequeue index of Q
                    int enqueue_id = enqueuedIndex[index]; //get the current enqueue index of Q
                    
                    int curr_vertex = Q_submit[dequeue_id];//current deququed
                    int curr_level = W_submit[curr_vertex];                
                    for (auto i = nodePtr[curr_vertex]; i < nodePtr[curr_vertex+1]; i++) { // for all neighbors
                            auto src = edgeDst[i];
                            if(W_submit[src]==0){//unvisited
                               W_submit[src]=curr_level+1; //mark i as visited
                               enqueue_id=enqueue_id+1;
                               Q_submit[enqueue_id] = src;
                            }         
                    }     
                    dequeuedIndex[index]=dequeue_id+1;
                    enqueuedIndex[index]=enqueue_id;    
                  }
                );

            });
            queue.wait_and_throw();

    }

    }

    std::cout << "Levels: " << std::endl;
    std::cout<<std::endl;
    for (int i = 0; i < n; i++) {
       std::cout <<i <<": "<<  W[i] << "\n ";

    }
    std::cout<<std::endl;
    




    return 0;
}
