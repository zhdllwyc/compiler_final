#include <iostream>
#include <cstdint>
#include <cassert>
#include <CL/sycl.hpp>
#include "sycl_csr_graph.h"


namespace sycl = cl::sycl;

//source: https://www.nvidia.co.uk/content/cudazone/CUDABrowser/downloads/Accelerate_Large_Graph_Algorithms/HiPC.pdf
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
    sycl::device device = sycl::gpu_selector{}.select_device();
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
    
    int n = g->numNodes; //number of nodes
    int * Frontier = (int*)malloc(g->numNodes*sizeof(int)); //the frontier, the level of this iteration
    int * Visited = (int*)malloc(g->numNodes*sizeof(int));//visited node
    int * Level = (int*)malloc(g->numNodes*sizeof(int));//the level of each node from the source node
    int * done = (int*)malloc(1*sizeof(int)); // whether we have done
    int * local = (int*)malloc(g->numNodes*sizeof(int));//visited node
    int * global = (int*)malloc(g->numNodes*sizeof(int));//visited node
    int * group = (int*)malloc(g->numNodes*sizeof(int));//visited node
    
    for (int i = 0; i < g->numNodes; i++) Frontier[i] = 0;
    for (int i = 0; i < g->numNodes; i++) Visited[i] = 0;
    for (int i = 0; i < g->numNodes; i++) Level[i] = (g->numNodes+1);
    Frontier[starting_node] = 1;
    Level[starting_node]=0;
    done[0]=0;
    
    //auto part_size = wgroup_size * 2;
    auto n_wgroups = (n+wgroup_size)/ wgroup_size;
    std::cout<<n_wgroups<<std::endl;
    //while not done
    while(done[0]==0){   
    {
           done[0] = 1;
           sycl::buffer<int, 1> Frontier_buf(Frontier, sycl::range<1>(n));
           sycl::buffer<int, 1> Visited_buf(Visited, sycl::range<1>(n));
           sycl::buffer<int, 1> Level_buf(Level, sycl::range<1>(n));
           sycl::buffer<int, 1> done_buf(done, sycl::range<1>(1));
           sycl::buffer<int, 1> nodePtr_buf(g->nodePtr, sycl::range<1>(n+1));
           sycl::buffer<int, 1> edgeDst_buf(g->data, sycl::range<1>(g->numEdges));
           sycl::buffer<int, 1> local_buf(local, sycl::range<1>(n));
           sycl::buffer<int, 1> global_buf(global, sycl::range<1>(n));
           sycl::buffer<int, 1> group_buf(group, sycl::range<1>(n));
           
           queue.submit([&] (sycl::handler& cgh) {
                auto Frontier_submit = Frontier_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto Visited_submit = Visited_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto Level_submit = Level_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto done_submit = done_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto nodePtr = nodePtr_buf.get_access<sycl::access::mode::read>(cgh);
                auto edgeDst = edgeDst_buf.get_access<sycl::access::mode::read>(cgh);
                auto global_submit = global_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto group_submit = group_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto local_submit = local_buf.get_access<sycl::access::mode::read_write>(cgh);
                //for each vertex V in parallel do
                cgh.parallel_for<class bfs_OP>(
                    sycl::nd_range<1>(n_wgroups*wgroup_size, wgroup_size),
                    [=] (sycl::nd_item<1> item){

                        size_t global_id = item.get_global_linear_id();
                        size_t group_id = item.get_group_linear_id();
                        size_t local_id = item.get_local_linear_id();
                        size_t index = wgroup_size*group_id+local_id;   
                        global_submit[index] = index;
                        group_submit[index] = group_id;             
                        local_submit[index] = local_id; 
                        if (group_id >= n_wgroups) return;        
                        if( index >=n) return;
                        
                        if(Frontier_submit[index] == 1){
                            Frontier_submit[index] =0;
                            Visited_submit[index] =1;
                            item.barrier(sycl::access::fence_space::global_and_local);
                            
                            for (auto i = nodePtr[index]; i < nodePtr[index+1]; i++) { // for all neighbors
                                 auto src = edgeDst[i];
                                 if(Visited_submit[src]==0){
                                     Level_submit[src] = Level_submit[index]+1;
                                     Frontier_submit[src] = 1;
                                     done_submit[0] = 0;
                                 }
                            }
                            //item.barrier(sycl::access::fence_space::local_space);
                            
                        }            
                    }
                );
           });
           queue.wait_and_throw();
           std::cout << "Levels: " << std::endl;
           std::cout<<done[0]<<std::endl;
           for (int i = 0; i < n; i++) {
               std::cout <<i <<": "<<  Level[i] << "\n ";

           }
           std::cout<<std::endl;
    }

    }

    std::cout << "final Levels: " << std::endl;
    std::cout<<std::endl;
    for (int i = 0; i < n; i++) {
       std::cout <<i <<": "<<  Level[i] << " "<< local[i]<<" "<<group[i]<<" "<<global[i]<< "\n ";

    }
    std::cout<<std::endl;
    




    return 0;
}
