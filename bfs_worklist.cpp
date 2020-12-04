#include <iostream>
#include <cstdint>
#include <cassert>
#include <CL/sycl.hpp>
#include "sycl_csr_graph.h"
#include <chrono>


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
    int max_outdegree = g->max_outdegree;
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
    int toExplore = n;//max_outdegree*wgroup_size;//the number of nodes we need to explore
    
    int * Frontier = (int*)malloc(g->numNodes*sizeof(int)); //the frontier, the level of this iteration
    int * Visited = (int*)malloc(g->numNodes*sizeof(int));//visited node
    int * Level = (int*)malloc(g->numNodes*sizeof(int));//the level of each node from the source node
    int * done = (int*)malloc(1*sizeof(int)); // whether we have done
    
    //int * local = (int*)malloc(g->numNodes*sizeof(int));//visited node
    //int * global = (int*)malloc(g->numNodes*sizeof(int));//visited node
    //int * group = (int*)malloc(g->numNodes*sizeof(int));//visited node
    int * frontier_number = (int*)malloc(1*sizeof(int));//number of frontier nodes
    int * new_frontier_number = (int*)malloc(1*sizeof(int));//number of frontier nodes    
    for (int i = 0; i < g->numNodes; i++) Frontier[i] = 0;
    for (int i = 0; i < g->numNodes; i++) Visited[i] = 0;
    for (int i = 0; i < g->numNodes; i++) Level[i] = 0;
    Frontier[0] = starting_node;
    Level[starting_node]=1;
    frontier_number[0] = 1;
    new_frontier_number[0] = 0;
    int old_frontier_number = 1;
    done[0]=0;
    auto t1 = std::chrono::high_resolution_clock::now();
    
    //while not done
//sycl scope
   while(true){   
   {       
           auto n_wgroups = (toExplore+wgroup_size-1)/ wgroup_size;
       //    std::cout<<"nw_groups: "<<n_wgroups<<std::endl;           
         //  std::cout<<"toExplore: "<<toExplore<<std::endl;
          // std::cout << "max_degree: "<<max_outdegree << std::endl;
           sycl::buffer<int, 1> Frontier_buf(Frontier, sycl::range<1>(toExplore));
           sycl::buffer<int, 1> Visited_buf(Visited, sycl::range<1>(n));
           sycl::buffer<int, 1> Level_buf(Level, sycl::range<1>(n));
           sycl::buffer<int, 1> nodePtr_buf(g->nodePtr, sycl::range<1>(n+1));
           sycl::buffer<int, 1> edgeDst_buf(g->data, sycl::range<1>(g->numEdges));
           //sycl::buffer<int, 1> local_buf(local, sycl::range<1>(n));
           //sycl::buffer<int, 1> global_buf(global, sycl::range<1>(n));
           //sycl::buffer<int, 1> group_buf(group, sycl::range<1>(n));
           sycl::buffer<int, 1> frontier_number_buf(frontier_number, sycl::range<1>(1));
           sycl::buffer<int, 1> new_frontier_number_buf(new_frontier_number, sycl::range<1>(1));
           sycl::buffer<int, 1> done_buf(done, sycl::range<1>(1));
   
           {
           queue.submit([&] (sycl::handler& cgh) {
                auto Frontier_submit = Frontier_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto Visited_submit = Visited_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto Level_submit = Level_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto nodePtr = nodePtr_buf.get_access<sycl::access::mode::read>(cgh);
                auto edgeDst = edgeDst_buf.get_access<sycl::access::mode::read>(cgh);
                //auto global_submit = global_buf.get_access<sycl::access::mode::read_write>(cgh);
                //auto group_submit = group_buf.get_access<sycl::access::mode::read_write>(cgh);
                //auto local_submit = local_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto frontier_number_submit = frontier_number_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto new_frontier_number_submit = new_frontier_number_buf.get_access<sycl::access::mode::atomic>(cgh);
                auto done_submit = done_buf.get_access<sycl::access::mode::read_write>(cgh);
                
                sycl::accessor
                    <unsigned int,
                     1,
                     sycl::access::mode::atomic,
                     sycl::access::target::local>
                     local_frontier_counter(sycl::range<1>(1), cgh);
                sycl::accessor
                    <unsigned int,
                     1,
                     sycl::access::mode::read_write,
                     sycl::access::target::local>
                     local_visit(sycl::range<1>(max_outdegree*wgroup_size), cgh);
                
                //for each vertex V in parallel do
                cgh.parallel_for<class bfs_OP>(
                    sycl::nd_range<1>((n_wgroups)*wgroup_size, wgroup_size),
                    [=] (sycl::nd_item<1> item){
                        int frontier_number_curr = frontier_number_submit[0];//.load();
                        size_t global_id = item.get_global_linear_id();
                        size_t group_id = item.get_group_linear_id();
                        size_t local_id = item.get_local_linear_id();
                        size_t index =global_id;// wgroup_size*group_id+local_id;   
                        //global_submit[index] =global_id;// index;
                        //group_submit[index] = group_id;
                        //local_submit[index] = local_id; 
                        if((index<frontier_number_curr)&&(group_id<=n_wgroups) && (index==(wgroup_size*group_id+local_id)))
                        {
                        if((group_id == 0) && (local_id == 0)){
                            done_submit[0] = 1;
                            sycl::atomic_store(new_frontier_number_submit[0], 0);
                        }    
                        //item.barrier(sycl::access::fence_space::global_and_local);
                                    
                        if (local_id == 0) {
                                for(int i=0;i<max_outdegree*wgroup_size;i++){
                                    local_visit[i] = 0;
                                }
                                sycl::atomic_store(local_frontier_counter[0], (unsigned int)0);
                        }                       
                        item.barrier(sycl::access::fence_space::local_space);

                        int vertex_index = Frontier_submit[index];   
                        //Visited_submit[vertex_index] =1;
                        //item.barrier(sycl::access::fence_space::global_and_local);
                        int local_frontier = 0;
                        for (auto i = nodePtr[vertex_index]; i < nodePtr[vertex_index+1]; i++) { // for all neighbors
                             auto src = edgeDst[i];
                             if(Level_submit[src]==0){
 //                                sycl::atomic_fetch_add(Level_submit[src],(Level_submit[vertex_index].load()+1));                                
 
                                 Level_submit[src] = Level_submit[vertex_index]+1;
                                 // unsigned int old_frontier = sycl::atomic_fetch_add(new_frontier_number_submit[0], 1);
                                 //Frontier_submit[old_frontier] = src;        
                                 unsigned int local_old_counter = sycl::atomic_fetch_add(local_frontier_counter[0],(unsigned int)1);                      
                                 local_visit[local_old_counter] = src;
                                 //Level_submit[src] = Level_submit[vertex_index]+1;
                                 local_frontier = local_frontier+1;
                                 done_submit[0] = 0;

                             }
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                        
                        if(local_id == 0){
                             int local_curr = local_frontier_counter[0].load();
                             unsigned int old_frontier = sycl::atomic_fetch_add(new_frontier_number_submit[0], local_curr);
                             for(int i = old_frontier; i<(local_curr+old_frontier); i++){
                                 Frontier_submit[i] = local_visit[i-old_frontier];
                             }
                        }

                        }
                        item.barrier(sycl::access::fence_space::global_and_local);
                        if((group_id == 0) && (local_id == 0)){
                            //new_frontier_number_submit[0] = 1;
                            frontier_number_submit[0] = new_frontier_number_submit[0].load();
                            //sycl::atomic_store(frontier_number_submit[0], new_frontier_number_submit[0].load());
                        }
                        //item.barrier(sycl::access::fence_space::global_and_local);
                        
                    }
                );
           });
           queue.wait_and_throw();
           }

           }
           /*std::cout << "Levels: " << std::endl;
           for (int i = 0; i < n; i++) {
               std::cout <<i <<": "<<  Level[i] << " "<< local[i]<<" "<<group[i]<<" "<<global[i]<< "\n ";
           }
           //if(new_frontier_number[0]!=1){  
             //toExplore = frontier_number[0];
           //}
           
           std::cout << "Frontier: " << std::endl;
           for (int i = 0; i < toExplore; i++) {
               std::cout <<i <<": "<< Frontier[i] << "\n ";
           }
           std::cout << "frontier_number " << frontier_number[0]<<std::endl;
           std::cout << "new_frontier_number " << new_frontier_number[0]<<std::endl;
           std::cout << "done " << done[0]<<std::endl;
           */
           if((done[0]==1)){
               break;
           }
           //done[0] = 0;
           //new_frontier_number[0] = 0;
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();


    std::cout << "final Levels: "<<max_outdegree <<" "<<duration<<" "<<done[0]<< std::endl;
    std::cout<<std::endl;
    for (int i = 0; i < 100; i++) {
       std::cout <<i <<": "<<  Level[i]<< /*" "<<local[i]<<" "<<group[i]<<" "<<global[i]<<*/"\n ";

    }
    std::cout<<std::endl;
    



    return 0;
}
