#include <iostream>
#include <cstdint>
#include <cassert>
#include <CL/sycl.hpp>
#include "sycl_csr_graph.h"
#include "stats.h"

#define ALPHA 0.85
#define EPSILON 0.000001
//#define EPSILON 0.001

#define MAX_WGROUP_BLOCKS 100

#define SCALE_FACTOR 0x100000

// My attempt at push-based PageRank (with a worklist)

namespace sycl = cl::sycl;

typedef struct {
    int src;
    int offset;
} extra_point;

Stats stats;
void edge_push_based_bfs(SYCL_CSR_Graph * g, sycl::device device, sycl::queue queue, int starting_node)
{
    int n = g->numNodes;
    int m = g->numEdges;
    int* outDegree = g->nodeDegree;
    int max_outdegree = g->max_outdegree;
    int* edge_start = g->edge_start;
    int* edge_end = g->edge_end;
    std::cout << "Loaded graph." << std::endl;
     
    // Figure out what the work group size is (and the number of threads per work-group)
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
    std::cout << "max_outdegree " << max_outdegree << std::endl;
    if (!has_local_mem || local_mem_size < (wgroup_size * sizeof(int32_t)))
    {
       throw "Device doesn't have enough local memory!";
    }
    local_mem_size = 6000;
    int local_bound = 4000;
    //int local_copy_bound = 2000; 
    //int n = g->numNodes; //number of nodes
    int frontier_size=n;
    //auto n_wgroups = (n+wgroup_size-1)/ wgroup_size; 
    if(frontier_size<local_mem_size){
        frontier_size = local_mem_size;
    }
    auto n_wgroups = (m+wgroup_size-1)/ wgroup_size;   
    int * Frontier_lock = (int*)malloc(n*sizeof(int));
    int * Level = (int*)malloc(g->numNodes*sizeof(int));//the level of each node from the source node
    int * done = (int*)malloc(1*sizeof(int)); // whether we have done
    int * groups = (int*)malloc(n_wgroups*sizeof(int));
    for (int i = 0; i < g->numNodes; i++) Frontier_lock[i] = 0;
    for (int i = 0; i < g->numNodes; i++) Level[i] = 0;
    
    Frontier_lock[starting_node] = 1;
    Level[starting_node]=1;
    done[0]=1;
    int iteration = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    //sycl scope
    

       
    int first_time = 0;  
       while(true){
           {
           sycl::buffer<int, 1> groups_buf(groups, sycl::range<1>(n_wgroups)); 
           sycl::buffer<int, 1> nodePtr_buf(g->nodePtr, sycl::range<1>(n+1));
           sycl::buffer<int, 1> edgeDst_buf(g->data, sycl::range<1>(g->numEdges));
           sycl::buffer<int, 1> edge_start_buf(edge_start, sycl::range<1>(g->numEdges));
           sycl::buffer<int, 1> edge_end_buf(edge_end, sycl::range<1>(g->numEdges));
      
           //auto t3 = std::chrono::high_resolution_clock::now();   
           sycl::buffer<int, 1> Level_buf(Level, sycl::range<1>(n));
           sycl::buffer<int, 1> done_buf(done, sycl::range<1>(1));
           sycl::buffer<int, 1> Frontier_lock_buf(Frontier_lock, sycl::range<1>(n));
           //auto t4 = std::chrono::high_resolution_clock::now();
           //auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();
           //std::cout << "duration: "<<duration<< std::endl;
           //std::cout << "n_wgroups: "<<n_wgroups << std::endl;
           //std::cout << "n: "<<n << std::endl;

           queue.submit([&] (sycl::handler& cgh) {
                auto groups_submit = groups_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto Level_submit = Level_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto nodePtr = nodePtr_buf.get_access<sycl::access::mode::read>(cgh);
                auto edgeStart_submit = edge_start_buf.get_access<sycl::access::mode::read>(cgh);
                auto edgeEnd_submit = edge_end_buf.get_access<sycl::access::mode::read>(cgh);
                auto edgeDst = edgeDst_buf.get_access<sycl::access::mode::read>(cgh);
                auto done_submit = done_buf.get_access<sycl::access::mode::atomic>(cgh);
                auto Frontier_lock_submit = Frontier_lock_buf.get_access<sycl::access::mode::atomic>(cgh);
                sycl::accessor
                    <int,
                     1,
                     sycl::access::mode::atomic,
                     sycl::access::target::local>
                     local_counter(sycl::range<1>(1), cgh);     
                //for each edge in parallel do
                cgh.parallel_for<class edge_bfs>(
                    sycl::nd_range<1>((n_wgroups)*wgroup_size, wgroup_size),
                    [=] (sycl::nd_item<1> item){

                        size_t group_id = item.get_group_linear_id();
                        size_t local_id = item.get_local_linear_id();
                        size_t index = wgroup_size*group_id+local_id;   
                        if(local_id==0){
                            sycl::atomic_store(local_counter[0], 0);
                        }

                        item.barrier(sycl::access::fence_space::local_space);
                        item.barrier(sycl::access::fence_space::global_and_local);
                        int old_counter = -1;
                        int curr_counter = local_counter[0].load();
                        item.barrier(sycl::access::fence_space::local_space);
                        while(old_counter!=curr_counter){
                          if(index<m)
                          {
                            int Edge_start = edgeStart_submit[index];  
                            int Edge_end = edgeEnd_submit[index]; 
                            if((Level_submit[Edge_end]==0) && (Level_submit[Edge_start]!=0)){
                                int old_lock = sycl::atomic_fetch_add(Frontier_lock_submit[Edge_end], 1);
                                if(old_lock==0){//we made a change
                                    //sycl::atomic_fetch_add(done_submit[0], 1);
                                    sycl::atomic_fetch_add(local_counter[0], 1);
                                    Level_submit[Edge_end] = Level_submit[Edge_start]+1;
                                }
                            }                                        
                          }
                          item.barrier(sycl::access::fence_space::local_space);
                          item.barrier(sycl::access::fence_space::global_and_local);
                          old_counter = curr_counter;//whether globally the done[0] has change
                          curr_counter = local_counter[0].load();
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                        item.barrier(sycl::access::fence_space::global_and_local);
    //                    if(local_id == 0){
                            groups_submit[group_id] = local_counter[0].load();
      //                  }                  
                        item.barrier(sycl::access::fence_space::local_space);
                        item.barrier(sycl::access::fence_space::global_and_local);
                });
           });
            
           queue.wait_and_throw();
           }
           int total=0;
           for(int i=0;i<n_wgroups;i++){
              total = total+groups[i];
           }
           std::cout << "meta: "<<" "<<total<<" "<<done[0]<< std::endl;
           if((total==0)){
               break;
           }
       
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();


    std::cout << "final Levels: "<<max_outdegree <<" "<<duration<<" "<<iteration<< std::endl;
    std::cout<<std::endl;
    for (int i = 0; i < 100; i++) {
       std::cout <<i <<": "<<  Level[i]<<  "\n ";
    }
    std::cout<<std::endl;
    

}

void global_push_based_bfs(SYCL_CSR_Graph * g, sycl::device device, sycl::queue queue, int starting_node)
{
    int* outDegree = g->nodeDegree;
    int max_outdegree = g->max_outdegree;

    std::cout << "Loaded graph." << std::endl;
    
    // Figure out what the work group size is (and the number of threads per work-group)
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
    std::cout << "max_outdegree " << max_outdegree << std::endl;
    if (!has_local_mem || local_mem_size < (wgroup_size * sizeof(int32_t)))
    {
       throw "Device doesn't have enough local memory!";
    }
    local_mem_size = 12000;
    //int local_bound = 3000; 
    int n = g->numNodes; //number of nodes
    int toExplore = n;//max_outdegree*wgroup_size;//the number of nodes we need to explore
    int frontier_size=2*n;
    auto n_wgroups = (n+wgroup_size-1)/ wgroup_size; 
    if(frontier_size<local_mem_size){
        frontier_size = local_mem_size;
    }   
    int * Frontier = (int*)malloc(n*sizeof(int)); //the frontier, the level of this iteration
    int * Frontier_lock = (int*)malloc(n*sizeof(int));
    int * Level = (int*)malloc(g->numNodes*sizeof(int));//the level of each node from the source node
    int * Level_lock = (int*)malloc(n*sizeof(int));
    int * meta = (int*)malloc(g->numNodes*sizeof(int)); 
    int * reading_index = (int*)malloc(1*sizeof(int));//number of frontier nodes
    int * writing_index = (int*)malloc(1*sizeof(int));//number of frontier nodes
    for (int i = 0; i < n; i++) Frontier[i] = n+1;
    for (int i = 0; i < g->numNodes; i++) Frontier_lock[i] = 0;
    for (int i = 0; i < g->numNodes; i++) Level_lock[i] = 0;
    for (int i = 0; i < g->numNodes; i++) Level[i] = 0;
    for (int i = 0; i < g->numNodes; i++) meta[i] = 0;
    Frontier[0] = starting_node;
    Frontier_lock[0] = 0;
    Level_lock[starting_node] = 1;
    Level[starting_node]=1;
    reading_index[0] = 0;
    writing_index[0] = 1;
   
    auto t1 = std::chrono::high_resolution_clock::now();
    //sycl scope
    {
       sycl::buffer<int, 1> nodePtr_buf(g->nodePtr, sycl::range<1>(n+1));
       sycl::buffer<int, 1> edgeDst_buf(g->data, sycl::range<1>(g->numEdges));  
           
       sycl::buffer<int, 1> Frontier_buf(Frontier, sycl::range<1>(n));
       sycl::buffer<int, 1> Level_buf(Level, sycl::range<1>(n));
       sycl::buffer<int, 1> reading_index_buf(reading_index, sycl::range<1>(1));
       sycl::buffer<int, 1> writing_index_buf(writing_index, sycl::range<1>(1));
       sycl::buffer<int, 1> Frontier_lock_buf(Frontier_lock, sycl::range<1>(n));
       sycl::buffer<int, 1> Level_lock_buf(Level_lock, sycl::range<1>(n));
       queue.submit([&] (sycl::handler& cgh) {
           auto Frontier_submit = Frontier_buf.get_access<sycl::access::mode::read_write>(cgh);
           auto Level_submit = Level_buf.get_access<sycl::access::mode::read_write>(cgh);
           auto nodePtr = nodePtr_buf.get_access<sycl::access::mode::read>(cgh);
           auto edgeDst = edgeDst_buf.get_access<sycl::access::mode::read>(cgh);
           auto reading_index_submit = reading_index_buf.get_access<sycl::access::mode::atomic>(cgh);
           auto writing_index_submit = writing_index_buf.get_access<sycl::access::mode::atomic>(cgh);          
           auto Frontier_lock_submit = Frontier_lock_buf.get_access<sycl::access::mode::atomic>(cgh);
           auto Level_lock_submit = Level_lock_buf.get_access<sycl::access::mode::atomic>(cgh);
                //for each vertex V in parallel do
                cgh.parallel_for<class global_bfs_OP>(
                    sycl::nd_range<1>((n_wgroups)*wgroup_size, wgroup_size),
                    [=] (sycl::nd_item<1> item){
                        size_t group_id = item.get_group_linear_id();
                        size_t local_id = item.get_local_linear_id();
                        size_t index = wgroup_size*group_id+local_id;   

                        item.barrier(sycl::access::fence_space::local_space);
                        int curr_reading = reading_index_submit[0].load();//how many numbers are in frontier
                        int iterator=0;
                        while(true){
                            iterator = iterator+1;
                            if(index < n){       
                                int old_reading_index = sycl::atomic_fetch_add(reading_index_submit[0], 1);
                                if(old_reading_index<n){//the position we can read from 
                                   int vertex_index = Frontier_submit[old_reading_index];
                                         
                                   if(vertex_index!=(n+1)){
                                        Frontier_submit[old_reading_index]=n+1;
                                        for (auto i = nodePtr[vertex_index]; i < nodePtr[vertex_index+1]; i++) {
                                            auto src = edgeDst[i];
                                            if(Level_submit[src]==0){
                                                int old_lock = sycl::atomic_fetch_add(Level_lock_submit[src], 1);
                                                if(old_lock==0){
                                                    Level_submit[src] = Level_submit[vertex_index]+1;
                                                    //the position we can write to
                                                    int old_writing_index = sycl::atomic_fetch_add(writing_index_submit[0], 1);
                                                    if(old_writing_index<n){
                                                        Frontier_submit[old_writing_index] = src;
                                                    }
                                                }
                                            }
                                        }
                                    }else{//reading too much
                                        int curr_read = reading_index_submit[0].load();
                                        int curr_write = writing_index_submit[0].load();
                                        if( curr_read>=curr_write ){
                                            sycl::atomic_fetch_sub(reading_index_submit[0], 1);
                                        }
                                    }
                                }else{//reading too much
                                    sycl::atomic_fetch_sub(reading_index_submit[0], 1);
                                }   
                            }
                            int curr_writing_index = writing_index_submit[0].load();
                            if(curr_writing_index>=n){
                                break;
                            }
                        } 
                        item.barrier(sycl::access::fence_space::local_space);

                });
           });
           queue.wait_and_throw();
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();


    std::cout << "final Levels: "<<max_outdegree <<" "<<duration<< std::endl;
    for (int i = 0; i < 100; i++) {
       std::cout <<i <<": "<<  Level[i]<<  "\n ";
    }
}

void local_push_based_bfs(SYCL_CSR_Graph * g, sycl::device device, sycl::queue queue, int starting_node)
{
    int* outDegree = g->nodeDegree;
    int max_outdegree = g->max_outdegree;

    std::cout << "Loaded graph." << std::endl;
    
    // Figure out what the work group size is (and the number of threads per work-group)
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
    std::cout << "max_outdegree " << max_outdegree << std::endl;
    if (!has_local_mem || local_mem_size < (wgroup_size * sizeof(int32_t)))
    {
       throw "Device doesn't have enough local memory!";
    }
    local_mem_size = 6000;
    int local_bound = 5800;
    //int local_copy_bound = 2000; 
    int n = g->numNodes; //number of nodes
    int toExplore = n;//max_outdegree*wgroup_size;//the number of nodes we need to explore
    int frontier_size=n;
    //auto n_wgroups = (n+wgroup_size-1)/ wgroup_size; 
    if(frontier_size<local_mem_size){
        frontier_size = local_mem_size;
    }
    auto n_wgroups = (n+wgroup_size-1)/ wgroup_size;   
    int * Frontier = (int*)malloc(frontier_size*sizeof(int)); //the frontier, the level of this iteration
    int * Frontier_lock = (int*)malloc(n*sizeof(int));
    int * new_Frontier = (int*)malloc(frontier_size*sizeof(int)); //the frontier, the level of this iteration
    int * Visited = (int*)malloc(g->numNodes*sizeof(int));//visited node
    int * Level = (int*)malloc(g->numNodes*sizeof(int));//the level of each node from the source node
    int * done = (int*)malloc(1*sizeof(int)); // whether we have done
    int * meta = (int*)malloc(g->numNodes*sizeof(int)); 
    int * groups = (int*)malloc(n_wgroups*sizeof(int));
    int * frontier_number = (int*)malloc(1*sizeof(int));//number of frontier nodes
    int * new_frontier_number = (int*)malloc(1*sizeof(int));//number of frontier nodes    
    for (int i = 0; i < frontier_size; i++) Frontier[i] = 0;
    for (int i = 0; i < frontier_size; i++) new_Frontier[i] = 0;
    for (int i = 0; i < g->numNodes; i++) Frontier_lock[i] = 0;
    for (int i = 0; i < g->numNodes; i++) Level[i] = 0;
    for (int i = 0; i < g->numNodes; i++) meta[i] = 0;
    for (int i = 0; i < n_wgroups; i++) groups[i] = 0;
    Frontier[0] = starting_node;
    Frontier_lock[starting_node] = 1;
    Level[starting_node]=1;
    frontier_number[0] = 1;
    new_frontier_number[0] = 0;
    int old_frontier_number = 1;
    int total_number = 0;
    done[0]=6;
    int iteration = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    //sycl scope
    

       
       int first_time = 0;  
       while(true){
           std::cout<<"frontier_number: "<<frontier_number[0]<<std::endl;
           iteration=iteration+1;
           toExplore = frontier_number[0];  
           new_frontier_number[0] =0;   
           if(first_time==1){
               auto tmp = new_Frontier;
               new_Frontier = Frontier;
               Frontier = tmp;

           }else{
               first_time=1;

           }
           {
           sycl::buffer<int, 1> nodePtr_buf(g->nodePtr, sycl::range<1>(n+1));
           sycl::buffer<int, 1> edgeDst_buf(g->data, sycl::range<1>(g->numEdges));

      
           //auto t3 = std::chrono::high_resolution_clock::now();   
           sycl::buffer<int, 1> Frontier_buf(Frontier, sycl::range<1>(frontier_size));
           sycl::buffer<int, 1> new_Frontier_buf(new_Frontier, sycl::range<1>(frontier_size));
           sycl::buffer<int, 1> Level_buf(Level, sycl::range<1>(n));
           sycl::buffer<int, 1> frontier_number_buf(frontier_number, sycl::range<1>(1));
           sycl::buffer<int, 1> new_frontier_number_buf(new_frontier_number, sycl::range<1>(1));
           sycl::buffer<int, 1> done_buf(done, sycl::range<1>(1));
           sycl::buffer<int, 1> Frontier_lock_buf(Frontier_lock, sycl::range<1>(n));
           sycl::buffer<int, 1> group_buf(groups, sycl::range<1>(n_wgroups));
           //auto t4 = std::chrono::high_resolution_clock::now();
           //auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();
           //std::cout << "duration: "<<duration<< std::endl;
           //std::cout << "n_wgroups: "<<n_wgroups << std::endl;
           //std::cout << "n: "<<n << std::endl;

           queue.submit([&] (sycl::handler& cgh) {
                auto Frontier_submit = Frontier_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto new_Frontier_submit = new_Frontier_buf.get_access<sycl::access::mode::read_write>(cgh); 
                auto Level_submit = Level_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto nodePtr = nodePtr_buf.get_access<sycl::access::mode::read>(cgh);
                auto edgeDst = edgeDst_buf.get_access<sycl::access::mode::read>(cgh);
                auto frontier_number_submit = frontier_number_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto new_frontier_number_submit = new_frontier_number_buf.get_access<sycl::access::mode::atomic>(cgh);
                auto group_submit = group_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto done_submit = done_buf.get_access<sycl::access::mode::read_write>(cgh);
                auto Frontier_lock_submit = Frontier_lock_buf.get_access<sycl::access::mode::atomic>(cgh);
                sycl::accessor
                    <int,
                     1,
                     sycl::access::mode::atomic,
                     sycl::access::target::local>
                     local_counter(sycl::range<1>(1), cgh);
                sycl::accessor
                    <int,
                     1,
                     sycl::access::mode::atomic,
                     sycl::access::target::local>
                     local_counter_expired(sycl::range<1>(1), cgh);
                sycl::accessor
                    <int,
                     1,
                     sycl::access::mode::read_write,
                     sycl::access::target::local>
                     local_visit(sycl::range<1>(local_mem_size), cgh);
                sycl::accessor
                    <int,
                     1,
                     sycl::access::mode::read_write,
                     sycl::access::target::local>
                     local_backup(sycl::range<1>(local_mem_size), cgh);  
                sycl::accessor
                    <int,
                     1,
                     sycl::access::mode::atomic,
                     sycl::access::target::local>
                     local_backup_counter(sycl::range<1>(1), cgh);//how many space are taken
      
                //for each vertex V in parallel do
                cgh.parallel_for<class bfs_OP>(
                    sycl::nd_range<1>((n_wgroups)*wgroup_size, wgroup_size),
                    [=] (sycl::nd_item<1> item){
                        int frontier_number_curr = frontier_number_submit[0];//.load();

                        size_t global_id = item.get_global_linear_id();
                        size_t group_id = item.get_group_linear_id();
                        size_t local_id = item.get_local_linear_id();
                        size_t index = wgroup_size*group_id+local_id;   
                        if((group_id == 0) && (local_id == 0)){
                            done_submit[0] = 1;
                        }
                        if(local_id==0){
                            sycl::atomic_store(local_counter[0], 0);
                            sycl::atomic_store(local_counter_expired[0], 0);
                            group_submit[group_id] = 0;
                            sycl::atomic_store(local_backup_counter[0], 0);
                        }

                        for(int i = local_id; i<local_mem_size; i+=wgroup_size){
                            local_visit[i]=n+1;
                            local_backup[i] = n+1;
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        if(index<frontier_number_curr)
                        {
                            int vertex_index = Frontier_submit[index];   
                            for (auto i = nodePtr[vertex_index]; i < nodePtr[vertex_index+1]; i++) { // for all neighbors
                                auto src = edgeDst[i];
                                if(Level_submit[src]==0){
                                    Level_submit[src] = Level_submit[vertex_index]+1;
                                    int old_lock = sycl::atomic_fetch_add(Frontier_lock_submit[src], 1);
                                    if(old_lock==0){
                                        int old_index = sycl::atomic_fetch_add(local_counter[0], 1);
            
                                        if(old_index<local_mem_size){ // we have space in local visit 
                                            local_visit[old_index] = src;
                                        }else{ // we don't have any space left, local_counter == local_mem_size, so copy into global
                                            sycl::atomic_fetch_sub(local_counter[0], 1);
                                            unsigned int old_global = sycl::atomic_fetch_add(new_frontier_number_submit[0], 1);
                                            new_Frontier_submit[old_global] = src;
                                        }

                                    }

                                }
                            }                                        
                        }
                        item.barrier(sycl::access::fence_space::global_and_local);
                        item.barrier(sycl::access::fence_space::local_space);            
                        int curr_local_counter = local_counter[0].load();
                        int iterator = 0;
                        if(curr_local_counter<local_mem_size){ //if we still have space in the local visit
                          while(/*iterator<500*/true){
                            item.barrier(sycl::access::fence_space::local_space);
                            for(int i = local_id; i<local_mem_size; i+=wgroup_size){
                                int curr_vertex = local_visit[i];
                                int thread_counter = 0;
                                local_visit[i] = n+1;
                                if(curr_vertex == (n+1)){
                                    continue;
                                }
                                //sycl::atomic_fetch_sub(local_counter_expired[0], 1);
                                for (auto j = nodePtr[curr_vertex]; j < nodePtr[curr_vertex+1]; j++) { 
                                    auto curr_src = edgeDst[j];//this is the new frontier now
                                    if(Level_submit[curr_src]==0){
                                        Level_submit[curr_src] = Level_submit[curr_vertex]+1;
                                        int old_lock = sycl::atomic_fetch_add(Frontier_lock_submit[curr_src], 1);
                                        if(old_lock==0){
                                            int old_index = sycl::atomic_fetch_add(local_backup_counter[0], 1);
                                            if(old_index<local_mem_size){ // we have space in local visit 
                                                 local_backup[old_index] = curr_src;
                                            }else{ // we don't have any space left, local_counter == local_mem_size, so copy into global
                                                 sycl::atomic_fetch_sub(local_backup_counter[0], 1);
                                                 unsigned int old_global = sycl::atomic_fetch_add(new_frontier_number_submit[0], 1);
                                                 new_Frontier_submit[old_global] = curr_src;
                                            }
                                        }
                                    }
                                }
                            }
                            item.barrier(sycl::access::fence_space::local_space);
                            item.barrier(sycl::access::fence_space::global_and_local);
                            //local_visit is empty, local_backup contains the new frontiers.
                            curr_local_counter = local_backup_counter[0].load();//how manny element in localbackup
                            iterator = iterator+1;
                            item.barrier(sycl::access::fence_space::local_space);
                            //copy to local_visit
                            for(int i = local_id; i<local_mem_size; i+=wgroup_size){
                                local_visit[i] = local_backup[i];                                  
                                local_backup[i] = n+1;
                            }
                            item.barrier(sycl::access::fence_space::local_space);
                            item.barrier(sycl::access::fence_space::global_and_local);
                            
                            if((curr_local_counter<1)||((curr_local_counter>local_bound))){
                                sycl::atomic_store(local_counter[0], curr_local_counter);
                                break;
                            }
                            sycl::atomic_store(local_backup_counter[0], 0);    
                            item.barrier(sycl::access::fence_space::local_space);
                            item.barrier(sycl::access::fence_space::global_and_local);
                            
                          }                       
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                        item.barrier(sycl::access::fence_space::global_and_local);

                        //copy everything from local to global
                        int local_curr = local_counter[0].load();
                        item.barrier(sycl::access::fence_space::local_space);
                        item.barrier(sycl::access::fence_space::global_and_local);
                        if(local_id==0){
                         
                            int curr_new_frontier_number = sycl::atomic_fetch_add(new_frontier_number_submit[0], local_curr);
                            sycl::atomic_store(local_counter[0], curr_new_frontier_number);
                            sycl::atomic_store(local_counter_expired[0], 0);
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                        item.barrier(sycl::access::fence_space::global_and_local);          
                        if(local_curr>0){
                            int starting = local_counter[0].load();
                            for(int i = local_id; i<local_mem_size; i+=wgroup_size){
                                int curr_vertex = local_visit[i];
                                if(curr_vertex!=(n+1)){
                                    int curr_offset = sycl::atomic_fetch_add(local_counter_expired[0], 1);
                                    new_Frontier_submit[starting+curr_offset] = curr_vertex;
                                }
                            }
                        }
                        group_submit[group_id] = local_curr;
                        item.barrier(sycl::access::fence_space::local_space);
                        item.barrier(sycl::access::fence_space::global_and_local);
                                  
                       
                });
           });
            
           queue.wait_and_throw();
           }
           int total=0;
           for(int i=0;i<n_wgroups;i++){
              total = total+groups[i];
           }
           std::cout << "meta: "<<" "<<new_frontier_number[0]<<" "<<total<< std::endl;
           //frontier_number[0] = total; 
           frontier_number[0] = new_frontier_number[0];
           if((frontier_number[0]==0)){
               break;
           }
       
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();


    std::cout << "final Levels: "<<max_outdegree <<" "<<duration<<" "<<iteration<< std::endl;
    std::cout<<std::endl;
    for (int i = 0; i < 100; i++) {
       std::cout <<i <<": "<<  Level[i]<<  "\n ";
    }
    std::cout<<std::endl;
    

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
    int starting_node = std::stoi(argv[2]);
    
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

    std::cout << "Running on "
            << device.get_info<sycl::info::device::name>()
            << "\n";

    //sycl::device device = sycl::gpu_selector{}.select_device();
    sycl::queue queue(device, [] (sycl::exception_list el) {
        for (auto ex : el) { std::rethrow_exception(ex); }
    } );
    stats.checkpoint("load");
    

    try {
        int max_outdegree = g.max_outdegree;
        int num_edges = g.numEdges;
        if(max_outdegree<100000){
            local_push_based_bfs(&g, device, queue, starting_node);
        }else{        
            edge_push_based_bfs(&g, device, queue, starting_node);
        }
    } catch (sycl::exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    return 0;
}
