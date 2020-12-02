import random
import sys

list_vertex=[]
edge_map = []
num_vertex = int(sys.argv[1])
edge_bound = int(sys.argv[2])#the number of edges in the system is bounded by  edge_bound*num_vertex
filename = sys.argv[3]
focus_vertex = sys.argv[4]

for i in range(num_vertex):
    list_vertex.append(i+1)

#we want a full connected graph
f = open(filename, "w")
if( int(edge_bound) == int(num_vertex/2) ):
    for i in range(num_vertex):
        for j in range(i+1, num_vertex):
            f.write(str(i+1))
            f.write(" ")
            f.write(str(j+1))
            f.write("\n")
else:
    if(int(focus_vertex)!=0):
        for x in list_vertex:
            edge_dst = random.randint(1, num_vertex)
            while((edge_dst==x) or ((edge_dst,x) in edge_map) or ((x,edge_dst) in edge_map)):
                edge_dst = random.randint(1, num_vertex)
            edge_map.append((x,edge_dst))
        for x in range(int(focus_vertex)):
            curr_vertex = random.randint(1, num_vertex)
            incoming_degree = random.randint(int((num_vertex/int(focus_vertex))/2), int(num_vertex/int(focus_vertex)))
            for i in range(incoming_degree):
                edge_dst = random.randint(1, num_vertex)
                while((edge_dst==curr_vertex) or ((edge_dst,curr_vertex) in edge_map) or ((curr_vertex,edge_dst) in edge_map)):
                    edge_dst = random.randint(1, num_vertex)
                edge_map.append((edge_dst, curr_vertex))
                
    else:
        for e in range(edge_bound):
            for x in list_vertex:
                print(x)
                edge_dst = random.randint(1, num_vertex)
                full = False
                while((edge_dst==x) or ((edge_dst,x) in edge_map) or ((x,edge_dst) in edge_map)):
                    edge_dst = random.randint(1, num_vertex)
                    full = True
                    for y in list_vertex:
                        if((y!=x) and ((x,y) not in edge_map) and ((y,x) not in edge_map)):
                            full = False
                    if(full):
                        break
                if(not full):    
                    edge_map.append((x,edge_dst))

#check for edge duplicate
final_edge_map=[]
for (x,y) in edge_map:
    final_edge_map.append((x-1, y-1))
    final_edge_map.append((y-1, x-1))
    

for (x,y) in final_edge_map:
    f.write(str(x))
    f.write(" ")
    f.write(str(y))
    f.write("\n")
