graph.py can be used to generate random graphs. 
python3 graph.py #numberofvertex #edge_bound outputfile #numberoffocusnode
there are three types of graphs: 
1. none focus: edge_bound bounds the number of edges by edge_bound*numberofvertex. By pass in number for #numberofvertex and #edge_bound, and setting #numberoffocusnode to be 0, we can generate random graphs(every vertex is guaranteed to be connected). 
2. fully connected graph, if #edge_bound == #numberofvertex, we will generate a fully connected graph of #numberofvertex.
3. focus: for this type of graph, #edge_bound can be set to 0. If focus is set to 10, we will pick 10 random points inside the graph and each has approximated (#numberoffocusnode/(2*10), #numberoffocusnode/10) number of incoming edge. 

All graph can be treated as directed or undirected. If edge (node1, node2) is directed, the direction is from node1 -> node2.

Pagerank(power, 30 iterations):
python3 pageRank.py "$filename" undirected/directed pagerank_(undirected/directed)_"$filename"

bfs:
python3 bfs.py "$filename" undirected outputfile.txt

test.sh is for generating answer for bfs and pagerank 

some file doesn't have an answer because it is too big, my laptop kills the bfs or pagerank task, e.g., graph_10000_5000.txt





