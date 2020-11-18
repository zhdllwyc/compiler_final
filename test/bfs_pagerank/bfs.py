import operator
import math, random, sys, csv 
from utils import parse, print_results
import networkx as nx

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Expected input format: python pageRank.py <data_filename> <directed OR undirected>')
    else:
        filename = sys.argv[1]
        isDirected = False
        if sys.argv[2] == 'directed':
            isDirected = True
        output_filename = sys.argv[3]
        graph = parse(filename, isDirected)
        T = nx.single_source_shortest_path_length(graph,"1")
        f = open(output_filename, "w")
        for x in T:    
            f.write(str(x))
            f.write(" ")
            f.write(str(T[x]))
            f.write("\n")   
            
