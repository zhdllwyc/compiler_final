import networkx as nx
import sys

# Supports only the txt graph format.
# Used to validate the serialPageRank implementation.

def load_graph(fname):
    G = nx.DiGraph()
    with open(fname, "r") as fp:
        first = True
        for line in fp:
            if not first:
                u,v = map(int, line.strip().split())
                G.add_edge(u, v)
            first = False

    return G

if __name__ == "__main__":
    G = load_graph(sys.argv[1])
    res = nx.pagerank(G, .85)
    print(res)
