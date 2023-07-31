import numpy as np
from sklearn.neighbors import NearestNeighbors


class GraphController:
    def __init__(self, graph_size, start, k_size, obstacles=None):
        self.graph_size = graph_size
        self.start = np.array(start)
        self.k_size = k_size
        self.obstacles = obstacles
        self.node_coords = None
        self.graph = Graph()
        self.dijkstra_dist = []
        self.dijkstra_prev = []

    def generate_graph(self):
        self.node_coords = np.random.rand(self.graph_size, 2)
        self.start = self.start.reshape(1, 2)
        self.node_coords = np.vstack([self.start, self.node_coords])

        if self.obstacles is None:
            X = self.node_coords
            knn = NearestNeighbors(n_neighbors=self.k_size)
            knn.fit(X)
            distances, indices = knn.kneighbors(X)
            for i, p in enumerate(X):
                for j, neighbour in enumerate(X[indices[i][:]]):
                    a = str(self.find_node_index(p))
                    b = str(self.find_node_index(neighbour))
                    self.graph.add_node(a)
                    self.graph.add_edge(a, b, distances[i, j])
            self.calc_all_path_cost()
        else:
            raise NotImplementedError
        # import matplotlib.pyplot as plt
        # plt.figure(dpi=200)
        # graph = [list(map(int, node)) for node in self.graph.edges.values()]
        # for i in range(101):
        #     for j in range(9):
        #         plt.plot([X[i, 0], X[graph[i][j + 1], 0]], [X[i, 1], X[graph[i][j + 1], 1]], c='gray', alpha=0.2)
        # for i in range(101):
        #     plt.scatter(X[i, 0], X[i, 1], c='gray', s=30)
        # plt.xlim(-0.1, 1.1)
        # plt.ylim(-0.1, 1.1)
        # plt.axis('equal')
        # plt.show()
        return X, self.graph.edges

    def find_node_index(self, p):
        if self.obstacles is None:
            X = self.node_coords
        else:
            raise NotImplementedError
        return np.where(np.linalg.norm(X - p, axis=1) < 1e-5)[0][0]

    def find_point_from_node(self, n):
        if self.obstacles is None:
            X = self.node_coords
        else:
            raise NotImplementedError
        return X[int(n)]

    def calc_all_path_cost(self):
        for coord in self.node_coords:
            start_node = str(self.find_node_index(coord))
            dist, prev = dijkstra(self.graph, start_node)
            self.dijkstra_dist.append(dist)
            self.dijkstra_prev.append(prev)

    def calc_distance(self, current, destination):
        start_node = str(self.find_node_index(current))
        end_node = str(self.find_node_index(destination))
        if start_node == end_node:
            return 0
        # path_to_end = to_array(self.dijkstra_prev[int(start_node)], end_node)
        # if len(path_to_end) <= 1: # not expand this node
        #     return 1000
        distance = self.dijkstra_dist[int(start_node)][end_node]
        distance = 0 if distance is None else distance
        return distance


class Edge:
    def __init__(self, to_node, length):
        self.to_node = to_node
        self.length = length


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = dict()

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, from_node, to_node, length):
        edge = Edge(to_node, length)
        if from_node in self.edges:
            from_node_edges = self.edges[from_node]
        else:
            self.edges[from_node] = dict()
            from_node_edges = self.edges[from_node]
        from_node_edges[to_node] = edge


def dijkstra(graph, source):
    q = set()
    dist = {}
    prev = {}
    for v in graph.nodes:       # initialization
        dist[v] = float('Infinity')      # unknown distance from source to v
        prev[v] = float('Infinity')      # previous node in optimal path from source
        q.add(v)                # all nodes initially in q (unvisited nodes)

    # distance from source to source
    dist[source] = 0
    while q:
        # node with the least distance selected first
        u = min_dist(q, dist)
        q.remove(u)
        try:
            if u in graph.edges:
                for _, v in graph.edges[u].items():
                    alt = dist[u] + v.length
                    if alt < dist[v.to_node]:
                        # a shorter path to v has been found
                        dist[v.to_node] = alt
                        prev[v.to_node] = u
        except:
            pass
    return dist, prev


def min_dist(q, dist):
    """
    Returns the node with the smallest distance in q.
    Implemented to keep the main algorithm clean.
    """
    min_node = None
    for node in q:
        if min_node is None:
            min_node = node
        elif dist[node] < dist[min_node]:
            min_node = node
    return min_node


def to_array(prev, from_node):
    """Creates an ordered list of labels as a route."""
    previous_node = prev[from_node]
    route = [from_node]

    while previous_node != float('Infinity'):
        route.append(previous_node)
        temp = previous_node
        previous_node = prev[temp]

    route.reverse()
    return route


if __name__ == '__main__':
    graph_ctrl = GraphController(100, [0.5,0.5], 10)
    _, graph = graph_ctrl.generate_graph()
    edge_inputs = []
    for node in graph.values():
        node_edges = list(map(int, node))
        edge_inputs.append(node_edges)
    print(edge_inputs)