import networkx as nx
import math
import heapq
from networkx.utils import UnionFind
from networkx.algorithms.approximation.steinertree import steiner_tree as nx_baseline

class Baseline:
    def __init__(self):
        self.graph = nx.Graph()
        self.terminals = []
        self.is_baseline = True

    def add_node(self, index, **attrs):
        self.graph.add_node(index, **attrs)

    def decrease_edge_weight(self, u, v, weight):
        if u == v or self._get_edge_weight(u, v) <= weight:
            return

        self.graph.add_edge(u, v, weight=weight)

    def add_terminal(self, new_index):
        self.terminals.append(new_index)

    def remove_terminal(self, index):
        self.terminals.remove(index)

    def get_steiner_tree(self):
        steiner = nx_baseline(self.graph, self.terminals)
        return steiner, steiner.size(weight='weight')

    def _get_edge_weight(self, start, end):
        return self.graph[start][end]['weight'] if self.graph.has_edge(start, end) else float('inf')

class VDSTGraph:
    """ 
    Flexible class for modeling networks and dynamically maintaining
    approximate Steiner trees.


    Object invariants:

    - after each method, self.metric_closure is the metric closure 
    of self.graph

    - within self.metric_closure, the "path" data associated with any edge uv is
    a tuple containing 0 elements (indicating that the shortest path from u to v
    is the direct one) or 1 element (giving some node along the shortest path)
    """
    def __init__(self):
        # Ground truth network
        self.graph = nx.Graph()
        self.steiner_tree = None
        self.steiner_cost = 0

        # Metric closure
        self.metric_closure = nx.Graph()
        self.metric_steiner_tree = nx.Graph()

        self.terminals = {}  # list of indices of nodes that are terminals
        self.n_terminals_ever = 0
        self.is_baseline = False


    def add_node(self, index, **attrs):
        self.graph.add_node(index, **attrs)
        self.metric_closure.add_node(index)
        for node in self.metric_closure.nodes:
            d = 0 if node == index else float('inf')
            self.metric_closure.add_edge(index, node, distance=d, path=())


    def decrease_edge_weight(self, u, v, weight):
        if u == v or self._get_edge_weight(u, v) <= weight:
            return

        self.graph.add_edge(u, v, weight=weight)

        if self._get_path_weight(u, v) <= weight:
            return

        parent_update_requests = {}
        self._update_metric_closure(u, v, weight, (), parent_update_requests)

        # Loop over each pair of nodes and possibly update pairwise shortest path info 
        for a in self.metric_closure.nodes:
            a_u = self._get_path_weight(a, u)
            a_v = self._get_path_weight(a, v)

            for b in self.metric_closure.nodes:
                if a == b or (a == u and v == b) or (a == v and b == u):
                    continue

                a_b = self._get_path_weight(a, b)
                u_b = self._get_path_weight(u, b)
                v_b = self._get_path_weight(v, b)

                # 3 possible shortest paths: a->u->v->b, a->v->u->b, a->b
                best_dist = min((a_b, a_u + weight + v_b, a_v + weight + u_b))
                if best_dist != a_b:
                    new_path = (v,) if (a != v and b != v) else (u,)
                    self._update_metric_closure(a, b, best_dist, new_path, parent_update_requests)

        # Process parent swap requests
        for child, (new_parent, distance) in parent_update_requests.items():
            # Update edges in metric Steiner tree and child bookkeeping info
            self.metric_steiner_tree.remove_edge(child, self.metric_steiner_tree.node[child]['parent'])
            self.metric_steiner_tree.add_edge(child, new_parent)
            self.metric_steiner_tree.add_node(child, parent=new_parent, distance_to_parent=distance)

        self.steiner_tree = None 


    def _update_metric_closure(self, a, b, distance, path, parent_update_requests):
        self.metric_closure.add_edge(a, b, distance=distance, path=path)

        # Update parents if we've found a shorter path between two terminals
        if a in self.terminals and b in self.terminals:
            potential_parent, potential_child = (a, b) if self.terminals[a] < self.terminals[b] else (b, a)
            existing = parent_update_requests[potential_child][1] if potential_child in parent_update_requests else \
                    self.metric_steiner_tree.node[potential_child]['distance_to_parent']
            if distance < existing:
                parent_update_requests[potential_child] = (potential_parent, distance)


    def add_heavy_node(self, index, new_edges, **node_attrs):
        self.graph.add_node(index, node_attrs)
        for other, weight in new_edges:
            self.graph.add_edge(index, other, weight)

        self._recompute_all_pairs_shortest_paths()  # recompute metric closure from scratch
                                                    # TODO: only need to run one n^2 iteration of F-W


    def add_terminal(self, new_index, new_timestamp=None, just_deleted_root=False):
        if new_timestamp is None:
            new_timestamp = self.n_terminals_ever
            self.n_terminals_ever += 1

        if len(self.terminals) == 0 or just_deleted_root:
            self.metric_steiner_tree.add_node(new_index, parent=None, distance_to_parent=float('inf'))
        else:
            # Get closest possible parent terminal
            min_distance, closest_terminal = self._get_closest_prior_terminal(new_index, new_timestamp)

            if min_distance == float('inf'):
                raise RuntimeError("Terminal must be connected to at least one other terminal")


            # Update metric Steiner tree and lazily add to actual Steiner tree
            self.metric_steiner_tree.add_node(new_index, parent=closest_terminal, 
                distance_to_parent=min_distance)
            self.metric_steiner_tree.add_edge(new_index, closest_terminal)
            self.steiner_tree = None

        # Add new terminal
        self.terminals[new_index] = new_timestamp


    def _get_closest_prior_terminal(self, new_index, new_timestamp):
        terminal_distances = ((self._get_path_weight(new_index, terminal), terminal) 
            for terminal, timestamp in self.terminals.items() if timestamp < new_timestamp)
        return min(terminal_distances, key=lambda x: x[0])


    def remove_terminal_imase_waxman(self, index):
        # Unmark terminal, and only remove from Steiner tree if truly unneeded
        del self.terminals[index]
        if self.metric_steiner_tree.degree(index) == 1:
            self.metric_steiner_tree.remove_node(index)


    def remove_terminal(self, index):
        # Unmark terminal
        cur_timestamp = self.terminals.pop(index)

        # # Prune unneeded leaves from Steiner tree -- not doing this bc we lazily reconstruct Steiner tree
        # self._prune_steiner_tree(index)

        # Connect orphans back into original tree
        neighbors = [i for i in self.metric_steiner_tree[index] if i != index]
        orphans = sorted(neighbors, key=lambda x: self.terminals[x])
        for c, child in enumerate(orphans):
            timestamp = self.terminals[child]
            if timestamp < cur_timestamp:  # parent
                continue
            self.add_terminal(child, timestamp, c == 0)  # if c == 0, the earliest child still isn't the parent,
                                                         # meaning that we've just deleted the root

        self.metric_steiner_tree.remove_node(index)


    def _prune_steiner_tree(self, delete_index):
        # Delete node, then DFS over non-terminal neighbors who now have degree 1
        neighbors = self.steiner_tree.neighbors(delete_index)
        self.steiner_tree.delete_node(delete_index)
        degrees = self.steiner_tree.degree(neighbors)
        for neighbor, degree in zip(neighbors, degrees):
            if degree == 1 and neighbor not in self.terminals:
                self._prune_steiner_tree(neighbor)


    def get_steiner_tree(self):
        if self.steiner_tree is None:
            # Path expansion from scratch
            self.steiner_tree = nx.Graph()
            self.steiner_cost = 0
            edges = []

            for parent, timestamp in self.terminals.items():
                for child in self.metric_steiner_tree[parent]:
                    if child in self.terminals and self.terminals[child] > timestamp:
                        new_path = []
                        self._fill_path(child, parent, new_path)
                        for i in range(len(new_path)):
                            prev_node = new_path[i]
                            next_node = new_path[i + 1] if (i + 1) < len(new_path) else parent 
                            heapq.heappush(edges, (self._get_edge_weight(prev_node, next_node), 
                                                   prev_node, next_node))

            # Kruskal's algorithm to break cycles
            subtrees = UnionFind()
            while edges:
                d, u, v = heapq.heappop(edges)
                if subtrees[u] != subtrees[v]:
                    self.steiner_tree.add_edge(u, v, weight=d)
                    self.steiner_cost += d
                    subtrees.union(u, v)

        return self.steiner_tree, self.steiner_cost

    # Helpers 

    def _get_edge_weight(self, start, end):
        return self.graph[start][end]['weight'] if self.graph.has_edge(start, end) else float('inf')

    def _get_path_weight(self, start, end):
        return self.metric_closure[start][end]['distance'] if self.metric_closure.has_edge(start, end) else float('inf')

    def _fill_path(self, start, end, path):
        if start == end:
            return
        p = self.metric_closure[start][end]['path']
        if len(p) == 0:
            path.append(start)
        else:
            self._fill_path(start, p[0], path)
            self._fill_path(p[0], end, path)

    def _recompute_all_pairs_shortest_paths(self):
        # TODO: at the very least we should do F-W
        self.metric_closure = nx.algorithms.approximation.steinertree.metric_closure(self.graph)
