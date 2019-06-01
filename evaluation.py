import algorithm
import click
import random
import time
from multiprocessing import Pool
import numpy as np
from itertools import product

initial_list = [300]  # number of initial nodes
connectivity_list = [0.025, 0.05, 0.1, 0.20]
nops_list = [500]
initial_terminals_list = [40]
fquery_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test(initial, connectivity, nops, initial_terminals, fquery):
    # Create graphs
    baseline = algorithm.Baseline()
    ours = algorithm.VDSTGraph()
    baseline_stats = []
    our_stats = []

    # Seed with some initial data
    for i in range(initial):
        baseline.add_node(i)
        ours.add_node(i)
        if i == 0:
            continue
        for j in range(max(1, int(connectivity * i))):
            other = np.random.randint(i)
            weight = np.random.uniform(0, 100)

            baseline.decrease_edge_weight(i, other, weight)
            ours.decrease_edge_weight(i, other, weight)

    # Compile operation list
    ops = []
    terminals = set()
    n_nodes = initial
    for i in range(nops):
        if len(terminals) < initial_terminals:
            idx = np.random.randint(n_nodes)
            if len(terminals) > 0:
                other = np.random.choice(list(terminals))
                weight = np.random.uniform(100, 200)
                ops.append(lambda ds, idx=idx, other=other, weight=weight: ds.decrease_edge_weight(idx, other, weight))
            ops.append(lambda ds, idx=idx: ds.add_terminal(idx))
            terminals.add(idx)
            continue

        r = np.random.random()
        if r < fquery:
            ops.append(lambda ds: (baseline_stats if ds.is_baseline else our_stats).append(ds.get_steiner_tree()[1]))
        else:
            r = int(12 * (r - fquery) / (1 - fquery))
            if r < 3:
                # Add a node and edge
                ops.append(lambda ds, n_nodes=n_nodes: ds.add_node(n_nodes))
                other = np.random.randint(n_nodes)
                weight = np.random.uniform(0, 100)
                ops.append(lambda ds, n_nodes=n_nodes, other=other, weight=weight: ds.decrease_edge_weight(n_nodes, other, weight))
                n_nodes += 1
            elif r < 6:
                a = np.random.randint(0, initial - 10)
                b = np.random.randint(initial - 10, n_nodes)
                weight = np.random.uniform(0, 100)
                ops.append(lambda ds, a=a, b=b, weight=weight: ds.decrease_edge_weight(a, b, weight))
            elif r < 9:
                idx = np.random.randint(n_nodes)
                other = np.random.choice(list(terminals))
                weight = np.random.uniform(100, 200)
                ops.append(lambda ds, idx=idx, other=other, weight=weight: ds.decrease_edge_weight(idx, other, weight))
                ops.append(lambda ds, idx=idx: ds.add_terminal(idx))
                terminals.add(idx)
            else:
                idx = np.random.choice(list(terminals))
                ops.append(lambda ds, idx=idx: ds.remove_terminal(idx))
                terminals.remove(idx)

    # Execute operation list -- baseline
    print('Executing baseline')
    start = time.perf_counter()
    for op in ops:
        op(baseline)
    baseline_time = time.perf_counter() - start
    print('-> took {} seconds'.format(baseline_time))

    # Execute operation list -- our method
    print('Executing our method')
    start = time.perf_counter()
    for op in ops:
        op(ours)
    our_time = time.perf_counter() - start
    print('-> took {} seconds'.format(our_time))

    # Write to file
    with open('test_{}_{}_{}_{}_{}.txt'.format(initial, connectivity, nops, initial_terminals, fquery), 'w') as f:
        f.write('{}, {}\n'.format(our_time, baseline_time))
        for our_perf, baseline_perf in zip(our_stats, baseline_stats):
            f.write('{}, {}\n'.format(our_perf, baseline_perf))

    print('Finished')


def main():
    p = Pool(60)
    p.starmap(test, product(initial_list, connectivity_list, nops_list, initial_terminals_list, fquery_list))


def test1():
    graph = algorithm.Graph()
    # run tests here
    for i in range(5):
        graph.add_node(i)

    for i in range(4):
        graph.decrease_edge_weight(i, i+1, 1.)

    graph.add_terminal(0)
    graph.add_terminal(2)
    graph.add_terminal(4)

    steiner_tree, _ = graph.get_steiner_tree()

    for edge in steiner_tree.edges():
        print(edge)

def test2():
    graph = algorithm.Graph()
    # run tests here
    for i in range(10):
        graph.add_node(i)

    for i in range(8):
        graph.decrease_edge_weight(i, i+1, 1)
        graph.decrease_edge_weight(i, i+2, 1)

    graph.add_terminal(3)
    graph.add_terminal(8)

    steiner_tree, _ = graph.get_steiner_tree()

    for edge in steiner_tree.edges():
        print(edge)


if __name__ == '__main__':
    main()