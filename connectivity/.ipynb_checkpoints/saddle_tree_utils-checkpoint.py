import numpy as np
import itertools
from collections import defaultdict
from scipy.sparse.csgraph import minimum_spanning_tree
import pandas as pd
saddle_x_positions = {}
minima_x_positions = {}
saddle_y_values = {}
minima_y_values = {}
def get_unique_nodes(results, global_minima):
    unique_saddles = []
    for r in results:
        s = tuple(r['saddle_point'])
        if s not in unique_saddles:
            unique_saddles.append(s)

    unique_minima = [tuple(m[0]) for m in global_minima]
    return unique_saddles, unique_minima

def get_function_values(results, global_minima):

    for r in results:
        s = tuple(r['saddle_point'])
        saddle_y_values[s] = r['saddle_value']
    for m in global_minima:
        coord = tuple(m[0])
        val = m[1]
        minima_y_values[coord] = val
    return saddle_y_values, minima_y_values

def build_saddle_minima_mapping(results, unique_minima, unique_saddles):
    minima_to_saddles = {m: [] for m in unique_minima}
    saddle_to_minima = {s: [] for s in unique_saddles}
    for r in results:
        m = tuple(r['minimizer'])
        s = tuple(r['saddle_point'])
        minima_to_saddles[m].append(s)
        saddle_to_minima[s].append(m)
    return minima_to_saddles, saddle_to_minima

def build_extra_edges(minima_to_saddles, unique_saddles):
    extra_edges = set()
    saddle_neighbors = {s: set() for s in unique_saddles}
    for saddle_list in minima_to_saddles.values():
        if len(saddle_list) >= 2:
            for s1, s2 in itertools.combinations(saddle_list, 2):
                edge = tuple(sorted((s1, s2)))
                extra_edges.add(edge)
                saddle_neighbors[s1].add(s2)
                saddle_neighbors[s2].add(s1)

    for s in unique_saddles:
        s_val = saddle_y_values[s]
        higher_neighbors = [
            n for n in saddle_neighbors[s] if saddle_y_values[n] >= s_val
        ]
        if len(higher_neighbors) >= 2:
            for s1, s2 in itertools.combinations(higher_neighbors, 2):
                edge = tuple(sorted((s1, s2)))
                extra_edges.add(edge)
    return extra_edges

def build_weight_matrix(unique_saddles, unique_minima, extra_edges, results, saddle_y_values, minima_y_values):
    all_nodes = unique_minima + unique_saddles
    node_idx = {node: i for i, node in enumerate(all_nodes)}
    f_values = {**saddle_y_values, **minima_y_values}
    n = len(all_nodes)
    weight_matrix = np.zeros((n, n))

    for s1, s2 in extra_edges:
        i, j = node_idx[s1], node_idx[s2]
        w = abs(f_values[s1] - f_values[s2])
        weight_matrix[i, j] = w
        weight_matrix[j, i] = w

    for r in results:
        s = tuple(r['saddle_point'])
        m = tuple(r['minimizer'])
        i, j = node_idx[s], node_idx[m]
        w = abs(f_values[s] - f_values[m])
        weight_matrix[i, j] = w
        weight_matrix[j, i] = w

    return weight_matrix, all_nodes, f_values


def compute_mst(weight_matrix):
    mst_sparse = minimum_spanning_tree(weight_matrix)
    return np.array(mst_sparse.nonzero()).T

def dfs_layout(tree_adj, node, unique_minima, f_values, visited, x_pos, counter, parent=None):
    visited.add(node)
    # print(visited)
    children = [n for n in tree_adj[node] if n != parent]
    if node in unique_minima:
        x_pos[node] = counter[0]
        counter[0] += 1
        return x_pos[node]
    child_xs = [dfs_layout(tree_adj, child, unique_minima, f_values, visited, x_pos, counter, node) for child in children]
    x_pos[node] = sum(child_xs) / len(child_xs) if child_xs else counter[0]
    return x_pos[node]


def spread_x_positions(x_pos, min_sep=0.3):
    # Prevent exact overlaps by nudging x positions
    used_x = {}
    for node, x in sorted(x_pos.items(), key=lambda item: item[1]):
        rounded = round(x, 4)
        if rounded in used_x:
            count = used_x[rounded]
            x_pos[node] += count * min_sep
            used_x[rounded] += 1
        else:
            used_x[rounded] = 1
    return x_pos

def dfs_order(mst_edges, node_count, node, critical_points_df):
    tree = defaultdict(list)
    for u, v in mst_edges:
        tree[u].append(v)
        tree[v].append(u)

    visited = [False] * node_count
    order = []

    def dfs(node):
        visited[node] = True
        order.append(node)
        for neighbor in tree[node]:
            if not visited[neighbor]:
                dfs(neighbor)

    x_cols = sorted([col for col in critical_points_df.columns if col.startswith("x")])

    coord_to_index = {
        tuple(row[col] for col in x_cols): idx
        for idx, row in critical_points_df.iterrows()
    }
    for value, idx in coord_to_index.items():
        if value == node:
            node = idx
    dfs(node)  # start from first node
    return order  # list of node indices in DFS order

def map_coords_to_indices(unique_saddles, unique_minima, critical_points_df):
    x_cols = sorted([col for col in critical_points_df.columns if col.startswith("x")])

    coord_to_index = {
        tuple(round(row[col], 8) for col in x_cols): idx
        for idx, row in critical_points_df.iterrows()
    }
    all_nodes = unique_minima + unique_saddles
    index_map = []

    for coord in all_nodes:
        rounded_coord = tuple(round(c, 8) for c in coord)
        index = coord_to_index.get(rounded_coord)
        index_map.append(index)
    
    return all_nodes, index_map

def save_tree_nodes_csv(
    all_nodes, index_map, saddle_y_values:dict, minima_y_values:dict, dfs_order_list, output_file="results/tree_nodes.csv"
):
    rows = []
    for order_idx, internal_idx in enumerate(dfs_order_list):
        coord = all_nodes[internal_idx]
        rounded_coord = tuple(round(c, 8) for c in coord)
        f_val = saddle_y_values.get(coord, minima_y_values.get(coord))
        rows.append((index_map[internal_idx], f_val, order_idx))

    df = pd.DataFrame(rows, columns=["index", "f_value", "order"])
    df.to_csv(output_file, index=False)
    print(f"Saved node order to {output_file}")


def save_tree_edges_csv(mst_edges, output_file="results/tree_connectivity.csv"):
    rows = [(i, j) for i, j in mst_edges]
    df = pd.DataFrame(rows, columns=["index_1", "index_2"])
    df.to_csv(output_file, index=False)
    print(f"Saved tree edges to {output_file}")
