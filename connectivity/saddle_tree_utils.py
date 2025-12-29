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

def _build_neighbors_dict(dict1):
    '''Builds the dict of keys of dict1 such that their values share a common element.
    '''
    dict2 = {}
    for k2, v2 in dict1.items():
        neighbors = []
        set_v2 = set(v2)  # tidy unique elements and fast lookup
        for k1, v1 in dict1.items():
            if k1 != k2 and set_v2.intersection(v1):
                neighbors.append(k1)
        dict2[k2] = neighbors
    return dict2

def _filter_neighbors_value(dict2, dict_values):
    '''Drops neighbors such that their values does not exceed a given value.
    '''
    dict3 = {}
    for k2, neighbors in dict2.items():
        v2 = dict_values[k2]
        dict3[k2] = [k1 for k1 in neighbors if dict_values[k1] >= v2]
    return dict3

def _filter_neighbors_min_value(dict3, dict_values):
    '''Keep only neighbor(s) with the lowest dict_values among the survivors
    '''
    dict4 = {}
    all_keys = set()

    for k2, neighbors in dict3.items():
        if neighbors:
            min_val = min(dict_values[k1] for k1 in neighbors)
            kept = [k1 for k1 in neighbors if dict_values[k1] == min_val]
            dict4[k2] = kept
        else:
            kept = []
            dict4[k2] = kept
        
        # collect keys
        all_keys.add(k2)
        all_keys.update(kept)
    
    return dict4, all_keys

# def build_extra_edges(minima_to_saddles, saddle_to_minima, saddle_y_values):
#     extra_edges = set()     # tidy unique elements and lookup operations

#     # connect each minimum only to its lowest saddle
#     minima_to_saddles_pruned = dict.fromkeys(minima_to_saddles.keys(),[]) # empty dict with same minima
#     for minimum, saddle_list in minima_to_saddles.items():
#         min_val = min([
#             saddle_y_values[n] for n in saddle_list
#         ])
#         lowest_saddles = [
#             n for n in saddle_list if saddle_y_values[n] == min_val
#         ]
#         minima_to_saddles_pruned[minimum] = lowest_saddles
#         # print('minima_to_saddles_pruned', len(minima_to_saddles_pruned[minimum]) )

#     # for each saddle, find its higher neighbours
#     saddle_neighbors = _build_neighbors_dict(saddle_to_minima)  # find its neighbours (saddles that share a minimum)
#     higher_neighbors = _filter_neighbors_value(saddle_neighbors, saddle_y_values)

#     # find the lowest among its higher neighbours
#     lowest_higher_neighbors, kept_saddles = _filter_neighbors_min_value(higher_neighbors, saddle_y_values)

#     # print('tot_saddles', len(saddle_to_minima.keys()))
#     # print('kept_saddles', len(kept_saddles))
#     # print('lowest_higher_neighbors', lowest_higher_neighbors)

#     # connect
#     all_edges = {**minima_to_saddles_pruned, **lowest_higher_neighbors}
#     all_nodes = list(minima_to_saddles.keys()) + list(kept_saddles)
#     pairs = {(k2, k1) for k2, neighbors in all_edges.items() for k1 in neighbors}
#     for pair in pairs:
#         edge = tuple(sorted(pair))
#         extra_edges.add(edge)

#     return extra_edges, all_nodes

# def build_extra_edges(minima_to_saddles, unique_saddles, saddle_y_values):
#     extra_edges = set()
#     saddle_neighbors = {s: set() for s in unique_saddles}
#     for saddle_list in minima_to_saddles.values():
#         if len(saddle_list) >= 2:
#             max_val = max([
#                 saddle_y_values[n] for n in saddle_list
#             ])
#             highest_saddle = [
#                 n for n in saddle_list if saddle_y_values[n] >= max_val
#             ]
#             for s2 in saddle_list:
#                 if s2!=highest_saddle[0]:
#                     edge = tuple(sorted((highest_saddle[0], s2)))
#                     extra_edges.add(edge)
#                     saddle_neighbors[highest_saddle[0]].add(s2)
#                     saddle_neighbors[s2].add(highest_saddle[0])

#     for s in unique_saddles:
#         s_val = saddle_y_values[s]
#         higher_neighbors = [
#             n for n in saddle_neighbors[s] if saddle_y_values[n] >= s_val
#         ]
#         print('s_val', s_val)
#         print('higher_neighbors', higher_neighbors)
#         if len(higher_neighbors) >= 2:
#             max_val = max([
#                 saddle_y_values[n] for n in higher_neighbors
#             ])
#             highest_neighbour = [
#                 n for n in higher_neighbors if saddle_y_values[n] >= max_val
#             ]
#             for s2 in higher_neighbors:
#                 if s2!=highest_neighbour[0]:
#                     edge = tuple(sorted((highest_neighbour[0], s2)))
#                     extra_edges.add(edge)

#     return extra_edges

def build_extra_edges(minima_to_saddles, unique_saddles, saddle_y_values):
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

# def _clean_extra_edges(minima_to_saddles, unique_saddles, saddle_y_values):

# def build_weight_matrix(all_nodes, extra_edges, saddle_y_values, minima_y_values):
# def build_weight_matrix(unique_saddles, unique_minima, extra_edges, saddle_y_values, minima_y_values):
#     all_nodes = unique_minima + unique_saddles
#     node_idx = {node: i for i, node in enumerate(all_nodes)}
#     f_values = {**saddle_y_values, **minima_y_values}
#     n = len(all_nodes)
#     weight_matrix = np.zeros((n, n))

#     for s1, s2 in extra_edges:
#         i, j = node_idx[s1], node_idx[s2]
#         w = 1.0 #abs(f_values[s1] - f_values[s2])
#         weight_matrix[i, j] = w
#         weight_matrix[j, i] = w

#     return weight_matrix, all_nodes, f_values

def build_weight_matrix(all_nodes, extra_edges, results, saddle_y_values, minima_y_values):
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

def nonzero_indices(arr: np.ndarray) -> np.ndarray:
    return np.argwhere(arr != 0)

def compute_mst(weight_matrix):
    mst_sparse = minimum_spanning_tree(weight_matrix)
    return np.array(mst_sparse.nonzero()).T

# def dfs_layout(tree_adj, node, unique_minima, f_values, visited, x_pos, counter, parent=None):
#     visited.add(node)
#     # print(visited)
#     children = [n for n in tree_adj[node] if n != parent]
#     if node in unique_minima:
#         x_pos[node] = counter[0]
#         counter[0] += 1
#         return x_pos[node]
#     child_xs = [dfs_layout(tree_adj, child, unique_minima, f_values, visited, x_pos, counter, node) for child in children]
#     x_pos[node] = sum(child_xs) / len(child_xs) if child_xs else counter[0]
#     return x_pos[node]

# def bfs_layout(tree_adj, node, unique_minima, f_values, visited, x_pos, counter, parent=None):
#     visited.add(node)
#     # print(visited)
#     children = [n for n in tree_adj[node] if n != parent]
#     if node in unique_minima:
#         x_pos[node] = counter[0]
#         counter[0] += 1
#         return x_pos[node]
#     child_xs = [bfs_layout(tree_adj, child, unique_minima, f_values, visited, x_pos, counter, node) for child in children]
#     x_pos[node] = sum(child_xs) / len(child_xs) if child_xs else counter[0]
#     return x_pos[node]

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

def dfs_order(mst_edges, node_count, root_id, all_values, ids):
    tree = defaultdict(set)
    for u, v in mst_edges:
        tree[u].add(v)
        tree[v].add(u)
    
    for u in tree.keys():
        tree[u] = sorted(tree[u], key=lambda n: all_values[n])

    visited = [False] * node_count
    order = []

    def dfs(node):
        visited[node] = True
        order.append(node)
        for neighbor in tree[node]:
            if not visited[neighbor]:
                dfs(neighbor)

    root_pos = ids.index(root_id)
    node = root_pos
    dfs(node)       # start from first node

    return order    # list of node indices in DFS order

def bfs_order(mst_edges, node_count, root_id, all_values, ids):
    tree = defaultdict(set)
    for u, v in mst_edges:
        tree[u].add(v)
        tree[v].add(u)
    
    for u in tree.keys():
        tree[u] = sorted(tree[u], key=lambda n: all_values[n])

    visited = [False] * node_count
    appended = [False] * node_count
    order = []

    def bfs(node):
        order.append(node)
        visited[node] = True
        appended[node] = True
        for neighbor in tree[node]:
            if not appended[neighbor]:
                order.append(neighbor)
        for neighbor in tree[node]:
                bfs(neighbor)

    root_pos = ids.index(root_id)
    node = root_pos
    bfs(node)       # start from first node

    return order    # list of node indices in DFS order

def map_coords_to_indices(critical_points_df):
    x_cols = sorted([col for col in critical_points_df.columns if col.startswith("x")])

    coord_to_index = {
        tuple(row[col] for col in x_cols): idx
        for idx, row in critical_points_df.iterrows()
    }
    coord_to_value = {
        tuple(row[col] for col in x_cols): row["f_value"]
        for idx, row in critical_points_df.iterrows()
    }
    coord_to_type = {
    tuple(row[col] for col in x_cols): row["type"]
    for idx, row in critical_points_df.iterrows()
    }

    all_nodes = [
    tuple(row[col] for col in x_cols) for idx, row in critical_points_df.iterrows()
    ]

    index_map = []
    all_values = []
    all_types = []

    for coord in all_nodes:
        rounded_coord = tuple(c for c in coord)
        index = coord_to_index.get(rounded_coord)
        value = coord_to_value.get(rounded_coord)
        typee = coord_to_type.get(rounded_coord)
        index_map.append(index)
        all_values.append(value)
        all_types.append(typee)

    return all_nodes, all_values, all_types, index_map

def save_tree_nodes_csv(
    index_map, all_values, dfs_order_list, out_dir='./'
):
    output_file=out_dir+"tree_nodes.csv"
    rows = []
    for order_idx, internal_idx in enumerate(dfs_order_list):
        f_val = all_values[internal_idx]
        rows.append((index_map[internal_idx], f_val, order_idx))

    df = pd.DataFrame(rows, columns=["index", "f_value", "order"])
    df.to_csv(output_file, index=False)
    print(f"Saved node order to {output_file}")


def save_tree_edges_csv(mst_edges, ids, out_dir='./'):
    output_file=out_dir+"tree_connectivity.csv"
    rows = [(ids[i], ids[j]) for i, j in mst_edges]
    df = pd.DataFrame(rows, columns=["index_1", "index_2"])
    df.to_csv(output_file, index=False)
    print(f"Saved tree edges to {output_file}")


import collections
def prune_graph(values, ids, edges, types, root_id):
    
    n = len(values)
    type1 = "minimum"
    type2 = "saddle"

    def build_adj(edges):
        adj = collections.defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    # Check the type2 node with max value
    root_pos = ids.index(root_id)
    # print(root_id, root_pos)
    assert values[root_pos] == max(values)
    max_type2 = root_id
    
    keep = set(range(n))  # start with all nodes
    edges = set(tuple(sorted(e)) for e in edges)  # deduplicate edges
    adj = build_adj(edges)
    
    changed = True
    while changed:
        changed = False
        
        for i in list(keep):
            node_type = types[i]
            neighbors = [j for j in adj[i] if j in keep]
            
            if node_type == type1:
                continue  # always keep minima
            
            if ids[i] == max_type2:
                continue  # always keep the root node
            
            # Rule 1: remove saddles that have at most 2 neighbours, of which at least one is 
            # another saddle
            if len(neighbors) <= 2 and any(types[j] == type2 for j in neighbors):
                keep.remove(i)
                changed = True
                # print('removed', i)
                # if i==18: print('removed', i)
                # if i==18: print('neighbors', neighbors)
                
                # Rewire if exactly 2 neighbors
                if len(neighbors) == 2:
                    u, v = neighbors
                    edges.add(tuple(sorted((u, v))))
                    # print('rewired', neighbors)
                
                # Remove edges touching this node
                edges = {e for e in edges if i not in e}
                
                # Update adjacency
                adj = build_adj(edges)
                continue

            # Rule 2 (weak): remove saddles that have one or more equal-value neighbours and 
            # rewire to the equal-value neighbour (gets rid of equal value saddles)
            equal_neighbors = [j for j in neighbors if values[j] == values[i]]
            if len(equal_neighbors) >= 1:
                keep.remove(i) # remove node
                changed = True
                
                # Connect all neighbors to the first equal-value neighbor
                best_neighbor = equal_neighbors[0]
                for nb in neighbors:
                    if nb != best_neighbor:
                        edges.add(tuple(sorted((best_neighbor, nb))))
                
                edges = {e for e in edges if i not in e}
                adj = build_adj(edges)
                continue
            
            # DO NOT USE, DOES NOT GIVE CORRECT TOPOLOGY:
            # # Rule 2 (strong): remove saddles that have more than one higher- or equal-value 
            # # neighbour and rewire the neighbours to the highest neighbour 
            # # (gets rid of equal value saddles AND saddles that only have saddle children) 
            # higher_neighbors = [j for j in neighbors if values[j] >= values[i]]
            # if len(higher_neighbors) > 1:
            #     keep.remove(i)  # remove node
            #     changed = True
                
            #     # Connect all neighbors to the highest-value neighbor
            #     best_neighbor = max(higher_neighbors, key=lambda j: values[j])
            #     for nb in neighbors:
            #         if nb != best_neighbor:
            #             edges.add(tuple(sorted((best_neighbor, nb))))
                
            #     edges = {e for e in edges if i not in e}
            #     adj = build_adj(edges)
            #     continue
    
    pruned_values = [values[i] for i in keep]
    pruned_types = [types[i] for i in keep]
    pruned_edges = np.array([[u, v] for (u, v) in edges if u in keep and v in keep])
    
    return pruned_values, pruned_edges, pruned_types