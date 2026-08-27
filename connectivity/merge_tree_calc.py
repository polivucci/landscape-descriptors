from connectivity.saddle_tree_utils import (
    get_unique_nodes,
    get_function_values,
    build_saddle_minima_mapping,
    build_extra_edges,
    build_weight_matrix,
    nonzero_indices,
    compute_mst,
    spread_x_positions,
    prune_graph,
    dfs_order,
    bfs_order,
    map_coords_to_indices,
    get_tree_edges,
    get_tree_nodes,
)
from collections import defaultdict
from numpy import unique
from pandas import DataFrame

from connectivity.saddle_tree_utils import map_coords_to_indices
from connectivity.merge_tree import MergeTree, neighbours_from_adjacency


def merge_tree(critical_points_df, connectivity_df):

    dataframe = critical_points_df
    connected_nodes = unique(connectivity_df.to_numpy())
    connected_mask = dataframe.index.isin(connected_nodes)
    dataframe = dataframe[connected_mask]

    # sort CPs in descending order of function value
    sorted_cps = dataframe.sort_values('f_value', ascending=False).reset_index() 
    # type of each sorted node
    _, _, all_types, _ = map_coords_to_indices(sorted_cps)

    # map from sorted to unsorted (and its inverse)
    unsorted_sorted=sorted_cps['index'].reset_index().set_index('index')['level_0'].to_dict()
    sorted_unsorted=sorted_cps['index'].to_dict()
    input_sorted_vertices = sorted_cps.index.to_list()
    input_critical_points_map = sorted_cps['type']

    # adjacency list using new sorted node indexing
    input_edge_list = connectivity_df.to_numpy().tolist()
    input_edge_list = [[unsorted_sorted[edge[0]], unsorted_sorted[edge[1]]] for edge in input_edge_list]

    # build neighbour map for every node
    input_neighbor_map = neighbours_from_adjacency(input_edge_list, input_sorted_vertices)

    # initialise and solve merge tree
    merge_tree = MergeTree(
        sorted_vertices=input_sorted_vertices,
        neighbor_map=input_neighbor_map,
        critical_points_map=input_critical_points_map
    )

    merge_tree.compute_join_tree()

    # write tree as adjacency list
    tree_adjacency_list = merge_tree.tree_adjacency_list()
    all_values = sorted_cps['f_value']
    root_id = sorted_cps.index[0]

    # simplify tree by node pruning:
    _, tree_adjacency_list, _ = prune_graph(all_values, input_sorted_vertices, tree_adjacency_list, all_types, root_id)

    # compute DFS layout for visualisation
    dfs_order_list = dfs_order(tree_adjacency_list, len(all_values), root_id, all_values, input_sorted_vertices)

    # write out tree connectivity
    # get back to original unsorted node names:
    tree_adjacency_list = [[sorted_unsorted[n1], sorted_unsorted[n2]] for (n1, n2) in tree_adjacency_list]
    tree_adjacency_df = DataFrame(tree_adjacency_list, columns=['index_1', 'index_2'])

    # write out tree nodes
    # get back to original unsorted node names:
    tree_nodes_df = get_tree_nodes(sorted_unsorted, all_values, all_types, dfs_order_list)

    return tree_nodes_df, tree_adjacency_df
