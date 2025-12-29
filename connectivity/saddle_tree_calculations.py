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
    save_tree_edges_csv,
    save_tree_nodes_csv,
)
from collections import defaultdict

def saddle_tree_calculations(results, minima, dataframe, out_dir='./'):

    unique_saddles, unique_minima = get_unique_nodes(results, minima)
    saddle_y_values, minima_y_values = get_function_values(results, minima)
    
    all_nodes, all_values, all_types, index_map = map_coords_to_indices(dataframe)
    print('index_map', index_map)

    # Build mapping from minima to saddles
    minima_to_saddles, saddle_to_minima = build_saddle_minima_mapping(
        results, unique_minima, unique_saddles
    )

    # STEP 1: Connect saddles that share the same minimum
    # STEP 2: For each saddle, connect its neighbors that have >= function value
    extra_edges = build_extra_edges(minima_to_saddles, unique_saddles, saddle_y_values)
    # extra_edges, all_nodes = build_extra_edges(minima_to_saddles, saddle_to_minima, saddle_y_values)
    # print('all_nodes', all_nodes)
     # --- Step 3: Build connectivity matrix and compute MST ---
    weight_matrix, all_nodes, f_values = build_weight_matrix(
        all_nodes, extra_edges, results, saddle_y_values, minima_y_values
    )
    # weight_matrix, all_nodes, f_values = build_weight_matrix(
    #     unique_saddles, unique_minima, extra_edges, saddle_y_values, minima_y_values
    # )
    # print('all_nodes', all_nodes)

    # print(weight_matrix.shape)

    mst_edges = compute_mst(weight_matrix)
    # mst_edges = nonzero_indices(weight_matrix)
    # print('mst_edges', mst_edges.shape)

     # --- Step 4: prune MST
    root = max(unique_saddles, key=lambda s: saddle_y_values[s]) # highest saddle
    root_id = index_map[all_nodes.index(root)]
    # print('root_id', root_id)

    # print('root', root_id, root)
    # print('index_map', index_map)
    pruned_values, pruned_edges, pruned_types = prune_graph(all_values, index_map, mst_edges, all_types, root_id)
    # pruned_edges = mst_edges
    # print('pruned_edges', pruned_edges)

    # Step 5: DFS-based tidy layout
    # print('all_values', all_values)
    dfs_order_list = dfs_order(pruned_edges, len(all_nodes), root_id, all_values, index_map)
    # dfs_order_list = reversed(list(range(len(all_nodes))))
    # print('dfs_order_list', dfs_order_list)

    save_tree_nodes_csv(index_map, all_values, dfs_order_list, out_dir=out_dir)
    save_tree_edges_csv(pruned_edges, index_map, out_dir=out_dir)
