from saddle_tree_utils import (
    get_unique_nodes,
    get_function_values,
    build_saddle_minima_mapping,
    build_extra_edges,
    build_weight_matrix,
    compute_mst,
    dfs_layout,
    spread_x_positions,
    dfs_order,
    map_coords_to_indices,
    save_tree_edges_csv,
    save_tree_nodes_csv,
)
from collections import defaultdict

def saddle_tree_calculations(results, minima, dataframe):
    unique_saddles, unique_minima = get_unique_nodes(results, minima)
    saddle_y_values, minima_y_values = get_function_values(results, minima)
    # Build mapping from minima to saddles
    minima_to_saddles, saddle_to_minima = build_saddle_minima_mapping(
        results, unique_minima, unique_saddles
    )

    # STEP 1: Connect saddles that share the same minimum
    # STEP 2: For each saddle, connect its neighbors that have >= function value
    extra_edges = build_extra_edges(minima_to_saddles, saddle_y_values)
   
    
     # --- Step 3: Build connectivity matrix and compute MST ---
    weight_matrix, all_nodes, f_values = build_weight_matrix(
        unique_saddles, unique_minima, extra_edges, results, saddle_y_values, minima_y_values
    )

    mst_edges = compute_mst(weight_matrix)

    # Step 5: DFS-based tidy layout
    tree_adj = defaultdict(list)
    for i, j in mst_edges:
        n1, n2 = all_nodes[i], all_nodes[j]
        tree_adj[n1].append(n2)
        tree_adj[n2].append(n1)

    root = max(unique_saddles, key=lambda s: f_values[s])
    x_pos = {}
    visited = set()

    # dfs_layout(tree_adj, root, unique_minima, f_values, visited, x_pos, [1])
    
    x_pos = spread_x_positions(x_pos)
  
    all_nodes, index_map = map_coords_to_indices(unique_saddles, unique_minima, dataframe)

    dfs_order_list = dfs_order(mst_edges, len(all_nodes), root, dataframe)
  
    save_tree_nodes_csv(all_nodes, index_map, saddle_y_values, minima_y_values, dfs_order_list)
    save_tree_edges_csv(mst_edges)
