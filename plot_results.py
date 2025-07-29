import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
from scipy.sparse.csgraph import minimum_spanning_tree

def plot_results_with_paths(func, minima=None, bounds=(0, 1),
                            saddle_points=None, paths=None, res=100,
                            show_connectivity=True):
    """
    Visualize function landscape with minima, saddle points, and descent paths
    with consistent labels S0/S1.. for saddle points and M0/M1.. for minima.
    """

    # Set up grid and evaluate function
    plt.figure(figsize=(12, 6))
    x = np.linspace(bounds[0], bounds[1], res)
    y = np.linspace(bounds[0], bounds[1], res)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.array([[func((x, y)) for x, y in zip(row_x, row_y)]
                  for row_x, row_y in zip(X, Y)])

    cmap = cm.inferno

    # 3D Plot
    ax1 = plt.subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8,
                            antialiased=True, linewidth=0)
    plt.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Function Value')

    # Plot minima
    if minima:
        minima_points = np.array([m[0] for m in minima])
        minima_values = [func(p) for p in minima_points]
        ax1.scatter(minima_points[:, 0], minima_points[:, 1], minima_values,
                    c='red', s=50, depthshade=True, label=f'Minima (n={len(minima_points)})')

        # Label them
        for idx, (point, val) in enumerate(zip(minima_points, minima_values)):
            ax1.text(point[0], point[1], val + 0.02, f"M{idx}", color="white", fontsize=8)

    # Plot saddle points
    if saddle_points:
        saddle_points_array = np.array([sp[0] for sp in saddle_points])
        saddle_values = [func(p) for p in saddle_points_array]
        ax1.scatter(saddle_points_array[:, 0], saddle_points_array[:, 1], saddle_values,
                    c='green', s=50, depthshade=True, label=f'Saddles (n={len(saddle_points_array)})')

        # Label them
        for idx, (point, val) in enumerate(zip(saddle_points_array, saddle_values)):
            ax1.text(point[0], point[1], val + 0.02, f"S{idx}", color="white", fontsize=8)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title("3D Surface with Labels")
    ax1.legend()

    # 2D Contour plot
    ax2 = plt.subplot(1, 2, 2)
    contour = ax2.contourf(X, Y, Z, levels=20, cmap=cmap)
    plt.colorbar(contour, ax=ax2, label='Function Value')

    # Plot minima with labels
    if minima:
        ax2.scatter(minima_points[:, 0], minima_points[:, 1],
                    c='red', s=80, edgecolor='k', linewidth=1.5,
                    label=f'Minima (n={len(minima_points)})')
        for idx, point in enumerate(minima_points):
            ax2.text(point[0] + 0.01, point[1] + 0.01, f"M{idx}", fontsize=8, color="white")

    # Plot saddle points with labels
    if saddle_points:
        ax2.scatter(saddle_points_array[:, 0], saddle_points_array[:, 1],
                    c='green', s=80, edgecolor='k', linewidth=1.5,
                    label=f'Saddles (n={len(saddle_points_array)})')
        for idx, point in enumerate(saddle_points_array):
            ax2.text(point[0] + 0.01, point[1] + 0.01, f"S{idx}", fontsize=8, color="white")

    # Plot descent paths
    if paths and show_connectivity:
        for path in paths:
            if "trajectories" in path:
                for traj in path["trajectories"]:
                    traj_arr = np.array(traj)
                    ax2.plot(traj_arr[:, 0], traj_arr[:, 1], color='blue', lw=1)
            else:
                # fallback to arrows if no full trajectory
                saddle = path['saddle']
                for m in path['minimizers']:
                    ax2.annotate('', xy=m, xytext=saddle,
                                arrowprops=dict(arrowstyle='->', color='blue'))

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('2D Contour with Labels and Descent Paths')
    ax2.legend()

    plt.tight_layout()
    plt.show()

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

def plot_saddle_tree_with_function(results, global_minima):
    

    # Step 0: Prepare data
    unique_saddles = []
    for r in results:
        s = tuple(r['saddle_point'])
        if s not in unique_saddles:
            unique_saddles.append(s)

    unique_minima = [tuple(m[0]) for m in global_minima]
    saddle_idx = {s: i for i, s in enumerate(unique_saddles)}
    saddle_x_positions = {s: i*3 for i, s in enumerate(unique_saddles)}
    minima_x_positions = {m: i*3 for i, m in enumerate(unique_minima)}


    saddle_y_values = {}
    minima_y_values = {}

    for r in results:
        s = tuple(r['saddle_point'])
        m = tuple(r['minimizer'])
        saddle_y_values[s] = r['saddle_value']

    for m in global_minima:
        coord = tuple(m[0])
        val = m[1]
        minima_y_values[coord] = val

    # Build mapping from minima to saddles
    minima_to_saddles = {m: [] for m in unique_minima}
    saddle_to_minima = {s: [] for s in unique_saddles}
    node_positions = {**saddle_x_positions, **minima_x_positions}
    node_heights = {**saddle_y_values, **minima_y_values}
    for r in results:
        m = tuple(r['minimizer'])
        s = tuple(r['saddle_point'])
        minima_to_saddles[m].append(s)
        saddle_to_minima[s].append(m)

    # STEP 1: Connect saddles that share the same minimum
    extra_edges = set()
    saddle_neighbors = {s: set() for s in unique_saddles}
    for saddle_list in minima_to_saddles.values():
        if len(saddle_list) >= 2:
            for s1, s2 in itertools.combinations(saddle_list, 2):
                edge = tuple(sorted((s1, s2)))
                extra_edges.add(edge)
                saddle_neighbors[s1].add(s2)
                saddle_neighbors[s2].add(s1)

    # STEP 2: For each saddle, connect its neighbors that have >= function value
    for s in unique_saddles:
        s_val = saddle_y_values[s]
        higher_neighbors = [
            n for n in saddle_neighbors[s] if saddle_y_values[n] >= s_val
        ]
        if len(higher_neighbors) >= 2:
            for s1, s2 in itertools.combinations(higher_neighbors, 2):
                edge = tuple(sorted((s1, s2)))
                extra_edges.add(edge)

     # --- Step 3: Build connectivity matrix and compute MST ---
    all_nodes = unique_saddles + unique_minima
    node_idx = {node: i for i, node in enumerate(all_nodes)}
    f_values = {**saddle_y_values, **minima_y_values}
    n = len(all_nodes)
    weight_matrix = np.zeros((n, n))
    for s1, s2 in extra_edges:
        i, j = node_idx[s1], node_idx[s2]
        w = abs(f_values[s1] - f_values[s2])
        weight_matrix[i, j] = w
        weight_matrix[j, i] = w  # undirected

    # Saddle–Minima edges (from original graph)
    for r in results:
        s = tuple(r['saddle_point'])
        m = tuple(r['minimizer'])
        i, j = node_idx[s], node_idx[m]
        w = abs(f_values[s] - f_values[m])
        weight_matrix[i, j] = w
        weight_matrix[j, i] = w  # undirected

    mst_sparse = minimum_spanning_tree(weight_matrix)
    mst_edges = np.array(mst_sparse.nonzero()).T    

    # Step 5: DFS-based tidy layout
    tree_adj = defaultdict(list)
    for i, j in mst_edges:
        n1, n2 = all_nodes[i], all_nodes[j]
        tree_adj[n1].append(n2)
        tree_adj[n2].append(n1)

    root = max(unique_saddles, key=lambda s: f_values[s])
    x_pos = {}
    visited = set()
    counter = [1]

    # Custom DFS recursive function for creating the layout of spanning tree
    def dfs_layout(node, parent=None):
        visited.add(node)
        children = [n for n in tree_adj[node] if n != parent]
        if node in unique_minima:
            x_pos[node] = counter[0]
            counter[0] += 1
            return x_pos[node]
        child_xs = [dfs_layout(child, node) for child in children]
        x_pos[node] = sum(child_xs) / len(child_xs) if child_xs else counter[0]
        return x_pos[node]

    dfs_layout(root)
    
    x_pos = spread_x_positions(x_pos)
    node_positions = {node: x_pos[node] * 2 for node in all_nodes}

    node_heights = {**saddle_y_values, **minima_y_values}

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 8))

    for edge in mst_edges:
        i, j = edge
        n1, n2 = all_nodes[i], all_nodes[j]
        x1, y1 = node_positions[n1], node_heights[n1]
        x2, y2 = node_positions[n2], node_heights[n2]
        ax.plot([x1, x2], [y1, y2], color="red", lw=2)

    for idx, s in enumerate(unique_saddles):
        x, y = node_positions[s], saddle_y_values[s]
        ax.scatter(x, y, color="orange", edgecolor="black", s=60, zorder=3)
        ax.text(x, y + 0.025, f"S{idx}", ha="center", fontsize=8)

    for idx, m in enumerate(unique_minima):
        x, y = node_positions[m], minima_y_values[m]
        ax.scatter(x, y, color="skyblue", edgecolor="black", s=60, zorder=3)
        ax.text(x, y - 0.03, f"M{idx}", ha="center", fontsize=8)

    
    # Original saddle-minima edges
    # for r in results:
    #     s = tuple(r['saddle_point'])
    #     m = tuple(r['minimizer'])
    #     x1 = saddle_x_positions[s]
    #     y1 = saddle_y_values[s]
    #     x2 = minima_x_positions[m]
    #     y2 = minima_y_values[m]
    #     ax.plot([x1, x2], [y1, y2], color="gray", lw=1)

    # Extra saddle-saddle edges (Steps 1 + 2)
    # for s1, s2 in extra_edges:
    #     x1, y1 = saddle_x_positions[s1], saddle_y_values[s1]
    #     x2, y2 = saddle_x_positions[s2], saddle_y_values[s2]
    #     ax.plot([x1, x2], [y1, y2], color="red", lw=1, linestyle="--")
    # for edge in mst_edges:
    #     i, j = edge
    #     s1 = unique_saddles[i]
    #     s2 = unique_saddles[j]
    #     x1, y1 = saddle_x_positions[s1], saddle_y_values[s1]
    #     x2, y2 = saddle_x_positions[s2], saddle_y_values[s2]
    #     ax.plot([x1, x2], [y1, y2], color="red", lw=2)

    ax.set_xlabel("Tree-ordered x-position (DFS)")
    ax.set_ylabel("Function Value f(x)")
    ax.set_title("Saddle-Minima MST Tree")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.yaxis.set_major_locator(MaxNLocator(10))
    plt.tight_layout()
    plt.show()