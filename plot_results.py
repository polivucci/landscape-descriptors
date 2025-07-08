import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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


def plot_saddle_tree_with_function(results, global_minima):
    """
    Visualize saddle-minima connectivity as a tree structure,
    using consistent minima IDs from global_minima.
    """

    # collect unique saddles
    unique_saddles = []
    for r in results:
        s = tuple(r['saddle_point'])
        if s not in unique_saddles:
            unique_saddles.append(s)

    # use consistent minima order from global_minima
    unique_minima = [tuple(m[0]) for m in global_minima]

    # assign x positions equally spaced
    saddle_x_positions = {s: i*3 for i,s in enumerate(unique_saddles)}
    minima_x_positions  = {m: i*3 for i,m in enumerate(unique_minima)}

    # store their function values
    saddle_y_values = {}
    minima_y_values = {}

    for r in results:
        s = tuple(r['saddle_point'])
        m = tuple(r['minimizer'])
        saddle_y_values[s] = r['saddle_value']

    # get minima values from global minima list
    for m in global_minima:
        coord = tuple(m[0])
        val = m[1]
        minima_y_values[coord] = val

    fig, ax = plt.subplots(figsize=(12,8))

    # plot saddle points
    for idx,s in enumerate(unique_saddles):
        x = saddle_x_positions[s]
        y = saddle_y_values[s]
        ax.scatter(x, y, color="orange", edgecolor="black", s=30, zorder=3)
        ax.text(x, y + 0.02, f"S{idx}", ha="center", fontsize=8)

    # plot minima points with consistent IDs
    for idx,m in enumerate(unique_minima):
        x = minima_x_positions[m]
        y = minima_y_values[m]
        ax.scatter(x, y, color="skyblue", edgecolor="black", s=30, zorder=3)
        ax.text(x, y - 0.02, f"M{idx}", ha="center", fontsize=8)

    # draw edges
    for r in results:
        s = tuple(r['saddle_point'])
        m = tuple(r['minimizer'])
        x1 = saddle_x_positions[s]
        y1 = saddle_y_values[s]
        x2 = minima_x_positions[m]
        y2 = minima_y_values[m]
        ax.plot([x1,x2],[y1,y2], color="gray", lw=1)

    ax.set_xlabel("Abstract horizontal layout")
    ax.set_ylabel("Function Value f(x)")
    ax.set_title("Saddle-Minima Tree")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.yaxis.set_major_locator(plt.MaxNLocator(10))

    plt.tight_layout()
    plt.show()