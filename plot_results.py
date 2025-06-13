import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_results_with_paths(func, minima=None, bounds=(0, 1),
                            saddle_points=None, paths=None, res=100,
                            show_connectivity=True):

    """
    Visualize function landscape with minima, saddle points, and descent paths.

    Parameters:
        func: callable, the 2D objective function taking (x, y) tuple.
        minima: list of [((x1, x2), f_value)] minima points.
        bounds: tuple of domain bounds in both x and y directions.
        saddle_points: list of [((x1, x2), f_value)] saddle points.
        paths: list of dictionaries with keys:
               - 'saddle': (x1, x2)
               - 'minimizers': [(x1, x2), (x1, x2)]
        res: int, resolution of the grid.
        show_connectivity: bool, whether to draw arrows from saddles to minimizers.

    """

    # Set up grid and evaluate function
    plt.figure(figsize=(10, 5))
    x = np.linspace(bounds[0], bounds[1], res)
    y = np.linspace(bounds[0], bounds[1], res)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.array([[func((x, y)) for x, y in zip(row_x, row_y)]
                  for row_x, row_y in zip(X, Y)])

    cmap = cm.inferno

    # --- 3D PLOT ---
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

    # Plot saddle points
    if saddle_points:
        saddle_points_array = np.array([sp[0] for sp in saddle_points])
        saddle_values = [func(p) for p in saddle_points_array]
        ax1.scatter(saddle_points_array[:, 0], saddle_points_array[:, 1], saddle_values,
                    c='green', s=50, depthshade=True, label=f'Saddle Points (n={len(saddle_points_array)})')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()

    # --- 2D CONTOUR PLOT ---
    ax2 = plt.subplot(1, 2, 2)
    contour = ax2.contourf(X, Y, Z, levels=20, cmap=cmap)
    plt.colorbar(contour, ax=ax2, label='Function Value')

    # Plot minima
    if minima:
        ax2.scatter(minima_points[:, 0], minima_points[:, 1],
                    c='red', s=80, edgecolor='k', linewidth=1.5,
                    label=f'Minima (n={len(minima_points)})')
    # Plot saddle points
    if saddle_points:
        ax2.scatter(saddle_points_array[:, 0], saddle_points_array[:, 1],
                    c='green', s=80, edgecolor='k', linewidth=1.5,
                    label=f'Saddle Points (n={len(saddle_points_array)})')
        
    # Plot Path connectivity from saddle point to minima
    if paths and show_connectivity:
        for path in paths:            
            saddle = path['saddle']
            mins = path['minimizers']
            for m in mins:
                ax2.annotate('', xy=m, xytext=saddle,
                             arrowprops=dict(arrowstyle='->', color='blue'))

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Critical Points and Descent Paths')
    ax2.legend()

    plt.tight_layout()
    plt.show()