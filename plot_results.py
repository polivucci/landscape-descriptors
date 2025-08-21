import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools
from matplotlib.ticker import MaxNLocator

import pandas as pd

dataframe = pd.read_csv("critical_points_schwefel.csv")
def plot_results_with_paths(func, critical_points_csv, 
                            saddle_to_minima_csv, bounds=(0, 1),
                            res=100, fig=None):
    """
    Visualize function landscape with minima, saddle points, and descent paths
    with consistent labels S0/S1.. for saddle points and M0/M1.. for minima.
    """

    # Set up grid and evaluate function
    if fig is None:
        fig = plt.figure(figsize=(6, 6))

    ax = fig.gca()
    # Load data 
    crit_df = pd.read_csv(critical_points_csv)
    conn_df = pd.read_csv(saddle_to_minima_csv)

    # Set up grid and evaluate function
    x = np.linspace(bounds[0], bounds[1], res)
    y = np.linspace(bounds[0], bounds[1], res)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.array([[func((x, y)) for x, y in zip(row_x, row_y)]
                  for row_x, row_y in zip(X, Y)])

    cmap = cm.inferno

    # 2D Contour plot
    
    contour = ax.contourf(X, Y, Z, levels=20, cmap=cmap)
    fig.colorbar(contour, label='Function Value')

    # Extracting the indices which are connected
    saddle_indices = conn_df['index_saddle'].unique()
    minima_indices = conn_df['index_minimum'].unique()

    minima_coords = []
    saddle_coords = []
    # Plot minima with labels
    for i in minima_indices:
        row = crit_df.iloc[i]
        point = (row['x1'], row['x2'])
        minima_coords.append(point)
        ax.text(point[0] + 0.01, point[1] + 0.01, f"M{i}", fontsize=8, color="white")
    
    minima_coords = np.array(minima_coords)
    ax.scatter(minima_coords[:, 0], minima_coords[:, 1],
           c='red', s=80, edgecolor='k', linewidth=1.5, label='Minima')
        
    # Plot saddle points with labels
    for i in saddle_indices:
        row = crit_df.iloc[i]
        point = (row['x1'], row['x2'])
        saddle_coords.append(point)
        ax.text(point[0] + 0.01, point[1] + 0.01, f"S{i}", fontsize=8, color="white")

    saddle_coords = np.array(saddle_coords)
    ax.scatter(saddle_coords[:, 0], saddle_coords[:, 1],
            c='green', s=80, edgecolor='k', linewidth=1.5, label='Saddles')


    # Plot descent paths
    for _, row in conn_df.iterrows():
        s_row = crit_df.iloc[int(row['index_saddle'])]
        m_row = crit_df.iloc[int(row['index_minimum'])]
        saddle = (s_row['x1'], s_row['x2'])
        minimum = (m_row['x1'], m_row['x2'])
        ax.annotate('', xy=minimum, xytext=saddle,
                    arrowprops=dict(arrowstyle='->', color='blue'))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Contour with Labels and Descent Paths')    
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig

def plot_saddle_tree_with_function(dataframe, nodes_csv, edges_csv, fig=None):
    

    # Read node and edge data
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    
    # --- Plotting ---
    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        ax = fig.gca()

    x_pos = dict(zip(nodes_df["index"], nodes_df["order"]))
    y_val = dict(zip(nodes_df["index"], nodes_df["f_value"]))

    # Plot edges
    for _, row in edges_df.iterrows():
        i, j = row["index_1"], row["index_2"]
        x1, y1 = x_pos[i] * 2, y_val[i]
        x2, y2 = x_pos[j] * 2, y_val[j]
        ax.plot([x1, x2], [y1, y2], color="red", lw=2)

    saddle_color = "orange"
    minima_color = "skyblue"
    saddle_label_added = False
    minima_label_added = False

    for idx, row in dataframe.iterrows():
        x = x_pos[idx] * 2
        y = y_val[idx]

        point_type = dataframe.loc[idx, "type"].lower()
        if point_type == "saddle":
            ax.scatter(x, y, color=saddle_color, edgecolor="black", s=60, zorder=3,
                       label="Saddle" if not saddle_label_added else "")
            ax.text(x, y + 0.025, f"S{idx}", ha="center", fontsize=8)
            saddle_label_added = True
        elif point_type == "minimum":
            ax.scatter(x, y, color=minima_color, edgecolor="black", s=60, zorder=3,
                       label="Minima" if not minima_label_added else "")
            ax.text(x, y - 0.03, f"M{idx}", ha="center", fontsize=8)
            minima_label_added = True

    ax.set_xlabel("Tree-ordered x-position (DFS)")
    ax.set_ylabel("Function Value f(x)")
    ax.set_title("Saddle-Minima MST Tree")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    ax.yaxis.set_major_locator(MaxNLocator(10))
    fig.tight_layout()
    return fig
