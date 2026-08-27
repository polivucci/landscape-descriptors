import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

# Plot font Settings
plt.rcParams['font.family'] = 'sans-serif' # or 'sans-serif' or 'monospace'
plt.rcParams['font.serif'] = 'cmr10'
plt.rcParams['font.sans-serif'] = 'cmss10'
plt.rcParams['font.monospace'] = 'cmtt10'
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["mathtext.default"] = "regular"
# plt.rcParams["axes.formatter.use_mathtext"] = True # to fix the minus signs
plt.rcParams["axes.unicode_minus"] = False

# Plot color scheme
style_params = {
'saddle_color': "white",
'minima_color': "blue",
'maxima_color': "blue",
'edge_color': "blue",
'text_color': "black",
'cmap': cm.RdPu,
}

def extract_point_data(crit_df):
    """Extracts coordinates and formats labels from the critical points DataFrame."""
    point_configs = {
        'minimum': {'prefix': 'M'},
        'maximum': {'prefix': 'm'},
        'saddle': {'prefix': 'S'}
    }
    
    data_dict = {}
    for p_type, config in point_configs.items():
        subset = crit_df[crit_df['type'] == p_type]
        data_dict[p_type] = {
            'x': subset['x1'].to_numpy(),
            'y': subset['x2'].to_numpy(),
            'labels': [f"{config['prefix']}{idx}" for idx in subset.index]
        }
        
    return data_dict

def plot_critical_points(ax, point_data, style_params):
    """Plots the scatter points and text annotations for critical points.
    """
    txtpad = style_params.get('txtpad', 0.025)
    
    plot_configs = {
        'minimum': {'color': style_params['minima_color'], 'label': 'Minima', 'fillstyle': 'bottom'},
        'maximum': {'color': style_params['maxima_color'], 'label': 'Maxima', 'fillstyle': 'top'},
        'saddle': {'color': style_params['saddle_color'], 'label': 'Saddles', 'fillstyle': 'full'}
    }
    
    for p_type, data in point_data.items():
        # Skip if no points of this type exist
        if len(data['x']) == 0:
            continue
            
        cfg = plot_configs[p_type]
        
        # Plot scatter points# Plot markers using ax.plot (linestyle='None' prevents lines connecting the dots)
        ax.plot(data['x'], data['y'], 
                marker='o', 
                linestyle='None',
                markerfacecolor=cfg['color'], 
                markerfacecoloralt='w',
                markeredgecolor=style_params['edge_color'], 
                markeredgewidth=1,
                fillstyle=cfg['fillstyle'], 
                markersize=6, # markersize=7 roughly matches the visual area of s=50
                label=cfg['label'])
        
        # Plot text annotations
        for x, y, label in zip(data['x'], data['y'], data['labels']):
            ax.text(x + txtpad, y + txtpad, label, fontsize=8, 
                    color=style_params['text_color'], zorder=10)

def plot_results_with_paths(func, critical_points_df, 
                            connectivity_df, bounds=(0, 1),
                            res=100, txtpad=0.015, fig=None, **plot_kwargs):
    """
    Visualize function landscape with minima, saddle points, and descent paths
    with consistent labels S0/S1.. for saddle points and M0/M1.. for minima.
    """

    # set up grid and eval function
    if fig is None:
        fig = plt.figure(figsize=(6, 6))

    ax = fig.gca()

    # set up grid and eval function
    if bounds==(0.0, 1.0): bounds = 2*((0.0, 1.0),)
    bounds = list(bounds.values())
    x = np.linspace(bounds[0][0], bounds[0][1], res)
    y = np.linspace(bounds[1][0], bounds[1][1], res)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.array([[func(x, y) for x, y in zip(row_x, row_y)]
                  for row_x, row_y in zip(X, Y)])

    # 2D Contour plot
    contour = ax.contourf(X, Y, Z, levels=20, cmap=style_params['cmap'], **plot_kwargs)
    fig.colorbar(contour, label='Function Value')

    # critical points plot
    if critical_points_df is not None:

        # Load data 
        crit_df = critical_points_df
        
        point_data = extract_point_data(crit_df)

        plot_critical_points(ax, point_data, style_params)

        if connectivity_df is not None:
            conn_df = connectivity_df

            # Extracting the indices which are connected
            saddle_indices = conn_df['index_1'].unique()
            minima_indices = conn_df['index_2'].unique()

            # Plot descent paths
            for _, row in conn_df.iterrows():
                s_row = crit_df.loc[int(row['index_1'])]
                m_row = crit_df.loc[int(row['index_2'])]
                saddle = (s_row['x1'], s_row['x2'])
                minimum = (m_row['x1'], m_row['x2'])
                ax.annotate('', xy=minimum, xytext=saddle,
                            arrowprops=dict(arrowstyle='->', color=edge_color))

    fig.tight_layout()
    return fig

def plot_results_with_paths_3d(func, critical_points_csv, 
                               saddle_to_minima_csv, bounds=(0, 1),
                               res=50, fig=None, slice_levels=[0.25, 0.5, 0.75]):
    """
    3D plot of the function labeled minima, saddle points, and descent paths.
    Supports functions of the form func(x, y, z).
    """

    if fig is None:
        fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111, projection='3d', )
    ax.set_proj_type('ortho')
    # ax.set_facecolor('pink')
   
    if critical_points_csv is not None and saddle_to_minima_csv is not None:
        crit_df = pd.read_csv(critical_points_csv)
        conn_df = pd.read_csv(saddle_to_minima_csv)

        saddle_indices = conn_df['index_1'].unique()
        minima_indices = conn_df['index_2'].unique()

        # Plot minima
        for i in minima_indices:
            row = crit_df.iloc[i]
            coords = [row[col] for col in sorted(row.index) if col.startswith("x")]
            ax.scatter(*coords, color=minima_color, edgecolor=edge_color, s=50, label='Minima' if i == minima_indices[0] else "")
            textcoords = [coord+0.02 for coord in coords]
            ax.text(*textcoords, f"M{i}", fontsize=9, color=text_color)

        # Plot saddles
        for i in saddle_indices:
            row = crit_df.iloc[i]
            coords = [row[col] for col in sorted(row.index) if col.startswith("x")]
            ax.scatter(*coords, color=saddle_color, edgecolor=edge_color, s=50, label='Saddle' if i == saddle_indices[0] else "")
            textcoords = [coord+0.02 for coord in coords]
            ax.text(*textcoords, f"S{i}", fontsize=9, color=text_color)

        # Plot descent paths
        for _, row in conn_df.iterrows():
            s_row = crit_df.iloc[int(row['index_1'])]
            m_row = crit_df.iloc[int(row['index_2'])]

            s_coords = [s_row[col] for col in sorted(s_row.index) if col.startswith("x")]
            m_coords = [m_row[col] for col in sorted(m_row.index) if col.startswith("x")]

            ax.plot([s_coords[0], m_coords[0]],
                    [s_coords[1], m_coords[1]],
                    [s_coords[2], m_coords[2]],
                    color=edge_color, linestyle='--')

    return fig

 # for zl in slice_levels:
    #     idx = np.argmin(np.abs(z - zl))
    #     z_fixed = z[idx]
    #     f_slice = f_vals[:, :, idx]
    #     X, Y = np.meshgrid(x, y, indexing='ij')
    #     ax.plot_surface(X, Y, z_fixed * np.ones_like(X),
    #                     facecolors=plt.cm.get_cmap(cmap)(f_slice / f_vals.max()),
    #                     rstride=1, cstride=1, antialiased=False, shade=False,
    #                     alpha=0.6)
    # # Choose slices (e.g., fixed x) to visualize
    # # slice_indices = [0, res // 4, res // 2, 3 * res // 4, res - 1]
    # # for h in slice_indices:
    # #     cs = ax.contourf(yy[h], zz[h], f_vals[h], zdir='x', offset=x[h], alpha=0.5, cmap='viridis')

def plot_saddle_tree_with_function(dataframe, nodes_csv, edges_csv, fig=None, 
                                   minus_y_axis=False, 
                                   txt_pads=(0.1, 0.1), markersize=50, fontsize=8, **kwargs):

    # Read node and edge data
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    # --- Plotting ---
    markersize = kwargs.get("markersize", 50)
    fontsize = kwargs.get("fontsize", 8)
    edge_color = kwargs.get("edge_color", "gray")
    saddle_color = kwargs.get("saddle_color", "red")
    minima_color = kwargs.get("minima_color", "blue")
    text_color = kwargs.get("text_color", "black")

    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        ax = fig.gca()

    # invert y axis
    mn = 1.0
    if minus_y_axis: mn = -1.0

    x_pos = dict(zip(nodes_df["index"], nodes_df["order"]))
    y_val = dict(zip(nodes_df["index"], mn*nodes_df["f_value"]))
    # yrange = nodes_df["f_value"].max()-nodes_df["f_value"].min()
    # xrange = nodes_df["order"].max()

    saddle_label_added = False
    minima_label_added = False
    
    # print('x_pos', x_pos)
    # print('nodes_df', nodes_df)
    # print('edges_df', edges_df)
    # markersize=50
    # fontsize=8
    # if 'markersize' in kwargs.keys(): markersize=kwargs['markersize']
    # if 'fontsize' in kwargs.keys(): fontsize=kwargs['fontsize']

    # Plot edges:
    for _, row in edges_df.iterrows():
        i, j = row["index_1"], row["index_2"]
        x1, y1 = x_pos[i] * 2, y_val[i]
        x2, y2 = x_pos[j] * 2, y_val[j]
        ax.plot([x1, x2], [y1, y2], color=edge_color, lw=1)
        # # eckige branches:
        # ax.plot([x1, x2], [y1, y1], color=edge_color, lw=1)
        # ax.plot([x2, x2], [y1, y2], color=edge_color, lw=1)

    for idx in nodes_df["index"].to_list():
        x = x_pos[idx] * 2
        y = y_val[idx]

        point_type = dataframe.loc[idx, "type"].lower()
        padx, pady = txt_pads[0], txt_pads[1]
        if point_type == "saddle":
            ax.scatter(x, y, color=saddle_color, edgecolor=edge_color, s=markersize, zorder=3,
                       label="Saddle" if not saddle_label_added else "")
            inv = ax.transData.inverted()
            # pad = -0.15*inv.transform((np.sqrt(markersize/np.pi),np.sqrt(markersize/np.pi)))
            ax.text(x+padx, y+pady, f"S{idx}", ha="center", va="center",  fontsize=fontsize, color=text_color, clip_on=True)
            saddle_label_added = True
        elif point_type == "minimum":
            ax.scatter(x, y, color=minima_color, edgecolor=edge_color, s=markersize, zorder=3,
                       label="Minima" if not minima_label_added else "")
            ax.text(x, y-pady, f"M{idx}", ha="center", va="center", fontsize=fontsize, color=text_color, clip_on=True)
            minima_label_added = True

    ax.grid(True, linestyle="--", alpha=0.4, axis='y')
    ax.set_xticks([])
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    
    return fig

def plot_full_connectivity_tree_style(nodes_csv, edges_csv, fig=None, txt_pads=(0.1, 0.1), **kwargs):
    """
    Plots a full connectivity graph with critical points as nodes,
    styled similarly to the saddle tree plot layout.
    Nodes are ordered by type (saddles first, then minima) and spaced along x by index.
    Vertical axis is f_value.
    """

    
    # Load CSVs
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    # Create an artificial "order" for plotting: saddles first, then minima
    saddles_df = nodes_df[nodes_df["type"] == "saddle"].copy()
    minima_df = nodes_df[nodes_df["type"] == "minimum"].copy()

    saddles_df["order"] = range(len(saddles_df))
    minima_df["order"] = range(len(saddles_df), len(saddles_df) + len(minima_df))

    nodes_df = pd.concat([saddles_df, minima_df])
    # nodes_df.set_index("index", inplace=True)

    # Plot setup
    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        ax = fig.gca()

    x_pos = nodes_df["order"].to_dict()
    y_val = nodes_df["f_value"].to_dict()
    yrange = nodes_df["f_value"].max() - nodes_df["f_value"].min()

    markersize = kwargs.get("markersize", 50)
    fontsize = kwargs.get("fontsize", 8)
    edge_color = kwargs.get("edge_color", "gray")
    saddle_color = kwargs.get("saddle_color", "red")
    minima_color = kwargs.get("minima_color", "blue")
    text_color = kwargs.get("text_color", "black")
    padx, pady = txt_pads

    # Plot edges
    for _, row in edges_df.iterrows():
        i, j = row["index_1"], row["index_2"]
        if i in x_pos and j in x_pos:
            x1, y1 = x_pos[i] * 2, y_val[i]
            x2, y2 = x_pos[j] * 2, y_val[j]
            ax.plot([x1, x2], [y1, y2], color=edge_color, lw=1)

    # Plot nodes
    saddle_label_added = False
    minima_label_added = False
    for idx in x_pos:
        x = x_pos[idx] * 2
        y = y_val[idx]
        point_type = nodes_df.loc[idx, "type"].lower()

        if point_type == "saddle":
            ax.scatter(x, y, color=saddle_color, edgecolor=edge_color, s=markersize, zorder=3,
                       label="Saddle" if not saddle_label_added else "")
            ax.text(x + padx, y + pady, f"S{idx}", ha="center", va="center", fontsize=fontsize, color=text_color)
            saddle_label_added = True
        elif point_type == "minimum":
            ax.scatter(x, y, color=minima_color, edgecolor=edge_color, s=markersize, zorder=3,
                       label="Minima" if not minima_label_added else "")
            ax.text(x, y - pady, f"M{idx}", ha="center", va="center", fontsize=fontsize, color=text_color)
            minima_label_added = True

    ax.set_title("Full Connectivity Graph (Tree Layout Style)")
    ax.grid(True, linestyle="--", alpha=0.4, axis='y')
    ax.set_xticks([])
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.legend()

    return fig


import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt


def plot_connectivity_tree_igraph(edges_csv, heights_csv, fig, **kwargs):

    edges_df = pd.read_csv(edges_csv)           # columns: index_1, index_2
    heights_df = pd.read_csv(heights_csv)       # columns: node, height

    # build graph, map your integer labels to igraph's 0..N-1 vertex ids
    nodes = sorted(set(edges_df["index_1"]) | set(edges_df["index_2"]))
    label_to_idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(directed=True)
    g.add_vertices(len(nodes))
    g.vs["label"] = nodes
    g.add_edges([(label_to_idx[a], label_to_idx[b]) for a, b in edges_df.itertuples(index=False)])

    root_node = heights_df["index"][0]
    layout = g.layout_reingold_tilford(root=[label_to_idx[root_node]])
    xs = {nodes[i]: layout[i][0] for i in range(len(nodes))}

    height_map = dict(zip(heights_df["index"], heights_df["f_value"]))
    type_map = dict(zip(heights_df["index"], heights_df["type"]))

    markersize = kwargs.get("markersize", 50)
    fontsize = kwargs.get("fontsize", 8)
    edge_color = kwargs.get("edge_color", "gray")
    saddle_color = kwargs.get("saddle_color", "red")
    minima_color = kwargs.get("minima_color", "blue")
    text_color = kwargs.get("text_color", "black")
    padx, pady = kwargs.get("txt_pads", (0.1, 0.1)) 

    # plot
    # fig, ax = plt.subplots()
    ax = fig.gca()
    for a, b in edges_df.itertuples(index=False):
        ax.plot([xs[a], xs[b]], [height_map[a], height_map[b]], lw=1, color=edge_color, zorder=1)
    for n in nodes:
        point_type = type_map[n]
        mcolor = minima_color
        if point_type == "saddle": 
            mcolor = saddle_color
            ax.annotate(f"S{n}", (xs[n], height_map[n]), xytext=(xs[n]+padx, height_map[n]+pady))
        elif point_type == "minimum": 
            ax.annotate(f"M{n}", (xs[n], height_map[n]), xytext=(xs[n], height_map[n]-pady))
        ax.scatter(xs[n], height_map[n], color=mcolor, edgecolor=edge_color, zorder=2)
        # ax.annotate(str(n), (xs[n], height_map[n]))

    ax.invert_yaxis()  # if you want "height" growing downward, e.g. depth-like trees
    ax.grid(True, linestyle="--", alpha=0.4, axis='y')
    ax.set_xticks([])
    ax.spines[['top', 'right', 'bottom']].set_visible(False)

    return fig