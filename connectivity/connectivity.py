import torch
import pandas as pd
import numpy as np
import os
from critical_points.optimizers import optimize_lbfgs

torch.set_default_dtype(torch.float64)

from critical_points.torch_critical_point_finder import flatten_hessian_blocks
def compute_model_hessian(model, input, loss_func):

    active_params = [True if p.requires_grad else False for p in model.parameters()]
    params_flatten, unflatten = torch.utils._pytree.tree_flatten(dict(model.named_parameters()))

    def eval_loss_fn_params(flat_params):
        """Defines the functional eval of the model given the parameters.
        Flatten is required for torch.func.hessian to return a 2-tensor and not a nested dict,
        as function like torch.func.hessian from torch.func expect a single input tensor or PyTree (nested dict) of tensors.
        and functional calls work with nested dicts (i.e. unflattened).
        """
        params_dict = torch.utils._pytree.tree_unflatten(flat_params, unflatten) # see ChatGPT convo
        y = torch.func.functional_call(model, params_dict, (input,))
        return loss_func(y)

    hessian = torch.func.hessian(eval_loss_fn_params)(params_flatten)
    hessian = flatten_hessian_blocks(hessian)
    hessian = hessian[active_params][:, active_params]
    
    return hessian

def offset_near_saddle(saddle_point, model_saddle, input, loss_func, epsilon=0.01):
    """
    Offset the saddle point along its most unstable direction,
    scaled by radius of curvature (1 / |eigenvalue|).
    """

    point = saddle_point.clone().detach().unsqueeze(1)

    # Compute Hessian
    hessian = compute_model_hessian(model_saddle, input, loss_func).detach()

    # Eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(hessian)
    
    # Get negative eigenvalues and their eigenvectors (unstable directions)
    unstable_idx = (eigvals<0)
    unstable_eigval = eigvals[unstable_idx]
    unstable_direction = eigvecs[:, unstable_idx]

    # Radius of curvature = 1 / |λ|
    radius_of_curvature = 1.0 / torch.abs(unstable_eigval)

    # Offset distance = ε * radius_of_curvature
    offset_distance = (epsilon * radius_of_curvature).unsqueeze(0)

    # Offset along unstable directions
    direction = unstable_direction / torch.norm(unstable_direction, dim=0, keepdim=True)
    offset_points = torch.cat((point + offset_distance * direction, point - offset_distance * direction), dim=1).T
    
    return offset_points

# def offset_near_saddle(saddle_point, func, epsilon=0.01):
#     """
#     Offset the saddle point along its most unstable direction,
#     scaled by radius of curvature (1 / |eigenvalue|).
#     """
#     point = saddle_point.clone().detach().requires_grad_(True)

#     # Compute Hessian
#     hessian = torch.autograd.functional.hessian(func, point)

#     # Eigendecomposition
#     eigvals, eigvecs = torch.linalg.eigh(hessian)
    
#     # Get most negative eigenvalue and its eigenvector (unstable direction)
#     unstable_idx = torch.argmin(eigvals)
#     unstable_eigval = eigvals[unstable_idx]
#     unstable_direction = eigvecs[:, unstable_idx]

#     # Radius of curvature = 1 / |λ|
#     radius_of_curvature = 1.0 / torch.abs(unstable_eigval)

#     # Offset distance = ε * radius_of_curvature
#     offset_distance = epsilon * radius_of_curvature

#     # Offset along unstable direction
#     direction = unstable_direction / torch.norm(unstable_direction)
#     return saddle_point + offset_distance * direction, saddle_point - offset_distance * direction



def compare_to_known_minima(min_point, minima_df, threshold=1e-3):
    """ Find closest known minimum and check if within threshold
    """
    # x_cols = sorted([col for col in minima_df.columns if col.startswith("x")])
    # dists = minima_df.apply(
    #     lambda r: np.linalg.norm(min_point.numpy() - r[x_cols].values.astype(np.float32)),
    #     axis=1
    # )
    # closest_idx = dists.idxmin()
    # return minima_df.loc[closest_idx][x_cols].tolist(), dists.min() < threshold, closest_idx
    x_cols = sorted([col for col in minima_df.columns if col.startswith("x")])
    dists = minima_df.apply(
        lambda r: np.linalg.norm(min_point.numpy() - r[x_cols].values.astype(np.float64)),
        axis=1
    )
    closest_idx = dists.idxmin()
    return minima_df.loc[closest_idx][x_cols].tolist(), dists.min() < threshold, closest_idx


def trajectory_length(traj):
    """
    Get Total length of the trajectory for the actual Descent path
    """
    return sum(np.linalg.norm(traj[i] - traj[i-1]) for i in range(1, len(traj)))

def trace_from_saddle(saddle_point, minima_df, idx_saddle, loss_fn, nn_model, input, log_paths=False):
    """
    Given a saddle point, trace descent on both sides and match to known minima
    """
    
    saddle_point = torch.tensor(saddle_point)
    model_saddle = nn_model(saddle_point)
    active_params = [True if p.requires_grad else False for p in model_saddle.parameters()]

    # offset_p1, offset_p2 = offset_near_saddle(saddle_point, loss_fn) #gets both offset point with curvature radius 
    # offset_p1, offset_p2 = offset_near_saddle(saddle_point, model_saddle, input, loss_fn, epsilon=0.01) 
    offsets = offset_near_saddle(saddle_point, model_saddle, input, loss_fn, epsilon=0.01) 

    saddle_value = loss_fn(model_saddle(input)).item() #Get exact saddle value for plotting on tree

    result=[]
    connected_minimizers = []
    trajectories = []
    minima_id =[]
    for offset_coords in offsets:
        #Optimzation to find Minima of a Saddle point and Trajectory for offset 1 
        model1 = nn_model(offset_coords)
        minima1, arrival1_val, traj1 = optimize_lbfgs(model1, input, loss_fn, log_paths=True)   
        minima1 = minima1[active_params]
        min1_coords, connected1, idx_minima1 = compare_to_known_minima(minima1, minima_df)   
        length1 = trajectory_length(traj1[0]) #calculating total length of the trajectory for offset 1
        
        #Making table for results found
        result.append(
            {
                "saddle_point": tuple(saddle_point.tolist()),
                "saddle_value": saddle_value,
                "descent":tuple(float(x) for x in minima1), #Undo Comment if you want to check what minima does the saddle point get after going through LBFG optimizer  
                "minimizer": tuple(float(x) for x in min1_coords),
                "min_value": arrival1_val,
                "trajectory_length": length1,
                "is_connected": "yes" if connected1 else "no"
            }
        ) 
        if connected1:
            connected_minimizers.append(tuple(minima1.tolist()))
            trajectories.append(traj1)
            minima_id.append(int(idx_minima1))

    path_data = None
    if connected_minimizers:
        path_data = {
            "saddle": tuple(saddle_point.tolist()),
            "minimizers": connected_minimizers,
            "trajectories": trajectories, 
            "saddle_id": idx_saddle,
            "minima_id": minima_id
        }
    
    return result, path_data

def save_descent_paths(paths):
    """
    Save descent paths to individual CSV files:
    Format: descent_path_S{saddle_index}_M{min_index}.csv
    Each file contains a trajectory: x, y coordinates
    """
    base_dir = "descent_paths"
    os.makedirs(base_dir, exist_ok=True)
    for path in paths:
        saddle_idx = path.get("saddle_id")
        minima_indices = path.get("minima_id", [])
        trajectories = path.get("trajectories", [])
        for min_idx, traj in zip(minima_indices, trajectories):
            rows = []
            for coords, loss in traj:
                row = list(coords) + [loss]
                rows.append(row)

            dim = len(traj[0][0]) if traj else 0
            columns = [f"x{i}" for i in range(dim)] + ["loss"]
            df = pd.DataFrame(rows, columns=columns)
            filename = f"descent_path_S{saddle_idx}_M{min_idx}.csv"
            filepath = os.path.join(base_dir, filename)
            df.to_csv(filepath, index=False)

def extract_connection_indices(all_results, critical_points_df):
    """
    From all_results, generate a CSV with index_saddle and index_minimum
    corresponding to connected pairs in the original critical_points.csv
    """
    base_dir = "results"
    os.makedirs(base_dir, exist_ok=True)
    # Create a mapping from (x1, x2)or(x1, x2, x3) → index in critical_points.csv
    x_cols = sorted([col for col in critical_points_df.columns if col.startswith("x")])

    coord_to_index = {
        tuple(round(row[col], 8) for col in x_cols): idx
        for idx, row in critical_points_df.iterrows()
    }
    dim = len(x_cols)
    output_file=f'{base_dir}/connectivity_graph.csv'
    connection_rows = []
    for entry in all_results:
        if entry["is_connected"] == "yes":
            saddle_coord = tuple(round(c, 8) for c in entry["saddle_point"])
            minima_coord = tuple(round(c, 8) for c in entry["minimizer"])

            idx_saddle = coord_to_index.get(saddle_coord)
            idx_minima = coord_to_index.get(minima_coord)

            if idx_saddle is not None and idx_minima is not None:
                connection_rows.append((idx_saddle, idx_minima))

    # Save to CSV
    df = pd.DataFrame(connection_rows, columns=["index_saddle", "index_minimum"])
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} connections to {output_file}")

def trace_connectivity(saddles_df, minima_df, dataframe, func, nn_model, input, return_paths=False):
    """
    Trace connectivity from saddle points to minima.

    Parameters:
        saddles_df (pd.DataFrame): DataFrame of saddle points.
        minima_df (pd.DataFrame): DataFrame of minima.
        dataframe (pd.DataFrame): Critical point index map (e.g. critical_points.csv).
        func (callable): The target function, e.g., schwefel(x[0], x[1]).
        return_paths (bool): Whether to return full descent path data.
        nn_model(Function); any given model

    Returns:
        all_results: list of saddle-to-minima connections.
        minima: list of minima as (coords, f_value) tuples.
        dataframe: passed-through dataframe.
        paths (optional): list of descent path data, if return_paths=True.
    """
    os.makedirs("plots", exist_ok=True)
    all_results = []
    paths=[]
    for idx, row in saddles_df.iterrows():
        saddle_point = [row[col] for col in sorted(row.index) if col.startswith("x")]

        results, path_data = trace_from_saddle(saddle_point, minima_df, idx, func, nn_model, input, log_paths=return_paths)
        all_results.extend(results)
        if path_data is not None:
            paths.append(path_data)

    minima = [
        ([row[col] for col in sorted(row.index) if col.startswith("x")], row["f_value"])
        for _, row in minima_df.iterrows()
    ]

    if return_paths:
        return all_results, minima, dataframe, paths
    else:
        return all_results, minima, dataframe