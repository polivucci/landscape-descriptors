import torch
import pandas as pd
import numpy as np
from torch.optim import LBFGS
from test_schwefel.schwefel_function_nnmodule import Schwefel2D
import os

torch.set_default_dtype(torch.float64)

def offset_near_saddle(saddle_point, func, epsilon=0.01):
    """
    Offset the saddle point along its most unstable direction,
    scaled by radius of curvature (1 / |eigenvalue|).
    """
    point = saddle_point.clone().detach().requires_grad_(True)

    # Compute Hessian
    hessian = torch.autograd.functional.hessian(func, point)

    # Eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(hessian)
    
    # Get most negative eigenvalue and its eigenvector (unstable direction)
    unstable_idx = torch.argmin(eigvals)
    unstable_eigval = eigvals[unstable_idx]
    unstable_direction = eigvecs[:, unstable_idx]

    # Radius of curvature = 1 / |λ|
    radius_of_curvature = 1.0 / torch.abs(unstable_eigval)

    # Offset distance = ε * radius_of_curvature
    offset_distance = epsilon * radius_of_curvature

    # Offset along unstable direction
    direction = unstable_direction / torch.norm(unstable_direction)
    return saddle_point + offset_distance * direction, saddle_point - offset_distance * direction


def optimize_lbfgs(model ,loss_fn , max_iter=100, atol=1e-6, rtol=1e-5):
    """
    Optimize using PyTorch LBFGS to find local minimum
    """
    # Make sure model params require grad
    for p in model.parameters():
        p.requires_grad_(True)

    # x = start_point.clone().detach().requires_grad_(True) # Makes a copy of the input so the original isn't modified.

    # optimizer = LBFGS([x], max_iter=max_iter, line_search_fn="strong_wolfe") #Initializes the L-BFGS optimizer in PyTorch
    optimizer = LBFGS(model.parameters(), max_iter=max_iter, line_search_fn="strong_wolfe")

    trajectory = []  # Start with initial point 
    prev_coords = model.forward().detach().clone()

    def closure(): #Closure required because LBFGS evaluates the function multiple times during each iteration.
            optimizer.zero_grad()
            coords = model()
            loss = loss_fn(coords)
            loss.backward()
            fval = float(loss.item())
            trajectory.append((coords.detach().cpu().numpy(), fval)) #Appends each trajectory point
            return loss
    
    for _ in range(max_iter):
        optimizer.step(closure)
        
    # stopping criterion
        try:
            torch.testing.assert_close(
                model.forward().detach(), prev_coords, atol=atol, rtol=rtol
            )
            # if assert_close passes, we break early
            break
        except AssertionError:
            pass

        prev_coords = model.forward().detach().clone()
    
    # for _ in range(max_iter):
    #     optimizer.step(closure) # Triggers one full optimization run using the closure
        
    
    return model.forward().detach(), trajectory  # Return both final point and path


def compare_to_known_minima(min_point, minima_df, threshold=1e-3):
    """
    Find closest known minimum and check if within threshold
    """
    x_cols = sorted([col for col in minima_df.columns if col.startswith("x")])
    dists = minima_df.apply(
        lambda r: np.linalg.norm(min_point.numpy() - r[x_cols].values.astype(np.float32)),
        axis=1
    )
    closest_idx = dists.idxmin()
    return minima_df.loc[closest_idx][x_cols].tolist(), dists.min() < threshold, closest_idx


def trajectory_length(traj):
    """
    Get Total length of the trajectory for the actual Descent path
    """
    return sum(np.linalg.norm(traj[i] - traj[i-1]) for i in range(1, len(traj)))

def trace_from_saddle(saddle_point, minima_df, idx_saddle, loss_fn):
    """
    Given a saddle point, trace descent on both sides and match to known minima
    """
    saddle_point = torch.tensor(saddle_point, dtype=torch.float64)

    offset_p1, offset_p2 = offset_near_saddle(saddle_point, loss_fn) #gets both offset point with curvature radius 


    saddle_value = loss_fn(saddle_point).item() #Get exact saddle value for plotting on tree

    model1 = Schwefel2D(offset_p1[0].item(), offset_p1[1].item())
    minima1, traj1 = optimize_lbfgs(model1, loss_fn)   #Optimzation to find Minima of a Saddle point and Trajectory for offset 1 
    arrival1_val = loss_fn(minima1).item() # Minimizer Function value of minima1

    model2 = Schwefel2D(offset_p2[0].item(), offset_p2[1].item())
    minima2, traj2 = optimize_lbfgs(model2, loss_fn)   #Optimzation to find Minima of a Saddle point and Trajectory for offset 2
    arrival2_val = loss_fn(minima2).item() # Minimizer Function value of minima2

    min1_coords, connected1, idx_minima1 = compare_to_known_minima(minima1, minima_df)   #Comparing Minima1 from Optimizer with known Minima from CSV
    min2_coords, connected2, idx_minima2 = compare_to_known_minima(minima2, minima_df)   #Comparing Minima2 from Optimizer with known Minima from CSV

    length1 = trajectory_length(traj1[0]) #calculating total length of the trajectory for offset 1
    length2 = trajectory_length(traj2[0]) #calculating total length of the trajectory for offset 1

    #Making table for results found
    result = [ 
        {
            "saddle_point": tuple(saddle_point.tolist()),
            "saddle_value": saddle_value,
            "descent":tuple(float(x) for x in minima1), #Undo Comment if you want to check what minima does the saddle point get after going through LBFG optimizer  
            "minimizer": tuple(float(x) for x in min1_coords),
            "min_value": arrival1_val,
            "trajectory_length": length1,
            "is_connected": "yes" if connected1 else "no"
        },
        {
            "saddle_point": tuple(saddle_point.tolist()),
            "saddle_value": saddle_value,
            "descent":tuple(float(x) for x in minima2), #Undo Comment if you want to check what minima does the saddle point get after going through LBFG optimizer  
            "minimizer": tuple(float(x) for x in min2_coords),
            "min_value": arrival2_val,
            "trajectory_length": length2,
            "is_connected": "yes" if connected2 else "no",
        }
    ]
    # Storing path data for visualizing further
    connected_minimizers = []
    trajectories = []
    minima_id =[] 
    if connected1:
        connected_minimizers.append(tuple(minima1.tolist()))
        trajectories.append(traj1)
        minima_id.append(int(idx_minima1))
    if connected2:
        connected_minimizers.append(tuple(minima2.tolist()))
        trajectories.append(traj2)
        minima_id.append(int(idx_minima2))

    path_data = None
    if connected_minimizers:
        path_data = {
            "saddle": tuple(saddle_point.tolist()),
            "minimizers": connected_minimizers,
            "trajectories": trajectories, 
            "saddle_id": idx_saddle,
            "minima_id": minima_id
        }
    return result,path_data

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
    output_file=f'{base_dir}/saddle_to_minima_indices{dim}D.csv'
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

def trace_connectivity(saddles_df, minima_df, dataframe, func, return_paths=False):
    """
    Trace connectivity from saddle points to minima.

    Parameters:
        saddles_df (pd.DataFrame): DataFrame of saddle points.
        minima_df (pd.DataFrame): DataFrame of minima.
        dataframe (pd.DataFrame): Critical point index map (e.g. critical_points.csv).
        func (callable): The target function, e.g., schwefel(x[0], x[1]).
        return_paths (bool): Whether to return full descent path data.

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

        results,path_data = trace_from_saddle(saddle_point, minima_df, idx, func)
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