import torch
import pandas as pd
import numpy as np
from torch.optim import LBFGS
from schwefel_function import schwefel  # Replace with your actual import
from plot_results import plot_results_with_paths
from plot_results import plot_saddle_tree_with_function
import matplotlib.pyplot as plt
from saddle_tree_calculations import saddle_tree_calculations
import itertools
import os

torch.set_default_dtype(torch.float64)

os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("descent_paths", exist_ok=True)

paths=[]
dataframe = pd.read_csv("critical_points_schwefel.csv")
saddles_df = dataframe[dataframe["type"] == "saddle"]
minima_df = dataframe[dataframe["type"] == "minimum"]

def offset_near_saddle(saddle_point, epsilon=0.01):
    """
    Offset the saddle point along its most unstable direction,
    scaled by radius of curvature (1 / |eigenvalue|).
    """
    point = saddle_point.clone().detach().requires_grad_(True)
    func = lambda x: schwefel(x[0], x[1])
    
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


def optimize_lbfgs(start_point, func, max_iter=100):
    """
    Optimize using PyTorch LBFGS to find local minimum
    """
    x = start_point.clone().detach().requires_grad_(True) # Makes a copy of the input so the original isn't modified.

    optimizer = LBFGS([x], max_iter=max_iter, line_search_fn="strong_wolfe") #Initializes the L-BFGS optimizer in PyTorch

    trajectory = []  # Start with initial point    

    def closure(): #Closure required because LBFGS evaluates the function multiple times during each iteration.
        optimizer.zero_grad()
        loss = func(x)
        loss.backward()
        trajectory.append(x.detach().clone().numpy()) #Appends each trajectory point
        return loss
    
    for _ in range(max_iter):
        optimizer.step(closure) # Triggers one full optimization run using the closure
        
    trajectory.append(x.detach().clone().numpy())
    return x.detach(), trajectory  # Return both final point and path


def compare_to_known_minima(min_point, minima_df, threshold=1e-3):
    """
    Find closest known minimum and check if within threshold
    """
    dists = minima_df.apply(lambda r: np.linalg.norm(min_point.numpy() - np.array([r["x1"], r["x2"]])), axis=1)
    closest_idx = dists.idxmin()
    return minima_df.loc[closest_idx][["x1", "x2"]].tolist(), dists.min() < threshold, closest_idx

def trajectory_length(traj):
    """
    Get Total length of the trajectory for the actual Descent path
    """
    return sum(np.linalg.norm(traj[i] - traj[i-1]) for i in range(1, len(traj)))

def trace_from_saddle(saddle_point, minima_df, idx_saddle):
    """
    Given a saddle point, trace descent on both sides and match to known minima
    """
    saddle_point = torch.tensor(saddle_point, dtype=torch.float64)

    offset_p1, offset_p2 = offset_near_saddle(saddle_point) #gets both offset point with curvature radius 

    func = lambda x: schwefel(x[0], x[1])

    saddle_value = func(saddle_point).item() #Get exact saddle value for plotting on tree
    minima1, traj1 = optimize_lbfgs(offset_p1, func)   #Optimzation to find Minima of a Saddle point and Trajectory for offset 1 
    minima2, traj2 = optimize_lbfgs(offset_p2, func)   #Optimzation to find Minima of a Saddle point and Trajectory for offset 2

    arrival1_val = func(minima1).item() # Minimizer Function value of minima1
    arrival2_val = func(minima2).item() # Minimizer Function value of minima2

    min1_coords, connected1, idx_minima1 = compare_to_known_minima(minima1, minima_df)   #Comparing Minima1 from Optimizer with known Minima from CSV
    min2_coords, connected2, idx_minima2 = compare_to_known_minima(minima2, minima_df)   #Comparing Minima2 from Optimizer with known Minima from CSV

    length1 = trajectory_length(traj1) #calculating total length of the trajectory for offset 1
    length2 = trajectory_length(traj2) #calculating total length of the trajectory for offset 1

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

    for path in paths:
        saddle_idx = path.get("saddle_id")
        minima_indices = path.get("minima_id", [])
        trajectories = path.get("trajectories", [])

        for min_idx, traj in zip(minima_indices, trajectories):
            df = pd.DataFrame(traj, columns=["x", "y"])
            filename = f"descent_path_S{saddle_idx}_M{min_idx}.csv"
            filepath = os.path.join("descent_paths", filename)
            df.to_csv(filepath, index=False)

def extract_connection_indices(all_results, critical_points_df, output_file="results/saddle_to_minima_indices.csv"):
    """
    From all_results, generate a CSV with index_saddle and index_minimum
    corresponding to connected pairs in the original critical_points.csv
    """
    # Create a mapping from (x1, x2) → index in critical_points.csv
    coord_to_index = {
        (round(row["x1"], 8), round(row["x2"], 8)): idx
        for idx, row in critical_points_df.iterrows()
    }

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

def main():
    # Load critical points

    all_results = []
    for idx, row in saddles_df.iterrows():
        saddle_point = [row["x1"], row["x2"]]
        results,path_data = trace_from_saddle(saddle_point, minima_df, idx)
        all_results.extend(results)
        if path_data is not None:
            paths.append(path_data)

    save_descent_paths(paths)

    minima = [
        ( (row["x1"], row["x2"]), row["f_value"] )
        for _, row in minima_df.iterrows()
    ]

    return all_results, minima, dataframe    

if __name__ == "__main__":
    main()

