import torch
import pandas as pd
import numpy as np
from torch.optim import LBFGS
from schwefel_function import schwefel  # Replace with your actual import
from plot_results import plot_results_with_paths

torch.set_default_dtype(torch.float64)

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

    def closure(): #Closure required because LBFGS evaluates the function multiple times during each iteration.
        optimizer.zero_grad()
        loss = func(x)
        loss.backward()
        return loss
    
    optimizer.step(closure)  # Triggers one full optimization run using the closure

    return x.detach()


def compare_to_known_minima(min_point, minima_df, threshold=1e-3):
    """
    Find closest known minimum and check if within threshold
    """
    dists = minima_df.apply(lambda r: np.linalg.norm(min_point.numpy() - np.array([r["x1"], r["x2"]])), axis=1)
    closest_idx = dists.idxmin()
    return minima_df.loc[closest_idx][["x1", "x2"]].tolist(), dists.min() < threshold

def trace_from_saddle(saddle_point, minima_df):
    """
    Given a saddle point, trace descent on both sides and match to known minima
    """
    saddle_point = torch.tensor(saddle_point, dtype=torch.float64)

    offset_p1, offset_p2 = offset_near_saddle(saddle_point) #gets both offset point with curvature radius 

    func = lambda x: schwefel(x[0], x[1])

    minima1 = optimize_lbfgs(offset_p1, func)   #Optimzation to find Minima of a Saddle point offset 1
    minima2 = optimize_lbfgs(offset_p2, func)   #Optimzation to find Minima of a Saddle point offset 2

    min1_coords, connected1 = compare_to_known_minima(minima1, minima_df)   #Comparing Minima1 from Optimizer with known Minima from CSV
    min2_coords, connected2 = compare_to_known_minima(minima2, minima_df)   #Comparing Minima2 from Optimizer with known Minima from CSV

    #Making table for results found
    result = [ 
        {
            "saddle_point": tuple(saddle_point.tolist()),
            "descent":tuple(float(x) for x in minima1), #Undo Comment if you want to check what minima does the saddle point get after going through LBFG optimizer  
            "minimizer": tuple(float(x) for x in min1_coords),
            "is_connected": "yes" if connected1 else "no"
        },
        {
            "saddle_point": tuple(saddle_point.tolist()),
            "descent":tuple(float(x) for x in minima2), #Undo Comment if you want to check what minima does the saddle point get after going through LBFG optimizer  
            "minimizer": tuple(float(x) for x in min2_coords),
            "is_connected": "yes" if connected2 else "no",
        }
    ]
    # Storing path data for visualizing further
    connected_minimizers = []
    if connected1:
        connected_minimizers.append(tuple(minima1.tolist()))
    if connected2:
        connected_minimizers.append(tuple(minima2.tolist()))
    path_data = None
    # print(connected_minimizers)
    if connected_minimizers:
        path_data = {
            "saddle": tuple(saddle_point.tolist()),
            "minimizers": connected_minimizers
        }

    return result,path_data

paths=[]
dataframe = pd.read_csv("critical_points_schwefel.csv")
saddles_df = dataframe[dataframe["type"] == "saddle"]
minima_df = dataframe[dataframe["type"] == "minimum"]

def main():
    # Load critical points
    
    all_results = []

    for i, row in saddles_df.iterrows():
        saddle_point = [row["x1"], row["x2"]]
        results,path_data = trace_from_saddle(saddle_point, minima_df)
        all_results.extend(results)
        if path_data is not None:
            paths.append(path_data)

    # Output results
    result_df = pd.DataFrame(all_results)
    result_df.to_csv("saddle_to_minima_connections.csv", index=False)
    print(result_df.head(24))

    
if __name__ == "__main__":
    main()
    # Plot
    minima_list = [[(r["x1"], r["x2"])] for _, r in minima_df.iterrows()]
    saddle_list = [[(r["x1"], r["x2"])] for _, r in saddles_df.iterrows()]

    def schwefel_numpy(x):
        x = [torch.tensor(xi) for xi in x]
        return schwefel(*x).detach().numpy()
    plot_results_with_paths(
        func=schwefel_numpy,
        minima=minima_list,
        saddle_points=saddle_list,
        paths=paths,
        bounds=(0, 1),
        res=200,
        show_connectivity=True # kwargs suggesting whether or not to draw arrows/paths showing how saddle points connect to minima.
    )
