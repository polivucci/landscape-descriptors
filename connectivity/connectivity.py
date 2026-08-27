import os

from numpy.linalg import norm as numpynorm
from numpy import array
from numpy import float64 as numpyfloat64
import pandas as pd

from optimizers import optimize_lbfgs, lbfgs_step, optimize_gd, line_search_monotonic
from critical_points.jax_critical_point_finder import (run_local_search, 
                                                       flat_loss_grad_hess_fns, 
                                                       flatten_bound)

import jax.numpy as jnp
from typeconfig import default_dtype

def offset_near_saddle(saddle_point, hessian, loss_fn, epsilon=0.01, monotonicity_check=0):
    """
    Offset the saddle point along its unstable directions (eigval < 0), 
    scaled by radius of curvature (1 / |eigval|). 
    """


    # eigendecomposition
    # eigvals, eigvecs = jnp.linalg.eig(hessian)
    eigvals, eigvecs = jnp.linalg.eigh(hessian)

    # get negative eigenvalues and their eigenvectors (unstable directions)
    unstable_mask = eigvals < 0
    unstable_eigval = eigvals[unstable_mask]
    unstable_direction = eigvecs[:, unstable_mask]

    # radius of curvature = 1 / |λ|
    radius_of_curvature = 1.0 / jnp.abs(unstable_eigval)

    # offset distance = ε * radius_of_curvature
    offset_distance = (epsilon * radius_of_curvature)[jnp.newaxis, :]  # unsqueeze(0)

    # offset along unstable directions
    offsets = jnp.concatenate(
        (+ offset_distance * unstable_direction,
         - offset_distance * unstable_direction),
        axis=1
    ).T

    # point = saddle_point[jnp.newaxis, ...]  # unsqueeze(1)

    # verify monotonic
    if monotonicity_check>0:
        offsets_mono = []
        for oj, offset in enumerate(offsets):
            print(f'Offset {oj} is monotonic.')
            _, offset = line_search_monotonic(saddle_point, 
                                              offset, 
                                              loss_fn, 
                                              res=monotonicity_check)
            offsets_mono.append(offset)
    else:
        offsets_mono = offsets

    # offset along unstable directions
    offset_points = jnp.stack(
        [saddle_point + om for om in offsets_mono],
        axis=0
    )

    return offset_points

def compare_to_known_minima(min_point, minima_df, threshold=1e-3):
    """ Find closest known minimum and check if within threshold.
    Uses NumPy because of Pandas.
    """
    if len(minima_df)==0:
        return None, False, None
    else:
        x_cols = sorted([col for col in minima_df.columns if col.startswith("x")])
        dists = minima_df.apply(
            lambda r: numpynorm(array(min_point) - r[x_cols].values.astype(numpyfloat64)),
            axis=1
        )
        closest_idx = dists.idxmin()
        return minima_df.loc[closest_idx][x_cols].tolist(), dists.min() < threshold, closest_idx

# TODO: do trajectory lengths without numpy
# def trajectory_length(traj):
#     """
#     Get Total length of the trajectory for the actual Descent path
#     """
#     return sum(numpynorm(traj[i] - traj[i-1]) for i in range(1, len(traj)))

def trace_from_saddle(saddle_point, 
                      minima_df, 
                      idx_saddle, 
                      cp_index,
                      opt_algo,
                      optim_step, 
                      loss_fn,
                      hess_fn,
                      threshold=1e-2, 
                      offset=1e-2,
                      monotonicity_check=0,
                      log_paths=False, 
                      **opt_kwargs): 
    """
    Given a saddle point, trace descent on both sides and match to known minima
    """
    
    saddle_value = loss_fn(saddle_point)
    hessian = hess_fn(saddle_point)

    # create offset points (along unstable )
    offsets = offset_near_saddle(saddle_point, hessian, loss_fn, epsilon=offset, monotonicity_check=monotonicity_check) 

    assert offsets.shape[0]==2*cp_index

    result=[]
    connected_minimizers = []
    trajectories = []
    minima_id =[]
    for io, offset_coords in enumerate(offsets):
        #Optimzation to find Minima of a Saddle point and Trajectory for offset 1 
        print(f'\t Offset {io}:', offset_coords.round(5))
        minima1, arrival1_val, traj1, conv = run_local_search(offset_coords, 
                                                              opt_algo, 
                                                              optim_step, 
                                                              log_paths=log_paths, 
                                                              **opt_kwargs)

        # length1 = trajectory_length(traj1[0]) #calculating total length of the trajectory for offset 1

        if conv==True:
            min1_coords, connected1, idx_minima1 = compare_to_known_minima(minima1, 
                                                                           minima_df, 
                                                                           threshold=threshold)   
            if min1_coords is not None: min1_coords = tuple(float(x) for x in min1_coords)
            
            #Making table for results found
            result.append(
                {
                    "saddle_point": tuple(saddle_point.tolist()),
                    "offset" : io, 
                    "saddle_value": saddle_value,
                    "descent": minima1, 
                    "min_value": arrival1_val,
                    "minimizer": min1_coords,
                    "converged": "yes",
                    # "trajectory_length": length1,
                    "is_connected": "yes" if connected1 else "no"
                }
            ) 
            if connected1:
                print('Connected to minimum', idx_minima1)
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

def save_descent_paths(paths, out_dir='./'):
    """
    Save descent paths to individual CSV files:
    Format: descent_path_S{saddle_index}_M{min_index}.csv
    Each file contains a trajectory: x, y coordinates
    """
    os.makedirs(out_dir, exist_ok=True)
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
            filepath = os.path.join(out_dir, filename)
            df.to_csv(filepath, index=False)

def extract_connection_indices(all_results, critical_points_df):
    """
    From all_results, generate a CSV with index_saddle and index_minimum
    corresponding to connected pairs in the original critical_points.csv
    """
    
    # Create a mapping from (x1, x2)or(x1, x2, x3) → index in critical_points.csv
    x_cols = sorted([col for col in critical_points_df.columns if col.startswith("x")])

    coord_to_index = {
        tuple(round(row[col], 8) for col in x_cols): idx
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

    df = pd.DataFrame(connection_rows, columns=["index_1", "index_2"])
    
    return df

def trace_connectivity(saddles_df, minima_df, 
                       func,
                       model_builder,
                       input, 
                       bounds,
                       symmetry=None,
                       threshold=1e-2, 
                       offset=1e-2,
                       optimizer='lbfgs',
                       **opt_kwargs):
    """
    Trace connectivity from saddle points to minima.

    Parameters:
        saddles_df (pd.DataFrame): DataFrame of saddle points.
        minima_df (pd.DataFrame): DataFrame of minima.
        func (callable): The target function, e.g., schwefel(x[0], x[1]).
        return_paths (bool): Whether to return full descent path data.
        nn_model(Function); any given model

    Returns:
        all_results: list of saddle-to-minima connections.
        minima: list of minima as (coords, f_value) tuples.
        paths (optional): list of descent path data, if return_paths=True.
    """
    
    flat_loss, grad_fn, hess_fn = flat_loss_grad_hess_fns(func, input)

    # set up bounds:
    idx, pms = list(saddles_df.iterrows())[0]
    pms = [pms[col] for col in sorted(pms.index) if col.startswith("x")]
    pms = jnp.array(pms, dtype=default_dtype)
    params0 = model_builder(pms) # model_builder returns active params pytree
    low_bounds = flatten_bound(bounds, params0, which=0)
    upp_bounds = flatten_bound(bounds, params0, which=1)
    opt_kwargs['bounds'] = {'low': low_bounds, 'up': upp_bounds}

    # set up x symmetries
    if symmetry is not None:
        symmetrize_fn = symmetry

    # set up optimizer:
    if optimizer=='gd':
        opt_algo=optimize_gd
        # optim_step = lbfgs_step(flat_loss, grad_fn=grad_fn, symmetry_fn=symmetrize_fn)
    if optimizer=='lbfgs':
        opt_algo=optimize_lbfgs
        # optim_step = lbfgs_step(flat_loss, grad_fn=grad_fn, symmetry_fn=symmetrize_fn)

    monotonicity_res = 0
    if 'monotonicity_res' in opt_kwargs.keys(): monotonicity_res=opt_kwargs.pop('monotonicity_res')

    all_results = []
    paths=[]
    for idx, row in saddles_df.iterrows():
        cpidx = row['cp_index']
        print() 
        print('Tracing saddle', idx, f'(index {cpidx})')

        saddle_point = [row[col] for col in sorted(row.index) if col.startswith("x")]
        saddle_point = jnp.array(saddle_point, dtype=default_dtype)

        # set up step:
        if optimizer=='gd':
            optim_step = lbfgs_step(flat_loss, grad_fn=grad_fn, symmetry_fn=symmetrize_fn, monotonicity_res=monotonicity_res)
        if optimizer=='lbfgs':
            optim_step = lbfgs_step(flat_loss, grad_fn=grad_fn, symmetry_fn=symmetrize_fn, monotonicity_res=monotonicity_res)

        # minimize from saddle
        results, path_data = trace_from_saddle(saddle_point, 
                                               minima_df, 
                                               idx, 
                                               cpidx,
                                               opt_algo, 
                                               optim_step,
                                               flat_loss,
                                               hess_fn,
                                               threshold=threshold, 
                                               offset=offset,
                                               monotonicity_check=monotonicity_res,
                                               **opt_kwargs)
        all_results.extend(results)
        if path_data is not None:
            paths.append(path_data)

    descent_points = [[result["descent"], result["min_value"]] for result in all_results if result["converged"]=="yes"]

    minima = [
        ([row[col] for col in sorted(row.index) if col.startswith("x")], row["f_value"])
        for _, row in minima_df.iterrows()
    ]

    return all_results, minima, descent_points, paths