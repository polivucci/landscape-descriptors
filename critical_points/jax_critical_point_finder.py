import jax
import jax.numpy as jnp

from numpy import array as nparray

from typeconfig import default_dtype

from scipy.spatial import KDTree
import pandas as pd
from memory_profiler import profile
import os

from scipy.stats.qmc import Sobol


class JAXMinimaFinder:
    def __init__(self, bounds=(0.0, 1.0), dimension=2, min_distance=0.01, m=64, seed=42):

        if bounds == (0.0, 1.0):
            bounds = dimension * ((0.0, 1.0),)
        if isinstance(bounds, dict):
            bounds = tuple(bounds.values())
        assert len(bounds) == dimension

        self.bounds = jnp.array(bounds, dtype=default_dtype)   # shape (dimension, 2)
        self.low_bounds = self.bounds[:, 0]
        self.upp_bounds = self.bounds[:, 1]
        self.ranges = self.upp_bounds - self.low_bounds

        self.dimension = dimension
        self.min_distance = min_distance
        self.m = m
        self.minima = []
        self.attempt_history = []
        self.kdtree = None
        self.kdtree_x0s = None
        self.seed = seed
        self.generate_starting_points()
        self.minima_counts = []
        self.total_converged = 0

    def generate_starting_points(self):
        sampler = Sobol(d=self.dimension, scramble=True, seed=self.seed)
        x0s = sampler.random(self.m)                      # numpy array, shape (m, dimension)
        self.x0s = jnp.array(x0s)                         # convert to JAX array
        self.x0s = self.x0s * self.ranges + self.low_bounds
        return self

    def _unit_cube(self, x):
        return (jnp.array(x, dtype=default_dtype) - self.low_bounds) / self.ranges

    def update_kdtree(self):
        if self.minima:
            self.kdtree = KDTree([self._unit_cube(m[0]) for m in self.minima])

    def _is_too_close(self, point):
        point = self._unit_cube(point)
        if not self.minima:
            return False
        if self.kdtree is None:
            self.update_kdtree()
        distances, indices = self.kdtree.query([point], k=1)
        if distances[0] < self.min_distance:
            idx = indices[0]
            print(f"Point is too close to existing minimum {idx} (distance {distances[0]:.6f} < {self.min_distance})")
            self.minima_counts[idx] += 1
            self.total_converged += 1
            return True
        return False

    def _is_out_bounds(self, point, tolerance=1e-6):
        """
        Check if point is outside the defined bounds (with optional tolerance).
        `point` should be a 1D JAX array or numpy array.
        """
        point = jnp.array(point, dtype=default_dtype)
        return jnp.any(point < self.low_bounds - tolerance) or jnp.any(point > self.upp_bounds + tolerance)

    def add_minimum(self, point, value):
        point = jnp.array(point, dtype=default_dtype)
        if self._is_out_bounds(point):
            print(f"Point {point} is out of bounds {self.bounds}. Rejecting.")
            return False
        if self._is_too_close(point):
            return False
        else:
            point_np = nparray(point)        # store as numpy for KDTree compatibility
            print(f"New critical point found at {point_np} with value {value:.6f}")
            self.minima.append((point_np, value))
            self.minima_counts.append(1)
            self.total_converged += 1
            self.update_kdtree()
            return True

    def load_from_dataframe(self, df):
        """
        Load previously found minima from a DataFrame.
        Expected columns: x1, x2, ..., f_value
        Stores the loaded points as (np.array, float) tuples in self.minima.
        """
        loaded = 0
        for _, row in df.iterrows():
            coords = [row[f'x{i+1}'] for i in range(self.dimension)]
            value = row['f_value']
            point = jnp.array(coords, dtype=default_dtype)
            point_np = nparray(point)
            if not self._is_out_bounds(point) and not self._is_too_close(point_np):
                self.minima.append((point_np, value))
                self.minima_counts.append(0)
                loaded += 1
        self.update_kdtree()
        print(f"Loaded {loaded} valid points from dataframe.")

    def get_basin_stats(self):
        """
        Returns a list of (minimum_point, count) and the total converged attempts.
        """
        return list(zip(self.minima, self.minima_counts)), self.total_converged


# JAX gradient descent
def run_local_search(params, optimizer, optim_step, **optimizer_kwargs):
    final_point, final_val, path = optimizer(params, optim_step, **optimizer_kwargs)
    return final_point, final_val, path

def dict_to_ravelled_array(p: any, d: dict) -> jnp.ndarray:
    """
    Build a flat array from dict `d` whose values are ordered to match
    the leaf order produced by ravel_pytree(p).

    Args:
        p: A JAX pytree whose leaves define the target structure.
        d: A dict mapping path-name strings → scalar or array values.
           Keys must be produced by jax.tree_util.keystr(path) for
           each path in the pytree (see `pytree_path_names` below).

    Returns:
        A 1-D array with the same total size as ravel_pytree(p)[0],
        with values drawn from `d` in leaf order.
    """
    segments = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(p):
        key = jax.tree_util.keystr(path, simple=True)          # e.g. ".w", ".b", "[0]"
        value = jnp.asarray(d[key])
        # Preserve the leaf's shape so sizes stay consistent with ravel_pytree
        if value.shape != jnp.asarray(leaf).shape:
            raise ValueError(
                f"Shape mismatch for '{key}': "
                f"pytree leaf {jnp.asarray(leaf).shape} vs dict value {value.shape}"
            )
        segments.append(value.ravel())

    return jnp.concatenate(segments)

def flatten_bound(bounds, params, which=0):
    # build bounds in the same order as the params dict
    bounds = {k: bounds[k][which] for k in bounds.keys()}
    flat_bounds = dict_to_ravelled_array(params, bounds)
    return flat_bounds

@jax.jit
def _jax_critical_point_index(hessian, tol=1e-9):
    """Returns -1 as a traced value, not a Python int for JIT-compatibility.
    """
    eigvals = jnp.linalg.eigvalsh(hessian)
    near_zero = ((eigvals > -tol) & (eigvals < tol)).any()
    index = (eigvals < -tol).sum()
    return jnp.where(near_zero, -1, index)

def save_basin_counts_csv(finder, filename="basin_counts.csv", out_dir="."):
    rows = []
    conv = finder.total_converged
    for i, ((pt, fval), count) in enumerate(zip(finder.minima, finder.minima_counts)):
        row = {"min_id": i, "count": int(count), "volume": count/conv}
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"Saved basin counts to: {path}")
    return df

# ---- Main search ----
from critical_points.jax_optimizers import optimize_lbfgs, lbfgs_step, optimize_newton, newton_step

def jax_find_critical_points(model_builder, 
                             loss_func, 
                             input, 
                             bounds, 
                             minima_only=False, 
                             dimension=2,
                             num_attempts=64, 
                             min_distance=0.01, 
                             seed=42,
                             resume_df=None, 
                             **optimizer_kwargs):
    """
    Finds critical points using JAX.
    """

    finder = JAXMinimaFinder(bounds, dimension, min_distance, num_attempts, seed=seed)
    
    if resume_df is not None:
        finder.load_from_dataframe(resume_df)
        print(f"Resuming with {len(finder.minima)} loaded points.")

    minima, maxima, saddles = [], [], []

    # model_builder now returns a params dict (all active)
    params0 = model_builder(finder.x0s[0])

    # flatten params to 1D vector 
    flat_params0, unravel_fn = jax.flatten_util.ravel_pytree(params0)
    # define flat losses
    def flat_loss(flat_params):
        params = unravel_fn(flat_params)
        return loss_func(params, input)
    grad_fn = jax.grad(flat_loss)
    hessian_fn = jax.hessian(flat_loss)

    # flatten bounds:
    low_bounds = flatten_bound(bounds, params0, which=0)
    upp_bounds = flatten_bound(bounds, params0, which=1)
    optimizer_kwargs['bounds'] = {'low': low_bounds, 'up': upp_bounds}

    optimizer=optimize_lbfgs
    optim_step = lbfgs_step(flat_loss)
    if not minima_only: 
        optimizer=optimize_newton
        optim_step = newton_step(flat_loss, grad_fn, hessian_fn)

    for i, x0 in enumerate(finder.x0s):
        # if (i+1)!=325: continue
        print()
        print('__________________________________________________________________________________________')
        print(f"Attempt {i+1}/{finder.m}:")
        print(f"Starting point: {x0}")

        # initialize model params 
        params = model_builder(x0)
        flat_params, _ = jax.flatten_util.ravel_pytree(params)
        flat_params = jnp.array(flat_params, dtype=default_dtype)

        # run optimizer
        final_point, final_val, path = run_local_search(flat_params, optimizer, optim_step, **optimizer_kwargs)
        grad = grad_fn(final_point)

        print(f"Arrival point: {final_point}")
        print('check grad:', grad)

        # check grad is small and final_point is new
        gradtol = 1e-5 or optimizer_kwargs['gradtol']
        gradnorm = jnp.linalg.norm(grad)
        critical=False
        if gradnorm<gradtol:             
            print(f"Converged to critical point with ||grad||={gradnorm:.2e} < {gradtol}")          
            critical = finder.add_minimum(final_point, float(final_val)) 

        # classify critical point
        if critical:
            hessian = hessian_fn(final_point)
            index = _jax_critical_point_index(hessian)
            
            if index == 0:
                minima.append((final_point, final_val, index))
                print("Type: minimum")
            elif index == dimension:
                maxima.append((final_point, final_val, index))
                print("Type: maximum")
            else:
                saddles.append((final_point, final_val, index))
                print("Type: saddle")

    # if searching minima only, compute minima and basin stats:
    if minima_only==True:
        print("\n=== Basin Stats ===")
        basin_stats, total_converged = finder.get_basin_stats()
        print(f"Total new distinct minima found: {len(basin_stats)}")
        print(f"Total converged attempts (including duplicates): {finder.minima_counts}")
        print(f"Converged to existing basin (duplicates): {total_converged}")
        print("Convergence count per minimum:")
        for (point, _), count in zip(finder.minima, finder.minima_counts):
            print(f"Min at {point} was hit {count} times, volume share is {count/total_converged}")
        save_basin_counts_csv(finder, filename="basin_counts.csv", out_dir=".")

    return minima, maxima, saddles


def save_critical_points_to_csv(minima, maxima, saddles, dimension=2, filename="critical_points_jax.csv"):
    '''Store to dataframe and write to csv.'''
    data = []
    for point_type, points in zip(["minimum", "maximum", "saddle"], [minima, maxima, saddles]):
        for p, fval, index in points:
            vardict = {f"x{i+1}": float(p[i]) for i in range(dimension)}
            data.append({**vardict, "f_value": float(fval), "type": point_type, "index": index})

    df = pd.DataFrame(data)
    df.to_csv(filename, float_format="%.10f")
    print(f"Saved {filename}")
    return df
