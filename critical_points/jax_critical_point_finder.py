import jax
import jax.numpy as jnp

from numpy import array as nparray

from typeconfig import default_dtype

from scipy.spatial import KDTree
import pandas as pd
import os

from scipy.stats.qmc import Sobol

from jax.debug import print as jaxprint


def generate_unit_cube_design(m, d, seed):
    sampler = Sobol(d=d, scramble=True, seed=seed)
    x0s = sampler.random(m)  # numpy array, shape (m, dimension)
    return jnp.array(x0s, dtype=default_dtype)  # convert to JAX array


class JAXMinimaFinder:
    def __init__(
        self, dimension=2, min_distance=0.01, m=64, seed=42, bounds=(0.0, 1.0)
    ):

        # if bounds == (0.0, 1.0):
        #     bounds = dimension * ((0.0, 1.0),)
        # if isinstance(bounds, dict):
        #     bounds = [[v[0], v[1]] for v in bounds.values()]
        # assert len(bounds) == dimension

        # self.bounds = jnp.array(bounds, dtype=default_dtype)   # shape (dimension, 2)
        # jaxprint('self.bounds', self.bounds)
        self.low_bounds = jnp.asarray(bounds[0])
        self.upp_bounds = jnp.asarray(bounds[1])
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
        self.x0s = generate_unit_cube_design(self.m, self.dimension, self.seed)
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
            print(
                f"Too close to existing point {idx} (distance {distances[0]:.6f} < {self.min_distance})"
            )
            self.minima_counts[idx] += 1
            self.total_converged += 1
            return True
        return False

    def add_minimum(self, point, value):
        point = jnp.array(point, dtype=default_dtype)
        if self._is_too_close(point):
            return False
        else:
            point_np = nparray(point)  # store as numpy for KDTree compatibility
            print(
                f"New critical point found at {point_np.round(4)} with value {value:.6f}"
            )
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
        nlen = len(str(self.dimension))
        for _, row in df.iterrows():
            coord_labels = ["x" + str(i + 1).zfill(nlen) for i in range(self.dimension)]
            coords = [row[lbl] for lbl in coord_labels]
            value = row["f_value"]
            point = jnp.array(coords, dtype=default_dtype)
            point_np = nparray(point)
            if not self._is_too_close(point_np):
                self.minima.append((point_np, value))
                self.minima_counts.append(0)
                loaded += 1
        self.update_kdtree()
        print(f"Loaded {loaded} valid points from dataframe.")

    def get_basin_stats(self):
        """
        Returns a list of (minimum_point, count) and the total attempts.
        """
        return list(zip(self.minima, self.minima_counts)), self.m


# JAX gradient descent
def run_local_search(params, optimizer, optim_step, **optimizer_kwargs):
    final_point, final_val, path, conv = optimizer(
        params, optim_step, **optimizer_kwargs
    )
    return final_point, final_val, path, conv


def dict_to_ravelled_array(p: any, d: dict) -> jnp.ndarray:
    """
    Build a list from dict `d` whose values are ordered to match the leaf order
    produced by ravel_pytree(p).

    Args:
        p: A JAX pytree whose leaves define the target structure.
        d: A dict mapping path-name strings to scalar values.
           Keys must be produced by `jax.tree_util.keystr(path)` for each path in the pytree).

    Returns:
        A list with length as ravel_pytree(p)[0], with values drawn from `d` in leaf order.
    """
    segments = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(p):
        key = jax.tree_util.keystr(path, simple=True, separator=".")
        if key not in d.keys():
            continue
        # value = jnp.asarray(d[key])
        value = d[key]
        segments.append(value)

    if type(value) is str:
        return segments
    else:
        return jnp.asarray(segments, dtype=default_dtype)


def flat_loss_grad_hess_fns(loss_func, input):
    def flat_loss(flat_params):
        return loss_func(flat_params, input)

    grad_fn = jax.grad(flat_loss)
    hessian_fn = jax.hessian(flat_loss)
    return flat_loss, grad_fn, hessian_fn


def flatten_bound(bounds, params, which=0):
    # build bounds in the same order as the params dict
    bounds = {k: bounds[k][which] for k in bounds.keys()}
    flat_bounds = dict_to_ravelled_array(params, bounds)
    return flat_bounds


def critical_point_index_fn(hessian_fn, tol=1e-9):
    """Returns function that computes CP index for classification."""

    @jax.jit
    def critical_point_index(final_point):
        hessian = hessian_fn(final_point)
        eigvals = jnp.linalg.eigvalsh(hessian)
        near_zero = ((eigvals > -tol) & (eigvals < tol)).any()
        index = (eigvals < -tol).sum()
        return jnp.where(near_zero, -1, index)

    return critical_point_index


def basin_counts_df(finder):
    rows = []
    conv = finder.total_converged
    for i, ((pt, fval), count) in enumerate(zip(finder.minima, finder.minima_counts)):
        row = {"min_id": i, "count": int(count), "volume": count / conv}
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ---- Main search ----
from optimizers import (
    optimize_lbfgs,
    optimize_gd,
    lbfgs_step,
    optimize_newton,
    newton_step,
)

from copy import deepcopy


def jax_find_critical_points(
    model_builder,
    loss_func,
    input,
    bounds,
    symmetry=None,
    minima_only=False,
    dimension=2,
    num_attempts=64,
    min_distance=0.01,
    seed=42,
    resume_df=None,
    skip_points=None,
    outfile=None,
    **optimizer_kwargs,
):
    """
    Finds critical points using JAX.
    """

    minima, maxima, saddles = [], [], []

    # flatten params and compute derivatives:
    flat_loss, grad_fn, hessian_fn = flat_loss_grad_hess_fns(loss_func, input)

    critical_point_index = critical_point_index_fn(hessian_fn)

    # manage bounds:
    key = jax.random.key(seed)
    rand_x = jax.random.uniform(key, shape=(dimension,), dtype=default_dtype)
    params0 = model_builder(rand_x)  # model_builder returns active params pytree
    low_bounds = flatten_bound(bounds, params0, which=0)
    upp_bounds = flatten_bound(bounds, params0, which=1)

    optimizer_kwargs["bounds"] = {"low": low_bounds, "up": upp_bounds}

    # initialise critical point manager
    finder = JAXMinimaFinder(
        dimension=dimension,
        min_distance=min_distance,
        m=num_attempts,
        seed=seed,
        bounds=(low_bounds, upp_bounds),
    )

    # load previously found critical point if given
    if resume_df is not None:
        finder.load_from_dataframe(resume_df)
        print(f"Resuming with {len(finder.minima)} loaded points.")

    # set up x symmetries
    if symmetry is not None:
        symmetrize_fn = symmetry

    if skip_points is None:
        skip_points = lambda i: False

    if minima_only == "gd":
        optimizer = optimize_gd
        # optim_step = lbfgs_step(flat_loss, grad_fn=grad_fn, symmetry_fn=symmetrize_fn)
    if minima_only == "lbfgs":
        optimizer = optimize_lbfgs
        # optim_step = lbfgs_step(flat_loss, grad_fn=grad_fn, symmetry_fn=symmetrize_fn)
    if not minima_only:
        optimizer = optimize_newton
        # optim_step = newton_step(flat_loss, grad_fn, hessian_fn, symmetrize_fn)

    for i, x0 in enumerate(finder.x0s):
        if skip_points(i):
            continue
        print()
        print(
            "__________________________________________________________________________________________"
        )
        print(f"Attempt {i+1}/{finder.m}:")
        print(f"Starting point: {x0.round(4)}")

        if minima_only == "gd":
            optim_step = lbfgs_step(
                flat_loss, grad_fn=grad_fn, symmetry_fn=symmetrize_fn
            )
        if minima_only == "lbfgs":
            optim_step = lbfgs_step(
                flat_loss, grad_fn=grad_fn, symmetry_fn=symmetrize_fn
            )
        if not minima_only:
            optim_step = newton_step(flat_loss, grad_fn, hessian_fn, symmetrize_fn)

        # run optimizer
        final_point, final_val, path, conv = run_local_search(
            x0, optimizer, optim_step, **optimizer_kwargs
        )

        # check if final_point is new
        critical = False
        if conv:
            critical = finder.add_minimum(final_point, float(final_val))

        # classify critical point
        if critical:
            index = critical_point_index(final_point)

            if index == 0:
                minima.append((final_point, final_val, index))
                print("Type: minimum")
            elif index == dimension:
                maxima.append((final_point, final_val, index))
                print("Type: maximum")
            else:
                saddles.append((final_point, final_val, index))
                print("Type: saddle")

            if outfile is not None:
                results_df = critical_points_to_df(
                    minima, maxima, saddles, dimension=dimension
                )
                results_df = pd.concat((resume_df, results_df), ignore_index=True)
                results_df.to_csv(outfile, float_format="%.10f")

    # if searching minima only, compute minima and basin stats:
    basin_counts = None
    if minima_only != False:
        print("\n=== Basin Stats ===")
        basin_stats, total_attempts = finder.get_basin_stats()
        print(f"Total distinct minima found: {len(basin_stats)}")
        print(f"Total converged attempts (including duplicates): {total_attempts}")
        print("Convergence count per minimum:")
        for j, count in enumerate(finder.minima_counts):
            print(
                f"Min {j} was hit {count} times, volume share is {count/total_attempts}"
            )
        basin_counts = basin_counts_df(finder)

    return minima, maxima, saddles, basin_counts


def critical_points_to_df(minima, maxima, saddles, dimension=2):
    """Store to dataframe and write to csv."""
    data = []
    nlen = len(str(dimension))
    for point_type, points in zip(
        ["minimum", "maximum", "saddle"], [minima, maxima, saddles]
    ):
        for p, fval, index in points:
            vardict = {
                "x" + str(i + 1).zfill(nlen): float(p[i]) for i in range(dimension)
            }
            data.append(
                {
                    **vardict,
                    "f_value": float(fval),
                    "type": point_type,
                    "cp_index": index,
                }
            )

    df = pd.DataFrame(data)

    return df
