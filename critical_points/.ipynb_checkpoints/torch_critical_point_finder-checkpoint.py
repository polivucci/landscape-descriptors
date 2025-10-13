"""
critical_points/torch_critical_point_finder.py
----------------------------------------------
PyTorch port of numpy_critical_point_finder.py.
Finds critical points (minima, maxima, saddles) of a scalar loss function
represented as a torch.nn.Module (e.g. Schwefel2D).

Key differences:
- Uses torch.autograd instead of numpy gradients.
- Uses torch.quasirandom.SobolEngine instead of scipy Sobol sampler.
- Uses LBFGS or Adam optimizers for descent.
- Still relies on scipy.spatial.KDTree for duplicate detection.
"""

import torch
import torch.nn as nn
from torch.optim import LBFGS
from torch.quasirandom import SobolEngine
import numpy as np
from scipy.spatial import KDTree
import pandas as pd


# ============================================================
#  Utility functions
# ============================================================

def squared_grad_norm(x: torch.Tensor, func):
    """
    Computes ||∇f(x)||² for given callable func(x)->scalar.
    """
    x = x.clone().detach().requires_grad_(True)
    y = func(x)
    g = torch.autograd.grad(y, x, create_graph=True)[0]
    return torch.sum(g * g)


def critical_point_index(hessian: np.ndarray, tol=1e-9):
    """
    Computes index (# negative eigenvalues) of Hessian.
    Returns -1 if degenerate (|eig|<tol).
    """
    eigvals = np.linalg.eigvals(hessian)
    neg_count = np.sum(eigvals < -tol)
    zero = np.logical_and(eigvals > -tol, eigvals < tol)
    if np.any(zero):
        return -1
    return int(neg_count)


def classify_point(x: torch.Tensor, func, tol=1e-9):
    """
    Classifies a critical point by Hessian eigenvalues:
    returns ('minimum'/'maximum'/'saddle'/'degenerate', index)
    """
    def wrapped(z):
        return func(z)

    H = torch.autograd.functional.hessian(wrapped, x)
    H_np = H.detach().cpu().numpy()
    idx = critical_point_index(H_np, tol)
    d = x.numel()
    if idx == -1:
        return "degenerate", idx
    if idx == 0:
        return "minimum", idx
    if idx == d:
        return "maximum", idx
    return "saddle", idx


# ============================================================
#  Optimization via PyTorch (replaces scipy minimize)
# ============================================================

def optimize_lbfgs(start_point: torch.Tensor, func, max_iter=100, atol=1e-6, rtol=1e-5):
    """
    Minimize squared gradient norm ||∇f||² using torch LBFGS.
    start_point : torch tensor (D,)
    func : callable x->scalar
    Returns (final_point, loss_value)
    """
    x = start_point.clone().detach().requires_grad_(True)
    optimizer = LBFGS([x], max_iter=20, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = squared_grad_norm(x, func)
        loss.backward()
        return loss

    prev_x = x.detach().clone()
    for i in range(max_iter):
        optimizer.step(closure)
        try:
            torch.testing.assert_close(x.detach(), prev_x, atol=atol, rtol=rtol)
            break
        except AssertionError:
            prev_x = x.detach().clone()
            continue

    final_val = squared_grad_norm(x, func).item()
    return x.detach(), final_val


# ============================================================
#  Torch version of MinimaFinder
# ============================================================

class TorchMinimaFinder:
    def __init__(self, bounds, dimension=2, min_distance=0.01, m=64):
        self.bounds = torch.tensor(bounds, dtype=torch.float64)
        self.dimension = dimension
        self.min_distance = min_distance
        self.m = m
        self.minima = []
        self.attempt_history = []
        self.generate_starting_points()

    def generate_starting_points(self):
        """Sobol sampling in [bounds[0], bounds[1]]^d"""
        log2m = int(np.rint(np.log2(self.m)))
        sampler = SobolEngine(dimension=self.dimension, scramble=True)
        pts = sampler.draw(2 ** log2m).to(dtype=torch.float64)
        lo, hi = self.bounds
        self.x0s = lo + (hi - lo) * pts  # scale to given bounds

    def add_minimum(self, point, value):
        point = np.array(point)
        if not self._is_too_close(point):
            self.minima.append((point, value))
            return True
        return False

    def _is_too_close(self, point):
        if not self.minima:
            return False
        kdt = KDTree([m[0] for m in self.minima])
        d, _ = kdt.query([point], k=1)
        return d[0] < self.min_distance


# ============================================================
#  Run local search (gradient descent) using LBFGS
# ============================================================

def run_search(finder, func):
    for i in range(finder.m):
        x0 = finder.x0s[i]
        finder.attempt_history.append(x0.numpy())
        try:
            point, val = optimize_lbfgs(x0, func)
            finder.add_minimum(point.numpy(), val)
        except Exception as e:
            print(f"Failed at attempt {i}: {e}")
            continue
    return finder.minima


# ============================================================
#  High-level critical point finder
# ============================================================

def find_critical_points_torch(
    func, bounds=(-1.0, 1.0),
    num_attempts=64, dimension=2, min_distance=0.1,
    known_minima=None
):
    """
    PyTorch version of find_critical_points.
    func : callable x->scalar tensor
    """

    finder = TorchMinimaFinder(bounds, dimension, min_distance, num_attempts)
    results = run_search(finder, func)

    critical_points, minima, maxima = [], [], []

    for point, val in results:
        if val < 1e-9:  # ||∇f||² ≈ 0
            x_t = torch.tensor(point, dtype=torch.float64)
            if known_minima:
                too_close = any(np.linalg.norm(point - np.array(m[0])) < min_distance/2
                                for m in known_minima)
                if too_close:
                    continue
            ptype, idx = classify_point(x_t, func)
            if ptype == "minimum":
                minima.append((point, val, idx))
            elif ptype == "maximum":
                maxima.append((point, val, idx))
            else:
                critical_points.append((point, val, idx))
            print(ptype, point)

    return minima, maxima, critical_points


# ============================================================
#  Test with toy function
# ============================================================

if __name__ == "__main__":
    # Polynomial test: f(x,y) = (x^2 - eps)^2 + y^2
    def f_torch(x, eps=0.1):
        x1, x2 = x[0], x[1]
        return (x1 ** 2 - eps) ** 2 + x2 ** 2

    BOUNDS = (0, 1)
    DIMENSION = 2
    ATTEMPTS = 128
    MIN_DISTANCE = 0.05

    minima, maxima, saddles = find_critical_points_torch(
        f_torch,
        bounds=BOUNDS,
        dimension=DIMENSION,
        num_attempts=ATTEMPTS,
        min_distance=MIN_DISTANCE,
    )

    print(f"\nFound {len(minima)} minima, {len(maxima)} maxima, {len(saddles)} saddles.")

    # Output CSV
    f_values = []
    for ptype, plist in zip(['minimum', 'saddle'], [minima, saddles]):
        for (x, val, idx) in plist:
            f_values.append({
                'x1': x[0],
                'x2': x[1],
                'f_value': f_torch(torch.tensor(x)).item(),
                'type': ptype,
                'index': idx
            })

    df = pd.DataFrame(f_values)
    print(df)
    df.to_csv(f"critical_points_torch_{DIMENSION}D.csv", float_format='%.10f', index=False)
