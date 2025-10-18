import torch
from torch.quasirandom import SobolEngine
from scipy.spatial import KDTree
import pandas as pd


class TorchMinimaFinder:
    def __init__(self, bounds, dimension=2, min_distance=0.01, m=64, device="cpu", seed=42):
        self.bounds = torch.tensor(bounds, dtype=torch.set_default_dtype(torch.float64), device=device)
        self.dimension = dimension
        self.min_distance = min_distance
        self.m = m
        self.minima = []
        self.attempt_history = []
        self.kdtree = None
        self.device = device
        self.seed = seed
        self.generate_starting_points()

    def generate_starting_points(self):
        sampler = SobolEngine(dimension=self.dimension, scramble=True, seed=self.seed)
        self.x0s = sampler.draw(self.m).to(self.device)

    def update_kdtree(self):
        if self.minima:
            self.kdtree = KDTree([m[0] for m in self.minima])

    def _is_too_close(self, point):
        if not self.minima:
            return False
        if self.kdtree is None:
            self.update_kdtree()
        distances, _ = self.kdtree.query([point], k=1)
        return distances[0] < self.min_distance
    def _is_out_bounds(self, point, tolerance=1e-6):
        """
        Check if point is outside the defined bounds (with optional tolerance).
        `point` should be a 1D torch tensor or numpy array.
        """
        lower, upper = self.bounds
        return torch.any(point < lower - tolerance) or torch.any(point > upper + tolerance)

    def add_minimum(self, point, value):
        point_np = point.detach().cpu().numpy()
        print(point_np, "Reject:", self._is_too_close(point_np) or bool(self._is_out_bounds(point)))

        if self._is_out_bounds(point):
            print(f"Rejected: Out of bounds -> {point.tolist()}")
            return False
        if not self._is_too_close(point_np):
            self.minima.append((point_np, value))
            self.update_kdtree()
            return True
        return False

# Torch gradient descent
from connectivity.connectivity import optimize_lbfgs
def run_local_search_torch(model, input, loss_fn, lr=1e-3, steps=100):
    fin_mod, path = optimize_lbfgs(model, input, loss_fn, lr=lr, max_iter=steps)
    fin_point = path[-1][0]
    fin_loss = path[-1][-1]
    return fin_point, fin_loss

# # ---- Torch gradient descent (replaces minimize) ----
# def run_local_search_torch(x0, loss_fn, lr=1e-3, steps=100):

#     x = x0.clone().detach().requires_grad_(True)
#     optimizer = torch.optim.LBFGS([x], lr=lr, max_iter=steps, line_search_fn='strong_wolfe')

#     def closure():
#         optimizer.zero_grad()
#         loss = loss_fn(x)
#         loss.backward()
#         return loss

#     optimizer.step(closure)
#     final_loss = loss_fn(x).item()
#     return x.detach(), final_loss

# Critical point index using Hessian
def _torch_critical_point_index(hessian, tol=1e-9):
    eigvals = torch.linalg.eigvalsh(hessian)
    if ((eigvals > -tol) & (eigvals < tol)).any():
        return -1  # near-zero eigenvalue
    return int((eigvals < -tol).sum().item())

# ---- Main search ----
def find_critical_points_torch(model_class, loss_func, input, bounds, dimension=2,
                               num_attempts=64, min_distance=0.01,
                               device="cpu", lr=0.01, steps=300, ):
    """
    Finds critical points using torch autograd + NNModule
    """
    finder = TorchMinimaFinder(bounds, dimension, min_distance, num_attempts, device=device)

    minima, maxima, saddles = [], [], []

    for i, x0 in enumerate(finder.x0s):
        print(f"Attempt {i+1}/{finder.m}: starting point {x0.cpu().numpy()}")

        # Initialize NN module (e.g., Schwefel2D)
        model = model_class(*x0.tolist()).to(device)

        # # Define loss as squared gradient norm of the Schwefel loss
        # def squared_grad_norm(x):
        #     mod = model_class(*x).to(device)
        #     y = mod(input)  # get [x1, x2]
        #     loss = loss_func(y)
        #     grad = torch.autograd.grad(loss, y, create_graph=True)[0]
        #     return torch.sum(grad ** 2)

        # Gradient descent to minimize ||∇f||²
        final_point, final_val = run_local_search_torch(model, input, loss_func, lr=lr, steps=steps)
        finder.add_minimum(final_point, final_val)

    # Classify critical points
    for point, val in finder.minima:
        point_t = torch.tensor(point, dtype=torch.set_default_dtype(torch.float64), requires_grad=True, device=device)
        y = model_class(*point_t.tolist())(input).to(device)
        f_val = loss_func(y)
        hessian = torch.autograd.functional.hessian(lambda z: loss_func(z), y)
        index = _torch_critical_point_index(hessian)

        if index == 0:
            minima.append((point, f_val.item(), index))
            print("minimum", point)
        elif index == dimension:
            maxima.append((point, f_val.item(), index))
            print("maximum", point)
        else:
            saddles.append((point, f_val.item(), index))
            print("saddle", point)

    return minima, maxima, saddles

# ---- CSV Writer ----
def save_critical_points_to_csv(minima, saddles, dimension=2, filename="critical_points_torch.csv"):
    data = []
    for point_type, points in zip(["minimum", "saddle"], [minima, saddles]):
        for p, fval, index in points:
            vardict = {f"x{i+1}": p[i] for i in range(dimension)}
            data.append({**vardict, "f_value": fval, "type": point_type, "index": index})

    df = pd.DataFrame(data)
    df.to_csv(filename, float_format="%.10f", index=False)
    print(f"Saved {filename}")
    return df
