import torch
from torch.quasirandom import SobolEngine
from scipy.spatial import KDTree
import pandas as pd
from memory_profiler import profile

torch.set_default_dtype(torch.float64)

class TorchMinimaFinder:
    def __init__(self, bounds=(0.0, 1.0), dimension=2, min_distance=0.01, m=64, device="cpu", seed=42):
        if bounds==(0.0, 1.0): bounds = dimension*((0.0, 1.0),)
        if isinstance(bounds, dict): bounds = tuple(bounds.values())
        assert len(bounds)==dimension
        self.bounds = torch.as_tensor(bounds, device=device, dtype=torch.get_default_dtype())
        self.low_bounds, self.upp_bounds = self.bounds[:,0], self.bounds[:,1]
        self.ranges = self.upp_bounds - self.low_bounds
        
        self.dimension = dimension
        self.min_distance = min_distance
        self.m = m
        self.minima = []
        self.attempt_history = []
        self.kdtree = None
        self.kdtree_x0s = None
        self.device = device
        self.seed = seed
        self.generate_starting_points()

        self.minima_counts = []
        self.total_converged = 0

    def generate_starting_points(self):
        sampler = SobolEngine(dimension=self.dimension, scramble=True, seed=self.seed)
        self.x0s = sampler.draw(self.m).to(self.device)
        self.x0s *= self.ranges
        self.x0s += self.low_bounds
        # self.kdtree_x0s = KDTree([x0 for x0 in self.x0s])

    def _unit_cube(self, x):
        return (x-self.low_bounds.numpy())/self.ranges.numpy()

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
            self.minima_counts[idx] += 1      # Increment count for this minimum
            self.total_converged += 1         # Increment total converged
            return True
        return False
    
    def _is_out_bounds(self, point, tolerance=1e-6):
        """
        Check if point is outside the defined bounds (with optional tolerance).
        `point` should be a 1D torch tensor or numpy array.
        """
        return torch.any(point < self.low_bounds - tolerance) or torch.any(point > self.upp_bounds + tolerance)

    def add_minimum(self, point, value):
        point_np = point.detach().cpu().numpy()
        print("Reject:", self._is_too_close(point_np) or bool(self._is_out_bounds(point)))
        if self._is_out_bounds(point):
            return False
        if self._is_too_close(point_np):
            return False
        if not self._is_too_close(point_np):
            self.minima.append((point_np, value))
            self.minima_counts.append(1)
            self.update_kdtree()
            return True
        return False
    
    def load_from_dataframe(self, df):
        """
        Load previously found minima from a DataFrame.
        Expected columns: x1, x2, ..., f_value
        Stores the loaded points as (torch.Tensor, float) tuples in self.minima.
        """
        loaded = 0
        for _, row in df.iterrows():
            coords = [row[f'x{i+1}'] for i in range(self.dimension)]
            value = row['f_value']
            point = torch.tensor(coords, dtype=torch.float32, device=self.device)
            if not self._is_out_bounds(point) and not self._is_too_close(point.cpu().numpy()):
                self.minima.append((point, value))
                self.minima_counts.append(0)
                loaded += 1
        self.update_kdtree()
        print(f"Loaded {loaded} valid minima from dataframe.")

    def get_basin_stats(self):
        """
    Returns a list of (minimum_point, count) and the total converged attempts.
    """
        return list(zip(self.minima, self.minima_counts)), self.total_converged

# Torch gradient descent
from critical_points.optimizers import optimize_lbfgs, optimize_newton
def run_local_search(optimizer, model, input, loss_fn, **optimizer_kwargs):
    _, final_val, path = optimizer(model, input, loss_fn, **optimizer_kwargs)
    # try: 
    #     if optimizer_kwargs['log_paths']: print(pd.DataFrame(path).tail(10))
    # except: 
    #     pass
    fin_point = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    return fin_point, final_val, path

def flatten_hessian_blocks(H):
    """Written by ChatGPT.
    """
    rows = []
    for row in H:
        # ensure each block is at least 2D
        row_blocks = [
            h.unsqueeze(0).unsqueeze(1) if h.ndim == 0 else  # scalar → (1,1)
            h.unsqueeze(0) if h.ndim == 1 else               # vector → (1,n)
            h for h in row                                   # matrix → keep as is
        ]
        rows.append(torch.cat(row_blocks, dim=1))
    return torch.cat(rows, dim=0)

# Critical point index using Hessian
def _torch_critical_point_index(hessian, tol=1e-9):
    """Computes the index of a critical point.
    """
    eigvals = torch.linalg.eigvalsh(hessian)
    if ((eigvals > -tol) & (eigvals < tol)).any():
        return -1  # near-zero eigenvalue
    return int((eigvals < -tol).sum().item())

def identity_loss(y):
    """
    Dummy loss needed for compatibility in the ||∇f||² method.
    """
    return y

def construct_SquaredGradModel(base_model_class, base_loss):

    class SquaredGradModel(base_model_class):
        """Helper model that returns the log squared gradient norm of base model + base loss.
        """

        def forward(self, input):
            y = super().forward(input)
            loss = base_loss(y)
            loss.backward(create_graph=True) # compute grads and retain graph
            grad2 = [p.grad.clone()**2 for p in self.parameters() if p.requires_grad] 
            return sum(grad2)
            # return torch.log10(sum(grad2))

    return SquaredGradModel

# ---- Main search ----
from memory_profiler import profile

def comp_hessian(func, params):
        return torch.func.hessian(func)(params)

# @profile
def find_critical_points_torch(model_builder, loss_func, input, bounds, minima_only=False, dimension=2,
                               num_attempts=64, min_distance=0.01, seed=42,
                               device="cpu",resume_df=None, **optimizer_kwargs):
    """
    Finds critical points using torch autograd + NNModule
    """
    finder = TorchMinimaFinder(bounds, dimension, min_distance, num_attempts, seed=seed, device=device)
    if resume_df is not None:
        finder.load_from_dataframe(resume_df)
        print(f"Resuming with {len(finder.minima)} loaded points.")

    optimizer=optimize_newton
    if minima_only: optimizer=optimize_lbfgs
    optimizer_kwargs['bounds'] = {'low': finder.low_bounds, 'up': finder.upp_bounds}

    minima, maxima, saddles = [], [], []

    mod0 = model_builder(finder.x0s[0])
    model_class = type(mod0)
    active_params_named =  {k: True if v.requires_grad else False for k, v in mod0.named_parameters()}
    active_params = list(active_params_named.values())
    active_params_names = [k for k, v in active_params_named.items() if v]
    
    # for the optimizer, sort bounds based on model internal order:
    low_bounds = torch.tensor([bounds[k][0] for k in active_params_names])
    upp_bounds = torch.tensor([bounds[k][1] for k in active_params_names])
    optimizer_kwargs['bounds'] = {'low': low_bounds, 'up': upp_bounds}
    
    sqgrad_class = construct_SquaredGradModel(model_class, loss_func)
    
    _, unflatten = torch.utils._pytree.tree_flatten(dict(mod0.named_parameters()))
    
    def eval_loss_fn_params(flat_params):
        """Defines the functional eval of the model given the parameters.
        Flatten is required for torch.func.hessian to return a 2-tensor and not a nested dict,
        as function like torch.func.hessian from torch.func expect a single input tensor or PyTree (nested dict) of tensors.
        and functional calls work with nested dicts (i.e. unflattened).
        """
        # print('flat_params', flat_params)
        params_dict = torch.utils._pytree.tree_unflatten(flat_params, unflatten) # see ChatGPT convo
        # print('params_dict', params_dict)
        y = torch.func.functional_call(mod0, params_dict, (input,), strict=True)
        return loss_func(y)

    for i, x0 in enumerate(finder.x0s):
        # if (i+1)!=325: continue
        print()
        print('__________________________________________________________________________________________')
        print(f"Attempt {i+1}/{finder.m}:")
        print(f"Starting point: {x0.cpu().numpy()}")

        # initialize NN module 
        mod = model_builder(x0)

        # # algorithm 0: minima only, gradient descent
        # final_point, final_val, path = run_local_search(optimize_lbfgs, mod, input, loss_func, **optimizer_kwargs)
        # final_params, _ = torch.utils._pytree.tree_flatten(dict(mod.named_parameters()))
        # grad = torch.cat([p.grad.flatten() for p in mod.parameters() if p.requires_grad])

        # # algorithm 1: gradient descent to minimize ||∇f||² 
        # sqgrad_model = sqgrad_class(*mod._init_args)
        # final_point, final_val_sqgrad, path = run_local_search(optimize_lbfgs, sqgrad_model, input, identity_loss, **optimizer_kwargs)
        # final_params, _ = torch.utils._pytree.tree_flatten(dict(sqgrad_model.named_parameters()))
        # grad = torch.sqrt(final_val_sqgrad)

        # algorithm 2: uncorrected newton's method:
        final_point, final_val, path = run_local_search(optimizer, mod, input, loss_func, **optimizer_kwargs)
        detached_params = {k: v.detach() for k, v in mod.named_parameters()} # detach to avoid memory blow up
        final_params, _ = torch.utils._pytree.tree_flatten(detached_params)
        grad = torch.cat([p.grad.flatten() for p in mod.parameters() if p.requires_grad])

        # filter active parameters
        final_point = final_point[active_params]
        final_val = eval_loss_fn_params(final_params)
        print(f"Arrival point: {final_point.cpu().numpy()}")
        print('check grad:', grad)
        final_point = torch.tensor([detached_params[name] for name in active_params_names])
        critical=False
        # check grad is small and final_point is new
        gradtol = 1e-5 or optimizer_kwargs['gradtol']
        if torch.norm(grad)<gradtol: critical = finder.add_minimum(final_point, final_val.item()) 
        
        # Classify critical point
        if critical:
            hessian = comp_hessian(eval_loss_fn_params, final_params)
            # hessian = torch.func.hessian(eval_loss_fn_params)(final_params)
            hessian = flatten_hessian_blocks(hessian)
            hessian = hessian[active_params][:, active_params]
            index = _torch_critical_point_index(hessian)
            
            if index == 0:
                minima.append((final_point, final_val, index))
                print("Type: minimum")
            elif index == dimension:
                maxima.append((final_point, final_val, index))
                print("Type: maximum")
            else:
                saddles.append((final_point, final_val, index))
                print("Type: saddle")

    # for point, value in finder.minima:
    #     point_t = torch.tensor(point, requires_grad=True)
    #     # y = model_builder(point_t)(input).to(device)

    #     print('model_eval', model_loss_eval(point_t), value)
    #     print('model_eval', model_loss_eval(point_t+0.1), value)

    #     hessian = torch.autograd.functional.hessian(model_loss_eval, point_t, strict=True)
    #     index = _torch_critical_point_index(hessian)

    #     if index == 0:
    #         minima.append((point, value, index))
    #         print("minimum", point)
    #     elif index == dimension:
    #         maxima.append((point, value, index))
    #         print("maximum", point)
    #     else:
    #         saddles.append((point, value, index))
    #         print("saddle", point)

    return minima, maxima, saddles

# ---- CSV Writer ----
def save_critical_points_to_csv(minima: torch.Tensor, maxima: torch.Tensor, saddles: torch.Tensor, dimension=2, filename="critical_points_torch.csv"):
    data = []
    for point_type, points in zip(["minimum", "maximum", "saddle"], [minima, maxima, saddles]):
        for p, fval, index in points:
            vardict = {f"x{i+1}": p[i].item() for i in range(dimension)}
            data.append({**vardict, "f_value": fval.item(), "type": point_type, "index": index})

    df = pd.DataFrame(data)
    df.to_csv(filename, float_format="%.10f")
    print(f"Saved {filename}")
    return df
