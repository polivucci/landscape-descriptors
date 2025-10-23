import torch
from torch.quasirandom import SobolEngine
from scipy.spatial import KDTree
import pandas as pd

torch.set_default_dtype(torch.float64)

class TorchMinimaFinder:
    def __init__(self, bounds=(0.0, 1.0), dimension=2, min_distance=0.01, m=64, device="cpu", seed=42):
        if bounds==(0.0, 1.0): bounds = dimension*((0.0, 1.0),)
        assert len(bounds)==dimension
        self.bounds = torch.tensor(bounds, device=device)
        self.low_bounds, self.ranges = self.bounds[:,0], self.bounds[:,1]-self.bounds[:,0]
        
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

    def generate_starting_points(self):
        sampler = SobolEngine(dimension=self.dimension, scramble=True, seed=self.seed)
        self.x0s = sampler.draw(self.m).to(self.device)
        self.x0s *= self.ranges
        self.x0s += self.low_bounds
        # self.kdtree_x0s = KDTree([x0 for x0 in self.x0s])

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

    def add_minimum(self, point, value):
        point_np = point.detach().cpu().numpy()
        print(point_np, "Reject:", self._is_too_close(point_np))
        # print(point_np, "Reject x0:", self._is_too_close_x0(point_np))
        if not self._is_too_close(point_np):
            self.minima.append((point_np, value))
            self.update_kdtree()
            return True
        return False

# Torch gradient descent
from critical_points.optimizers import optimize_lbfgs, optimize_newton
def run_local_search(optimizer, model, input, loss_fn, **optimizer_kwargs):
    _, final_val, path = optimizer(model, input, loss_fn, **optimizer_kwargs)
    if optimizer_kwargs['log_paths']: print(pd.DataFrame(path).tail(10))
    fin_point = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    return fin_point, final_val

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
    Dummy loss needed for compatibility.
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
def find_critical_points_torch(model_builder, loss_func, input, bounds, dimension=2,
                               num_attempts=64, min_distance=0.01, seed=42,
                               device="cpu", **optimizer_kwargs):
    """
    Finds critical points using torch autograd + NNModule
    """
    finder = TorchMinimaFinder(bounds, dimension, min_distance, num_attempts, seed=seed, device=device)

    minima, maxima, saddles = [], [], []

    mod0 = model_builder(finder.x0s[0])
    model_class = type(mod0)
    active_params = [True if p.requires_grad else False for p in mod0.parameters()]
    sqgrad_class = construct_SquaredGradModel(model_class, loss_func)
    
    _, unflatten = torch.utils._pytree.tree_flatten(dict(mod0.named_parameters()))
    
    def eval_loss_fn_params(flat_params):
        """Defines the functional eval of the model given the parameters.
        Flatten is required for torch.func.hessian to return a 2-tensor and not a nested dict,
        as function like torch.func.hessian from torch.func expect a single input tensor or PyTree (nested dict) of tensors.
        and functional calls work with nested dicts (i.e. unflattened).
        """
        params_dict = torch.utils._pytree.tree_unflatten(flat_params, unflatten) # see ChatGPT convo
        y = torch.func.functional_call(mod0, params_dict, (input,))
        return loss_func(y)

    for i, x0 in enumerate(finder.x0s):

        print()
        print('__________________________________________________________________________________________')
        print(f"Attempt {i+1}/{finder.m}: starting point {x0.cpu().numpy()}")
        print(f"Starting point: {x0.cpu().numpy()}")

        # initialize NN module 
        mod = model_builder(x0)

        # # algorithm 1: gradient descent to minimize ||∇f||² 
        # sqgrad_model = sqgrad_class(*mod._init_args)
        # final_point, final_val_sqgrad = run_local_search(optimize_lbfgs, sqgrad_model, input, identity_loss, lr=lr, steps=steps)
        # final_params, _ = torch.utils._pytree.tree_flatten(dict(sqgrad_model.named_parameters()))
        # grad = torch.sqrt(final_val_sqgrad)

        # algorithm 2: uncorrected newton's method:
        final_point, final_val = run_local_search(optimize_newton, mod, input, loss_func, **optimizer_kwargs)
        final_params, _ = torch.utils._pytree.tree_flatten(dict(mod.named_parameters()))
        grad = torch.cat([p.grad.flatten() for p in mod.parameters() if p.requires_grad])

        # filter active parameters
        final_point = final_point[active_params]
        final_val = eval_loss_fn_params(final_params)
        print('check grad', grad)
        critical=False
        # check grad is small and final_point is new
        if torch.norm(grad)<1e-5: critical = finder.add_minimum(final_point, final_val.item()) 
        
        # Classify critical point
        if critical:
            hessian = torch.func.hessian(eval_loss_fn_params)(final_params)
            hessian = flatten_hessian_blocks(hessian)
            hessian = hessian[active_params][:, active_params]
            index = _torch_critical_point_index(hessian)
            
            if index == 0:
                minima.append((final_point, final_val, index))
                print("minimum", final_point)
            elif index == dimension:
                maxima.append((final_point, final_val, index))
                print("maximum", final_point)
            else:
                saddles.append((final_point, final_val, index))
                print("saddle", final_point)

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
