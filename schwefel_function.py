import torch
torch.set_default_dtype(torch.float64)

from plot_results import plot_results_with_paths

def x_unnorm_schwe(x):
    '''transform from 0,1 to actual range
    '''
    return (x * 450.0) + 50.0

def y_norm_schwe(x):
    '''normalise outputs
    '''
    return x / 1000.0

class Schwefel2D(object):
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

def _schwefel(*x):
    '''Schwefel function in N dimensions
    '''
    xx = list(x)
    for i, xi in enumerate(x):
        if not isinstance(xi, torch.Tensor):
            xx[i] = torch.tensor(xx[i], requires_grad=True)
    flat_x = torch.cat([xi.reshape(-1) for xi in xx])
    return 418.9829 * len(x) - torch.sum(flat_x * torch.sin(torch.sqrt(torch.abs(flat_x))))

def eval_schwefel(schwefel_object):
    xs = [var for var in vars(schwefel_object).values()]
    return _schwefel(*xs)

def schwefel(*x):
    '''Schwefel 2 dimensions
    '''
    vars = [x_unnorm_schwe(xi) for xi in x]
    schwe = Schwefel2D(*vars)
    return y_norm_schwe(eval_schwefel(schwe))

# Hessian of generic function f
def f_hessian(func, *vars):
    if not all([var.requires_grad for var in vars]): 
        raise Exception('All inputs require gradients')
    h = torch.autograd.functional.hessian(func, vars, strict=True)
    return torch.tensor(h)


if __name__=='__main__':

    # Convert to numpy and Plot the function:

    def schwefel_numpy(x):
        x = [torch.tensor(xi) for xi in x]
        return schwefel(*x).detach().numpy()

    plot_results_with_paths(schwefel_numpy, minima=None, bounds=(0,1), saddle_points=None, res=50)