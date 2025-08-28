import torch
torch.set_default_dtype(torch.float64)
import matplotlib.pyplot as plt
def x_unnorm_schwe(x):
    '''transform from 0,1 to actual range
    '''
    return (x * 450.0) + 50.0

def y_norm_schwe(x):
    '''normalise outputs
    '''
    return x / 1000.0

class Schwefel1D(object):
    def __init__(self, x1):
        self.x1 = x1

class Schwefel2D(object):
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

class Schwefel3D(object):
    def __init__(self, x1, x2, x3):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

class Schwefel4D(object):
    def __init__(self, x1, x2, x3, x4):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4

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

def schwefel1D(*x):
    '''Schwefel 2 dimensions
    '''
    vars = [x_unnorm_schwe(xi) for xi in x]
    schwe = Schwefel1D(*vars)
    return y_norm_schwe(eval_schwefel(schwe))

def schwefel2D(*x):
    '''Schwefel 2 dimensions
    '''
    vars = [x_unnorm_schwe(xi) for xi in x]
    schwe = Schwefel2D(*vars)
    return y_norm_schwe(eval_schwefel(schwe))

def schwefel3D(*x):
    '''Schwefel 3 dimensions
    '''
    vars = [x_unnorm_schwe(xi) for xi in x]
    schwe = Schwefel3D(*vars)
    return y_norm_schwe(eval_schwefel(schwe))

def schwefel4D(*x):
    '''Schwefel 4 dimensions
    '''
    vars = [x_unnorm_schwe(xi) for xi in x]
    schwe = Schwefel4D(*vars)
    return y_norm_schwe(eval_schwefel(schwe))

# Hessian of generic function f
def f_hessian(func, *vars):
    if not all([var.requires_grad for var in vars]): 
        raise Exception('All inputs require gradients')
    h = torch.autograd.functional.hessian(func, vars, strict=True)
    return torch.tensor(h)

def schwefel2D_numpy(*x):
    x = [torch.tensor(xi) for xi in x]
    return schwefel2D(*x).detach().numpy()

def schwefel3D_numpy(*x):
    x = [torch.tensor(xi) for xi in x]
    return schwefel3D(*x).detach().numpy()

def schwefel4D_numpy(*x):
    x = [torch.tensor(xi) for xi in x]
    return schwefel4D(*x).detach().numpy()

if __name__=='__main__':

    

    # 2d plotting
    # fig2 = plt.figure(figsize=(5, 4), dpi=160)
    # plot_results_with_paths(schwefel_numpy, 
    #                         critical_points_csv=None, 
    #                         saddle_to_minima_csv=None, 
    #                         bounds=(0, 1), 
    #                         res=200, 
    #                         fig=fig2,
    # )
    # plt.show()

    # 3d plotting
    x = torch.linspace(0, 1, 20)
    xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
    # X = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=-1)
    f = torch.empty_like(xx.flatten())
    for i, (x1, x2, x3) in enumerate(zip(xx.flatten(), yy.flatten(), zz.flatten())):
        f[i] = schwefel3D(x1, x2, x3)
    f=f.reshape_as(xx)
    hs = [0, 5, 10, 15, 19]
    for h in hs:
        plt.contourf(yy[h,...], zz[h,...], f[h,...], levels=torch.linspace(0,2.2,20))
        plt.colorbar()
        plt.show()

    # 1d plotting
    # x = torch.linspace(0, 1, 200).unsqueeze(-1)
    # plt.plot(x, schwefel(x))
    # plt.show()
