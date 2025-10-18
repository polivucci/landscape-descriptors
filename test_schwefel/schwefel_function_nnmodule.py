import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

def _schwefel(*x):
    '''Schwefel function in N dimensions.
    The Schwefel function has m^N minima on the interval [a,b]^N, where m is the number of minima in 1D on [a,b]. 
    For instance in the interval [50,500]^N it has 3^N minima.
    '''
    xx = [(xi * 450.0) + 50.0 for xi in x] # transform from 0,1 to actual range
    for i, xi in enumerate(x):
        if not isinstance(xi, torch.Tensor):
            xx[i] = torch.tensor(xx[i], requires_grad=True)
    flat_x = torch.cat([xi.reshape(-1) for xi in xx])
    # return normalised to 1000
    return ( 418.9829 * len(x) - torch.sum(flat_x * torch.sin(torch.sqrt(torch.abs(flat_x)))) ) * 1e-3

class Schwefel2D(nn.Module):
    """
    Dummy model whose 'forward' just returns its learnable parameters.
    Parameters are initialized via x1_init, x2_init.
    """
    def __init__(self, x1_init, x2_init):
        super().__init__()
        self.x1 = nn.Parameter(torch.tensor(float(x1_init)))
        self.x2 = nn.Parameter(torch.tensor(float(x2_init)))

    def forward(self, input):
        # For ML consistency: return parameters as a vector
        return torch.stack([self.x1, self.x2])

class Schwefel3D(nn.Module):
    """
    Dummy model whose 'forward' just returns its learnable parameters.
    Parameters are initialized via x1_init, x2_init, x3_init.
    """
    def __init__(self, x1_init, x2_init, x3_init):
        super().__init__()
        self.x1 = nn.Parameter(torch.tensor(float(x1_init)))
        self.x2 = nn.Parameter(torch.tensor(float(x2_init)))
        self.x3 = nn.Parameter(torch.tensor(float(x3_init)))

    def forward(self, input):
        # For ML consistency: return parameters as a vector
        return torch.stack([self.x1, self.x2, self.x3])

class Schwefel4D(nn.Module):
    """
    Dummy model whose 'forward' just returns its learnable parameters.
    Parameters are initialized via x1_init, x2_init.
    """
    def __init__(self, x1_init, x2_init, x3_init, x4_init):
        super().__init__()
        self.x1 = nn.Parameter(torch.tensor(float(x1_init)))
        self.x2 = nn.Parameter(torch.tensor(float(x2_init)))
        self.x3 = nn.Parameter(torch.tensor(float(x3_init)))
        self.x4 = nn.Parameter(torch.tensor(float(x4_init)))

    def forward(self, input):
        # For ML consistency: return parameters as a vector
        return torch.stack([self.x1, self.x2, self.x3, self.x4])

def schwefel_loss(*coords):
    """
    Wraps schwefel() into a torch-friendly loss function for 3D.
    """
    return _schwefel(*coords)

def schwefel2D_numpy(*x):
    # combined nnmodule+schwefel loss machinery
    x = [torch.tensor(xi) for xi in x]
    model = Schwefel2D(*x)
    coords = model(x)                        # get parameters [x1, x2]
    loss = _schwefel(*coords)                 # evaluate Schwefel loss
    return float(loss.item())               # return scalar for NumPy

def schwefel3D_numpy(*x):
    # combined nnmodule+schwefel loss machinery
    x = [torch.tensor(xi) for xi in x]
    model = Schwefel3D(*x)
    coords = model(x)                        # get parameters [x1, x2]
    loss = _schwefel(*coords)                 # evaluate Schwefel loss
    return float(loss.item())      

def schwefel4D_numpy(*x):
    # combined nnmodule+schwefel loss machinery
    x = [torch.tensor(xi) for xi in x]
    model = Schwefel4D(*x)
    coords = model(x)                        # get parameters [x1, x2]
    loss = _schwefel(*coords)                 # evaluate Schwefel loss
    return float(loss.item())      