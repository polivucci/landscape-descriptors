import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

# Import your scalar Schwefel loss from the existing module
# It should accept a 1D tensor of shape (2,) and return a scalar tensor.
from schwefel_function import _schwefel  # NOTE: keeping your given name

class Schwefel2D(nn.Module):
    """
    Dummy model whose 'forward' just returns its learnable 2D parameters.
    Parameters are initialized via x1_init, x2_init.
    """
    def __init__(self, x1_init, x2_init):
        super().__init__()
        self.x1 = nn.Parameter(torch.tensor(float(x1_init)))
        self.x2 = nn.Parameter(torch.tensor(float(x2_init)))


    def forward(self):
        # For ML consistency: return parameters as a vector
        return torch.stack([self.x1, self.x2])

def schwefel_loss_2d(coords):
    """
    Wraps existing schwefel() into a torch-friendly loss function.
    """
    return _schwefel(coords[0], coords[1])

class Schwefel3D(nn.Module):
    """
    Dummy model whose 'forward' just returns its learnable 3D parameters.
    Parameters are initialized via x1_init, x2_init, x3_init.
    """
    def __init__(self, x1_init, x2_init, x3_init):
        super().__init__()
        self.x1 = nn.Parameter(torch.tensor(float(x1_init)))
        self.x2 = nn.Parameter(torch.tensor(float(x2_init)))
        self.x3 = nn.Parameter(torch.tensor(float(x3_init)))

    def forward(self):
        # For ML consistency: return parameters as a vector
        return torch.stack([self.x1, self.x2, self.x3])

def schwefel_loss_3d(coords):
    """
    Wraps schwefel() into a torch-friendly loss function for 3D.
    """
    return _schwefel(coords[0], coords[1], coords[2])

class Schwefel4D(nn.Module):
    """
    Dummy model whose 'forward' just returns its learnable 2D parameters.
    Parameters are initialized via x1_init, x2_init.
    """
    def __init__(self, x1_init, x2_init, x3_init, x4_init):
        super().__init__()
        self.x1 = nn.Parameter(torch.tensor(float(x1_init)))
        self.x2 = nn.Parameter(torch.tensor(float(x2_init)))
        self.x3 = nn.Parameter(torch.tensor(float(x3_init)))
        self.x4 = nn.Parameter(torch.tensor(float(x4_init)))


    def forward(self):
        # For ML consistency: return parameters as a vector
        return torch.stack([self.x1, self.x2, self.x3, self.x4])

def schwefel_loss_3d(coords):
    """
    Wraps schwefel() into a torch-friendly loss function for 3D.
    """
    return _schwefel(coords[0], coords[1], coords[2], coords[3])

