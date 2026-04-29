import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import equinox as eqx

from typeconfig import default_dtype

def _schwefel(*x):
    '''Schwefel function in N dimensions.
    The Schwefel function has m^N minima on the interval [a,b]^N, where m is the number of minima in 1D on [a,b].
    Example: in the interval [50,500]^N it has 3^N minima.
    '''
    xx = jnp.array([(xi * 450.0) + 50.0 for xi in x], dtype=default_dtype)
    return (418.9829 * len(x) - jnp.sum(xx * jnp.sin(jnp.sqrt(jnp.abs(xx))))) * 1e-3

def schwefel_loss(params, input): # jax friendly loss
    x = params(input)
    return _schwefel(*x)

class Schwefel2D(eqx.Module):
    """
    Dummy model whose 'forward' just returns its learnable parameters.
    Parameters are initialized via x1_init, x2_init.
    """
    x1: jnp.ndarray
    x2: jnp.ndarray
    def __init__(self, x1_init, x2_init):
        self.x1 = jnp.array(x1_init, dtype=default_dtype)
        self.x2 = jnp.array(x2_init, dtype=default_dtype)

    def __call__(self, x):
        return ravel_pytree(self)[0]

class Schwefel3D(eqx.Module):
    """
    Dummy model whose 'forward' just returns its learnable parameters.
    """
    x1: jnp.ndarray
    x2: jnp.ndarray
    x3: jnp.ndarray
    def __init__(self, x1_init, x2_init, x3_init):
        self.x1 = jnp.array(x1_init, dtype=default_dtype)
        self.x2 = jnp.array(x2_init, dtype=default_dtype)
        self.x3 = jnp.array(x3_init, dtype=default_dtype)

    def __call__(self, x):
        return ravel_pytree(self)[0]
    
class Schwefel4D(eqx.Module):
    """
    Dummy model whose 'forward' just returns its learnable parameters.
    """
    x1: jnp.ndarray
    x2: jnp.ndarray
    x3: jnp.ndarray
    x4: jnp.ndarray
    def __init__(self, x1_init, x2_init, x3_init, x4_init):
        self.x1 = jnp.array(x1_init, dtype=default_dtype)
        self.x2 = jnp.array(x2_init, dtype=default_dtype)
        self.x3 = jnp.array(x3_init, dtype=default_dtype)
        self.x4 = jnp.array(x4_init, dtype=default_dtype)

    def __call__(self, x):
        return ravel_pytree(self)[0]