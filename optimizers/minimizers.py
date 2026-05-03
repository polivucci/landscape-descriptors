import jax
import jax.numpy as jnp
import optax
from typeconfig import default_dtype
import functools

def lbfgs_step(flat_loss, grad_fn=None):

    if grad_fn is None: 
        val_and_grad_fn = optax.value_and_grad_from_state(flat_loss)
    else:
        def val_and_grad_fn(p, state=None): return (flat_loss(p), grad_fn(p)) 

    # @jax.jit
    @functools.partial(jax.jit, static_argnames=["optimizer"])
    def train_step(flat_params, opt_state, optimizer):
        loss, grads = val_and_grad_fn(flat_params, state=opt_state)
        updates, opt_state = optimizer.update(
            grads, opt_state, flat_params, value=loss, grad=grads, value_fn=flat_loss
        )
        return updates, loss, opt_state
    
    return train_step

def optimize_lbfgs(flat_params, 
                   optim_step, 
                   bounds=None,
                   lr=1e-3, max_iter=100, atol=1e-6, rtol=1e-5, 
                   log_paths=False):
    """
    Optimize using Optax's LBFGS to find local loss minimum.

    Args:
        flat_params:  flat 1d array of paramters
        flat_loss: callable (flat_params) -> scalar loss
        bounds:  dict with keys 'low' and 'up' as jnp.arrays, or None
    """
    trajectory = []

    optimizer = optax.lbfgs(learning_rate=lr)
    opt_state = optimizer.init(flat_params)

    prev_params = flat_params
    loss = None

    conv = False
    oob  = False

    while not conv and not oob:
        for it in range(max_iter):
            updates, loss, opt_state = optim_step(flat_params, opt_state, optimizer)

            flat_params = optax.apply_updates(flat_params, updates)

            if log_paths:    # flatten params to 1D vector 

                trajectory.append((flat_params, float(loss)))

            # stopping criterion
            if jnp.allclose(flat_params, prev_params, atol=atol, rtol=rtol):
                conv = True
                break

            # bounds check
            if bounds is not None:
                low = jnp.array(bounds['low'], dtype=default_dtype)
                up  = jnp.array(bounds['up'],  dtype=default_dtype)
                if jnp.any(flat_params < low) or jnp.any(flat_params > up):
                    oob = True
                    print(f"Out of bounds at {it} iterations.")
                    break

            prev_params = flat_params

    return flat_params, loss, trajectory, conv


def optimize_gd(flat_params, 
                optim_step, 
                bounds=None,
                lr=1e-3, max_iter=100, atol=1e-6, rtol=1e-5, 
                log_paths=False):
    """
    Optimize using Optax's SGD to find local loss minimum.

    Args:
        flat_params:  flat 1d array of paramters
        flat_loss: callable (flat_params) -> scalar loss
        bounds:  dict with keys 'low' and 'up' as jnp.arrays, or None
    """
    trajectory = []

    optimizer = optax.sgd(learning_rate=lr)
    opt_state = optimizer.init(flat_params)

    prev_params = flat_params
    loss = None

    conv = False
    oob  = False

    while not conv and not oob:
        for it in range(max_iter):
            updates, loss, opt_state = optim_step(flat_params, opt_state, optimizer)

            flat_params = optax.apply_updates(flat_params, updates)

            if log_paths:    # flatten params to 1D vector 

                trajectory.append((flat_params, float(loss)))

            # stopping criterion
            if jnp.allclose(flat_params, prev_params, atol=atol, rtol=rtol):
                conv = True
                break

            # bounds check
            if bounds is not None:
                low = jnp.array(bounds['low'], dtype=default_dtype)
                up  = jnp.array(bounds['up'],  dtype=default_dtype)
                if jnp.any(flat_params < low) or jnp.any(flat_params > up):
                    oob = True
                    print(f"Out of bounds at {it} iterations.")
                    break

            prev_params = flat_params

    return flat_params, loss, trajectory, conv

