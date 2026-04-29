import jax
import jax.numpy as jnp
import optax
from typeconfig import default_dtype

def lbfgs_step(flat_loss, grad_fn=None):

    if grad_fn is None: 
        val_and_grad_fn = optax.value_and_grad_from_state(flat_loss)

    @jax.jit
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

    for it in range(max_iter):
        updates, loss, opt_state = optim_step(flat_params, opt_state, optimizer)

        flat_params = optax.apply_updates(flat_params, updates)

        if log_paths:
            trajectory.append((flat_params, float(loss)))

        # stopping criterion
        if jnp.allclose(flat_params, prev_params, atol=atol, rtol=rtol):
            break

        # bounds check
        if bounds is not None:
            low = jnp.array(bounds['low'], dtype=default_dtype)
            up  = jnp.array(bounds['up'],  dtype=default_dtype)
            if jnp.any(flat_params < low) or jnp.any(flat_params > up):
                print(f"Out of bounds at {it} iterations.")
                break

        prev_params = flat_params

    return flat_params, loss, trajectory


def newton_step(flat_loss, grad_fn=None, hessian_fn=None):

    if grad_fn is None: grad_fn = jax.grad(flat_loss)
    if hessian_fn is None: 
        print('compute hess')
        hessian_fn = jax.hessian(flat_loss)

    @jax.jit
    def train_step(flat_params, lr):
        grad = grad_fn(flat_params)
        H    = hessian_fn(flat_params)

        # Compute Newton step: H * delta = grad
        delta = jnp.linalg.solve(H, grad)

        # Update parameters: p_new = p - lr * delta
        updates = flat_params - lr * delta

        loss = flat_loss(updates)
        return updates, loss, grad
    
    return train_step

def optimize_newton(flat_params, 
                    optim_step,
                    bounds=None, 
                    lr=1e-3, tol=1e-5, max_iter=50, 
                    log_paths=False):
    """
    Converge to a critical point (grad(loss_fn) = 0) of the loss function with respect to params
    using Newton's method with no additional modification to make the Hessian positive definite.

    Args:
        flat_params:  flat 1d array of paramters
        flat_loss: callable (flat_params) -> scalar loss
        bounds:  dict with keys 'low' and 'up' as jnp.arrays, or None
    """
    trajectory = []

    lr0  = lr
    conv = False
    oob  = False

    while not conv and not oob and lr >= lr0 * 1e-3:
        print(f"Learning rate: {lr}")
        prev_params = flat_params

        for it in range(max_iter):
            
            flat_params, loss, grad = optim_step(flat_params, lr)

            if log_paths:
                trajectory.append((prev_params, float(loss)))

            # Convergence check
            if jnp.linalg.norm(grad) < tol:
                conv = True
                print(f"Converged after {it} iterations.")
                break

            # Bounds check
            if bounds is not None:
                low = jnp.array(bounds['low'], dtype=default_dtype)
                up  = jnp.array(bounds['up'],  dtype=default_dtype)
                if jnp.any(flat_params < low) or jnp.any(flat_params > up):
                    oob = True
                    print(f"Out of bounds at {it} iterations.")
                    break

            prev_params = flat_params

        lr *= 0.1 # decimate learning rate if not converged

    if not conv and not oob:
        print("Failed to converge.")

    return flat_params, loss, trajectory