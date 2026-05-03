import jax
import jax.numpy as jnp
from typeconfig import default_dtype

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

    return flat_params, loss, trajectory, conv