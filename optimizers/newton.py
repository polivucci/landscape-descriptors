import jax
import jax.numpy as jnp
from typeconfig import default_dtype
import functools

from jax.debug import print as jaxprint

def newton_update(grad, H):
    '''Solves Newton's update with preconditioning.

    See 
    '''
    # # no preconditioning
    # delta = jnp.linalg.solve(H, grad)

    # D = 1./jnp.linalg.norm(H, axis=0, keepdims=True) # precondition with column norm 
    # delta = jnp.linalg.solve(H*D, grad*D[0,:])
    
    # precondition symmetrically w diagonal
    D = 1.0/jnp.sqrt(jnp.abs(jnp.diag(H)))[None,:]
    delta = jnp.linalg.solve(H*D*D.T, grad*D[0,:])
    delta *= D[0,:]

    # "trust region": bounds update if it's larger than 2x the domain (hard coded)
    trust = jnp.abs(delta) < 2.0
    sgn = jnp.sign(delta)
    delta = jnp.where(trust, delta, sgn*2.0) 

    # jaxprint(' ')
    # jaxprint('x={w}', w=flat_params.round(4))
    # jaxprint('grad={w}', w=grad)
    # # jaxprint('D={w}', w=D)
    # # jaxprint('H={w}', w=H)
    # # jaxprint('HD={w}', w=H*D)
    # eigv = jnp.linalg.eigh(H)
    # jaxprint('eigvals(H)={w}', w=eigv)
    # jaxprint('cond(H)={w}', w=jnp.linalg.cond(H))
    # jaxprint('cond(HD)={w}', w=jnp.linalg.cond(H*D))
    # # jaxprint('H={w}', w=H.flatten())

    return delta

def newton_step(flat_loss, grad_fn=None, hessian_fn=None, symmetry_fn=None):

    if grad_fn is None: grad_fn = jax.grad(flat_loss)
    if hessian_fn is None: hessian_fn = jax.hessian(flat_loss)

    @functools.partial(jax.jit, static_argnames=["lr"])
    def train_step(flat_params, lr):
        grad = grad_fn(flat_params)
        H    = hessian_fn(flat_params)

        # Solves Newton step: H @ delta = grad
        delta = newton_update(grad, H)

        # Update parameters: p_new = p - lr * delta
        flat_params -= lr * delta

        # apply periodicity
        if symmetry_fn is not None:
          flat_params = symmetry_fn(flat_params)

        loss = flat_loss(flat_params)
        return flat_params, loss, grad
    
    return train_step

def optimize_newton(flat_params, 
                    optim_step,
                    bounds=None, 
                    lr=1e-3, gradtol=1e-5, max_iter=50, 
                    log_paths=False,
                    **kwargs):
    """
    Converge to a critical point (grad(loss_fn) = 0) of the loss function with respect to params
    using Newton's method with no additional modification to make the Hessian positive definite.

    Args:
        flat_params:  flat 1d array of paramters
        flat_loss: callable (flat_params) -> scalar loss
        bounds:  dict with keys 'low' and 'up' as jnp.arrays, or None
    """

    if bounds is not None:
        low = jnp.array(bounds['low'], dtype=default_dtype)
        up  = jnp.array(bounds['up'],  dtype=default_dtype)
        
    trajectory = []

    conv = False
    oob  = False
    check_its = max(10, max_iter // 100)

    # lr0  = lr
    # while not conv and not oob and lr >= lr0 * 1e-3:
    # while not conv and not oob:
    print(f"Learning rate: {lr}")
    prev_params = flat_params

    for it in range(max_iter):
        
        flat_params, loss, grad = optim_step(prev_params, lr)

        if log_paths:
            trajectory.append((flat_params, float(loss)))

        # Convergence check
        gradnorm = jnp.linalg.norm(grad)
        # if it % 10 == 0: print(f'x at {it:d} =', flat_params.round(4))
        # if it % 10 == 0: print('grad', grad)
        # if it % 10 == 0: print(f'gradnorm at {it:d} =', gradnorm)
        if it % check_its == 0: print(f'gradnorm at {it:d} =', gradnorm)
        if gradnorm < gradtol:
            conv = True
            print(f"\t Converged after {it} iterations. ||grad||={gradnorm:.2e} < {gradtol}")
            break

        # Bounds check
        if bounds is not None:
            if jnp.any(flat_params < low) or jnp.any(flat_params > up): 
                oob = True
                print(f"\t Out of bounds at {it} iterations. Point ", flat_params.round(4))
                # break

        prev_params = flat_params

        # lr *= 0.1 # decimate learning rate if not converged

    if not conv and not oob:
        print(f"Failed to converge. Terminal ||grad||={gradnorm:.2e}, point", flat_params.round(4))

    return flat_params, loss, trajectory, conv