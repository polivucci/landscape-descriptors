import jax
import jax.numpy as jnp
import optax
from typeconfig import default_dtype
import functools

from jax.debug import print as jaxprint

jnp.set_printoptions(precision=15)

def line_search_monotonic(flat_params, updates, flat_loss, res=0):
    """
    Evaluate `flat_loss` at `res` equally spaced points strictly *between*
    flat_params and flat_params + updates, plus the two endpoints
    themselves (always included), and return the last point along that
    segment for which the loss sequence is still monotonic strictly decreasing.

    As soon as a point's loss is strictly greater than the previous
    point's loss, the search stops and the *previous* point is returned.
    If no such violation occurs, the final point (flat_params + updates)
    is returned.

    Args:
        flat_params: 1D jnp array, the starting point.
        updates: 1D jnp array, same shape as flat_params, the step to the
            end point.
        flat_loss: a function mapping a 1D array (a point on the segment)
            to a scalar loss. Must be jit/vmap-compatible.
        res: int (static), number of equally spaced points *between* the
            two endpoints. Total evaluated points = res + 2. Must be >= 0.
            If res == 0, no loss evaluation is performed at all and
            flat_params + updates is returned directly.

    Returns:
        1D jnp array with the same shape as flat_params: the selected
        point on the segment.
    """

    # jaxprint('monotonicity res = {w}', w=res)
    # jaxprint('flat_params = {w}', w=flat_params)
    # jaxprint('updates = {w}', w=updates)

    if res == 0:
        return 1, flat_params + updates

    n_points = res + 2  # both endpoints + res interior points

    # t = 0 -> flat_params, t = 1 -> flat_params + updates
    ts = jnp.linspace(0.0, 1.0, n_points)**2 # quadratic
    points = ts[:, None] * updates[None, :]
    # jaxprint('points shape {w}', w=points.shape)

    # Evaluate the loss at every point in one vectorized call.
    losses = jax.vmap(flat_loss)(flat_params[None, :]+points)

    # diffs[i] is True iff losses[i+1] > losses[i], i.e. a violation of
    # "strictly decreasing" occurred going from point i to point i+1.
    diffs = losses[1:].round(10) > losses[:-1].round(10)
    # equals = losses[1:].round(10) == losses[:-1].round(10)

    # if jnp.all(equals): raise RuntimeError('All equals')

    has_violation = jnp.any(diffs)
    # first index where a violation occurs
    first_violation = jnp.argmax(diffs)

    # # flags if violation occurs at second point 
    # violated_at_1 = first_violation==0 and has_violation

    # jaxprint(' ')
    # jaxprint('losses {w}', w=losses.round(10))
    # jaxprint('monotonicity violated at 1: {w}', w=violated_at_1)
    # jaxprint('monotonicity violated {w} at {p}', w=has_violation, p=first_violation)

    # If a violation was found at diffs[first_violation] 
    # (i.e. between points[first_violation] and points[first_violation + 1]), 
    # returns the previous one i.e. points[first_violation].
    # If no violation was found, returns the last point, points[n_points - 1].
    # If the second point already violates monotonicity (first_violation==0), the search would 
    # get stuck at the first point. 
    # Therefore the search is repeated between the first and the second point until violation
    # occurs at an interior point.
    stop_idx = jnp.where(has_violation, 
                         first_violation,
                        #  jnp.where(violated_at_1, 
                                #    line_search_monotonic(flat_params, points[1], flat_loss, res=res)[0], 
                                #    first_violation), 
                         n_points - 1
                         )

    # jaxprint('idx {w}', w=stop_idx)
    return stop_idx, points[stop_idx]
    
# try log loss (for each saddle, hence define in connectivity)
# add initialization verification (grad free before first step): this is in connectivity after offset near saddle 

def lbfgs_step(flat_loss, grad_fn=None, symmetry_fn=None, monotonicity_res=0):

    if grad_fn is None: 
        val_and_grad_fn = jax.value_and_grad(flat_loss)
    else:
        def val_and_grad_fn(p): return (flat_loss(p), grad_fn(p)) 

    @functools.partial(jax.jit, static_argnames=["optimizer"])
    def train_step(flat_params, opt_state, optimizer):
        loss, grads = val_and_grad_fn(flat_params)
        updates, opt_state = optimizer.update(
            grads, opt_state, flat_params, value=loss, grad=grads, value_fn=flat_loss
        )

        # monotonic line search
        if monotonicity_res>0:
            _, updates = line_search_monotonic(flat_params, updates, flat_loss, res=monotonicity_res)

        flat_params = optax.apply_updates(flat_params, updates)

        # apply periodicity
        # jaxprint('')
        # jaxprint('x={w}', w=flat_params)
        if symmetry_fn is not None:
          flat_params = symmetry_fn(flat_params)
        # jaxprint('x={w}', w=flat_params)
        return flat_params, loss, grads, opt_state
    
    return train_step

def optimize(optimizer,
             flat_params, 
             optim_step, 
             bounds=None,
             max_iter=100, atol=1e-6, rtol=1e-5, 
             gradtol=1e-5,
             log_paths=False,
             **kwargs):
    """
    Optimize using Optax's LBFGS to find local loss minimum.

    Args:
        flat_params:  flat 1d array of paramters
        flat_loss: callable (flat_params) -> scalar loss
        bounds:  dict with keys 'low' and 'up' as jnp.arrays, or None
    """

    opt_state = optimizer.init(flat_params)

    if bounds is not None:
        low = jnp.array(bounds['low'], dtype=default_dtype)
        up  = jnp.array(bounds['up'],  dtype=default_dtype)

    prev_params = flat_params
    loss = None

    conv = False
    oob  = False
    nans = False
    check_its = max(10, max_iter // 100)

    trajectory = []

    for it in range(max_iter):
            
            if conv or nans: break

            oob = False

            flat_params, loss, grad, opt_state = optim_step(prev_params, opt_state, optimizer)

            if log_paths:   
                trajectory.append((prev_params, float(loss)))

            # # stopping criterion
            # if jnp.allclose(flat_params, prev_params, atol=atol, rtol=rtol):
            #     conv = True
            #     print(f"\t Converged after {it} iterations.")
            #     break

            # print('prev_params', prev_params)
            # print('grad', grad)
            # print('flat_params', flat_params)

            gradnorm = jnp.linalg.norm(grad)

            if it % check_its == 0: print(f'gradnorm at {it:d} =', gradnorm)
            # print('dist', jnp.linalg.norm(flat_params-prev_params))
            
            # check nans
            if it % check_its==0: nans = jnp.any(jnp.isnan(gradnorm))
            if nans: 
                print('NaNs')
                break

            # stopping criterion
            if gradnorm < gradtol:
                conv = True
                print(f"\t Converged after {it} iterations. ||grad||={gradnorm:.2e} < {gradtol}")

            # bounds check
            if bounds is not None:
                if jnp.any(flat_params < low) or jnp.any(flat_params > up):
                    oob = True
                    print(f"\t Out of bounds at {it} iterations. Point ", flat_params.round(4))

            prev_params = flat_params
    
    if not conv and not oob:
        print(f"Failed to converge. Terminal ||grad||={gradnorm:.2e}, point", flat_params.round(4))

    if conv and oob:
        conv=False
        print(f"Converged out of bounds, point", flat_params.round(4))

    del opt_state, optimizer, prev_params, grad
    
    return flat_params, loss, trajectory, conv

def optimize_lbfgs(flat_params, 
                   optim_step, 
                   bounds=None,
                   lr=1e-3, max_iter=100, atol=1e-6, rtol=1e-5, 
                   log_paths=False,
                   **kwargs):
    """
    Optimize using Optax's LBFGS to find local loss minimum.

    Args:
        flat_params:  flat 1d array of paramters
        flat_loss: callable (flat_params) -> scalar loss
        bounds:  dict with keys 'low' and 'up' as jnp.arrays, or None
    """
    
    return optimize(optax.lbfgs(learning_rate=lr,
                                # linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=10, 
                                #                                           max_learning_rate=lr,
                                #                                           initial_guess_strategy='one')
                                ),
                    flat_params, 
                    optim_step, 
                    bounds=bounds,
                    max_iter=max_iter, atol=atol, rtol=rtol, 
                    log_paths=log_paths,
                    **kwargs
                    )

def optimize_gd(flat_params, 
                optim_step, 
                bounds=None,
                lr=1e-3, max_iter=100, atol=1e-6, rtol=1e-5, 
                log_paths=False,
                **kwargs):
    """
    Optimize using Optax's SGD to find local loss minimum.

    Args:
        flat_params:  flat 1d array of paramters
        flat_loss: callable (flat_params) -> scalar loss
        bounds:  dict with keys 'low' and 'up' as jnp.arrays, or None
    """
    
    return optimize(optax.sgd(learning_rate=lr),
                    flat_params, 
                    optim_step, 
                    bounds=bounds,
                    max_iter=max_iter, atol=atol, rtol=rtol, 
                    log_paths=log_paths,
                    **kwargs)

def optimize_hybrid(flat_params, 
                    optim_step, 
                    bounds=None,
                    lr=1e-3, max_iter=100, atol=1e-6, rtol=1e-5, 
                    log_paths=False,
                    **kwargs):
    """
    Optimize using Optax's SGD to find local loss minimum.

    Args:
        flat_params:  flat 1d array of paramters
        flat_loss: callable (flat_params) -> scalar loss
        bounds:  dict with keys 'low' and 'up' as jnp.arrays, or None
    """
    
    return optimize(optax.sgd(learning_rate=lr),
                    flat_params, 
                    optim_step, 
                    bounds=bounds,
                    max_iter=max_iter, atol=atol, rtol=rtol, 
                    log_paths=log_paths,
                    **kwargs)