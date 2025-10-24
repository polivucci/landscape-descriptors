import torch
from torch.optim import LBFGS
from torch.nn.utils import parameters_to_vector


def optimize_lbfgs(model, input, loss_fn, lr=1e-3, max_iter=100, atol=1e-6, rtol=1e-5, log_paths=False, bounds=None):
    """
    Optimize using PyTorch's LBFGS to find local minimum.
    """

    optimizer = LBFGS(model.parameters(), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

    trajectory = []  # Start with initial point 

    def closure(): #Closure required because LBFGS evaluates the function multiple times during each iteration.
            optimizer.zero_grad()
            y = model(input)
            loss = loss_fn(y)
            loss.backward()
            if log_paths: # optional, appends each trajectory point 
                trajectory.append((parameters_to_vector(model.parameters()).clone().detach(), loss.item())) 
            return loss

    active = [True if p.requires_grad else False for p in model.parameters()]
    prev_coords = parameters_to_vector(model.parameters()).detach() # initial coords
    for it in range(max_iter):
        loss = optimizer.step(closure)

        coords = parameters_to_vector(model.parameters()).clone().detach()

        # stopping criterion
        # grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad])
        # small_grad = torch.norm(grad) < rtol
        # if small_grad or close:
        try:
            # assert small_grad
            torch.testing.assert_close(
                coords, prev_coords, atol=atol, rtol=rtol
            )
            # if assert passes, break
            break
        except AssertionError:
            pass

        # bound check
        if bounds is not None:
            if torch.any(coords[active] < bounds['low']) or torch.any(coords[active] > bounds['up']): 
                print('out of bounds', coords[active])
                conv = True
                print(f"Out of bounds at {it} iterations.")
                break

        # Required when optimising Sqgrad: Detach to avoid graph blow-up (memory leak) 
        for p in model.parameters():
            p.grad = None

        prev_coords = coords

    return model, loss, trajectory  # Return both final point and path


def optimize_newton(model, input, loss_fn, lr=1e-3, tol=1e-5, max_iter=50, log_paths=False, bounds=None):
    """
    Converge to a critical point (grad(loss_fn) = 0) of the loss function with respect to model parameters
    using Newton's method with no additional modification to make the Hessian positive definite.
    """
    # Flatten model parameters into a single vector for Hessian computation
    trajectory = []  # for optional log trajectory
    
    conv = False; oob=False
    while conv==False and oob==False: # simple adaptive learning rate
        params = [p for p in model.parameters() if p.requires_grad]
        prev_coords = parameters_to_vector(params).clone().detach()
        print('learning rate', lr)
        # print('x0', prev_coords)
        for it in range(max_iter):
            # Prev gradients to zero
            model.zero_grad()
            
            # Loss and gradients 
            y = model(input)
            loss = loss_fn(y)
            # loss.backward(create_graph=True)
            grad1 = torch.autograd.grad(loss, params, create_graph=True)
            # Gradients into vector
            grad = torch.cat([g.flatten() for g in grad1])
            # grad = torch.cat([p.grad.flatten() for p in params])
            
            # Build Hessian 
            n_params = grad.numel()
            H = torch.zeros((n_params, n_params), dtype=grad.dtype, device=grad.device)
            for i in range(n_params):
                # derivative of grad[i] wrt all parameters:
                g_i = grad[i]
                grad2 = torch.autograd.grad(g_i, params, retain_graph=True)
                H[i] = torch.cat([g.flatten() for g in grad2])
            
            # Compute Newton step H * delta = grad
            delta = torch.linalg.solve(H, grad)
            
            # Update parameters p_new = p - delta
            idx = 0
            for p in params:
                numel = p.numel()
                p.data -= delta[idx:idx + numel].reshape_as(p) * lr
                idx += numel
            
            coords = parameters_to_vector(params).clone().detach()

            if log_paths: # optional, appends each trajectory point 
                    trajectory.append([prev_coords, loss.item()]) 

            # Convergence checks
            # diff = coords - prev_coords
            # close = torch.allclose(
            #         coords, prev_coords, atol=1e-6, rtol=1e-5
            #     )
            small_grad = torch.norm(grad) < tol
            # if small_grad or close:
            if small_grad:
                conv = True
                print(f"Converged after {it} iterations.")
                break

            if bounds is not None:
                if torch.any(coords < bounds['low']) or torch.any(coords > bounds['up']): 
                    # print('out of bounds', coords)
                    oob = True
                    print(f"Out of bounds at {it} iterations.")
                    break

            coords = prev_coords
        
        # print(trajectory[0])
        # print(trajectory[-1])

        # if failure to converge, decimate learning rate
        lr *= 0.1
        # max_iter *= 10

    # load gradients into the model
    for p, g in zip(params, grad1):
        p.grad = g

    return model, loss, trajectory
