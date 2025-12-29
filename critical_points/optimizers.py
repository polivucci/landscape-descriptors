import torch
from torch.optim import LBFGS, SGD
from torch.nn.utils import parameters_to_vector

torch.set_printoptions(precision=7, sci_mode=True)

def optimize_lbfgs(model, input, loss_fn, lr=1e-3, max_iter=100, atol=1e-6, rtol=1e-5, gradtol=1e-5, log_paths=False, bounds=None, **kwargs):
    """
    Optimize using PyTorch's LBFGS to find local minimum.
    """

    optimizer = LBFGS(model.parameters(), lr=lr, max_iter=max_iter, tolerance_grad=gradtol, **kwargs)

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
    conv = False; oob=False
    for it in range(max_iter):
        loss = optimizer.step(closure)

        coords = parameters_to_vector(model.parameters()).clone().detach()

        # stopping criterion
        grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad])
        small_grad = torch.norm(grad) < gradtol
        if small_grad: 
            conv=True
            break

        # try:
        #     # assert small_grad
        #     torch.testing.assert_close(
        #         coords, prev_coords, atol=atol, rtol=rtol
        #     )
        #     # if assert passes, break
        #     break
        # except AssertionError:
        #     pass

        # bound check
        if bounds is not None:
            if torch.any(coords[active] < bounds['low']) or torch.any(coords[active] > bounds['up']): 
                # print('out of bounds', coords[active])
                oob = True
                conv = True
                print(f"Out of bounds at {it} iterations.")
                break

        # # Required when optimising Sqgrad: Detach to avoid graph blow-up (memory leak) 
        # for p in model.parameters():
        #     p.grad = None

        prev_coords = coords
    
    if not conv and not oob: print(f"Failed to converge. Max iterations exceeded.")
    if conv and oob: conv='oob'
    
    print('last grad norm', torch.norm(grad))

    return model, loss, trajectory, conv  # Return both final point and path


def optimize_newton(model, 
                    input, 
                    loss_fn, 
                    lr=1e-3, gradtol=1e-5, max_iter=50, 
                    bounds=None,
                    log_paths=False, 
                    log_grad=False
                    ):
    """
    Converge to a critical point (grad(loss_fn) = 0) of the loss function using Newton's method with 
    no modification to make the Hessian positive definite. 
    """
    
    tol = gradtol
    
    # Flatten model parameters into a single vector for Hessian computation
    trajectory = []  # for optional log trajectory
    
    lr0 = lr
    conv = False; oob=False
    while conv==False and oob==False and lr>=lr0:#*1e-4: # simple adaptive learning rate
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
            grad_norm = torch.norm(grad)
            if log_grad and it%log_grad==0: 
                print('gradnorm', grad_norm.item(), 'coords', coords)

            # if small_grad or close:
            if grad_norm < tol:
                conv = True
                print(f"Converged after {it} iterations.")
                break

            if bounds is not None:
                if torch.any(coords < bounds['low']) or torch.any(coords > bounds['up']): 
                    # print('out of bounds', coords, bounds)
                    oob = True
                    conv = True
                    print(f"Out of bounds at {it} iterations.")
                    break

            # prev_coords = coords 
        
        # print(trajectory[0])
        # print(trajectory[-1:])

        # if failure to converge, decimate learning rate
        print('partial grad', grad)
        print('partial point', coords)
        lr *= 0.01
        # lr *= 10.0
        # max_iter *= 10

    if not conv and not oob: print(f"Failed to converge. Max iterations exceeded.")
    if conv and oob: conv='oob'

    # load gradients into the model
    for p, g in zip(params, grad1):
        p.grad = g

    return model, loss, trajectory, conv


def optimize_gd(model, 
                input, 
                loss_fn, 
                lr=1e-3, max_iter=100, atol=1e-6, rtol=1e-5, gradtol=1e-5, 
                log_paths=False, 
                log_grad=False,
                bounds=None, **kwargs):
    """
    Optimize using PyTorch's GD to find local minimum.
    """

    optimizer = SGD(model.parameters(), lr=lr)

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
    conv = False; oob=False
    for it in range(max_iter):
        loss = optimizer.step(closure)

        coords = parameters_to_vector(model.parameters()).clone().detach()

        grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.requires_grad])

        # stopping criterion
        grad_norm = torch.norm(grad)
        if log_grad and it%log_grad==0: 
            print('gradnorm', grad_norm.item(), 'coords', coords[active])

        if grad_norm < gradtol: 
            conv=True
            break

        # try:
        #     # assert small_grad
        #     torch.testing.assert_close(
        #         coords, prev_coords, atol=atol, rtol=rtol
        #     )
        #     # if assert passes, break
        #     break
        # except AssertionError:
        #     pass

        # bound check
        if bounds is not None:
            if torch.any(coords[active] < bounds['low']) or torch.any(coords[active] > bounds['up']): 
                # print('out of bounds', coords[active])
                oob = True
                conv = True
                print(f"Out of bounds at {it} iterations.")
                break

        prev_coords = coords
    
    if not conv and not oob: print(f"Failed to converge. Max iterations exceeded.")
    if conv and oob: conv='oob'

    print('last grad norm', torch.norm(grad))

    return model, loss, trajectory, conv  # Return both final point and path