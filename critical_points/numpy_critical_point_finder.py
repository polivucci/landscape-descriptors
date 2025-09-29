import numpy as np

from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.stats.qmc import Sobol

class MinimaFinder:

    def __init__(self, bounds, dimension=2, min_distance=0.01, m=64):
        self.bounds = np.array(bounds)
        self.dimension = dimension
        self.min_distance = min_distance
        self.m = m
        self.minima = []
        self.attempt_history = []
        self.kdtree = None
        
        self.generate_starting_points()

    def add_minimum(self, point, value):
        point = np.array(point)
        print(point, 'Reject:', self._is_too_close(point))
        if not self._is_too_close(point):
            self.minima.append((point, value))
            self.update_kdtree()
            return True
        return False

    def _is_too_close(self, point):
        if not self.minima:
            return False
        if self.kdtree is None:
            self.update_kdtree()
        distances, indices = self.kdtree.query([point], k=1)
        if distances[0] < self.min_distance: return True
        return False

    def _is_within_bounds(self, point, tolerance=1e-3):
        if np.any(point-self.bounds)<tolerance: 
            return True
        return False

    def update_kdtree(self):
        if self.minima:
            self.kdtree = KDTree([m[0] for m in self.minima])
    
    def generate_starting_points(self):
        sampler = Sobol(d=self.dimension, scramble=True, seed=42)
        log2m = np.rint(np.log2(self.m)).astype(np.int32)
        self.x0s = sampler.random_base2(m=log2m)

def run_local_search(x0, func, grad_func, bounds):
        
        res = minimize(
            func,
            x0,
            method='L-BFGS-B',
            jac=grad_func,
            bounds=bounds,
            # options={'disp': 1}
        )

        if res.success:
            return res.x, func(res.x)
        return None, None

# --- Modified optimization runner to allow dynamic target functions ---
def run_search(finder, func, grad_func):

    bounds = [tuple(finder.bounds)] * finder.dimension

    def wrapped_run(i):
        x0 = finder.x0s[i]
        print('x0', x0)
        finder.attempt_history.append(x0)
        point, value = run_local_search(x0, func, grad_func, bounds)
        if point is not None: finder.add_minimum(point, value)

    for i in range(finder.m): wrapped_run(i)

    results = finder.minima

    # if n_jobs == 1:
    # else:
    #     results = Parallel(n_jobs=n_jobs)(
    #     delayed(wrapped_run)(i) for i in range(1, num_attempts + 1))
    return results

def _critical_point_index(hessian, tol=1e-9):
    '''Calculates the index of a critical point, i.e. the number of positive eigenvalues 
    index = card(eigval(hessian)>0)
    Zero eigenvalues are handled as well.
    '''
    eigvals = np.linalg.eigvals(hessian)
    # print('eigv', eigvals)
    index = np.sum((eigvals<-tol), dtype=int)
    # print('index', index)
    zerotest = (eigvals>-tol) * (eigvals<tol)
    # print(zerotest)
    if np.any(zerotest):
        return -1
    else:
        return index

def find_critical_points(squared_gradient_norm, 
                         squared_gradient_norm_grad,
                         hess_func,
                         bounds,  
                         num_attempts=64, 
                         dimension=2, 
                         min_distance=0.1,
                         known_minima=None, 
                         n_jobs=1, 
                         ):
    
    finder = MinimaFinder(bounds, dimension, min_distance, num_attempts)
    
    results = run_search(finder, squared_gradient_norm, squared_gradient_norm_grad)

    # Filter to critical points (where ||grad||² ≈ 0)
    critical_points = []
    maxima = []
    minima = []
    for result in results:
        x, val = result
        if x is not None and val < 1e-9:  # ||∇f||² ~ 0

            point = x

            if known_minima:
                # Compare against known minima
                too_close = any(np.linalg.norm(point - np.array(m[0])) < min_distance/2 
                                for m in known_minima)
                if not too_close:
                    critical_points.append((point, val))
            else:

                hessian = hess_func(point)
                d = point.shape[0]

                tol = 1e-9 # tolerance around zero 
                index = _critical_point_index(hessian, tol=1e-9)
                if index==d:
                    maxima.append((point, val, index))
                    print('maximum', point)
                elif index==0:
                    minima.append((point, val, index))
                    print('minimum', point)
                else:
                    critical_points.append((point, val, index))
                    print('cp', point, index)

    return minima, maxima, critical_points


if __name__ == "__main__":

    # toy function and derivatives
    def f_numpy(x, eps=0.1):
        """
        Polynomial function f(x,y) = (x^2 - eps)^2 + y^2
        Has two minima at (±sqrt(eps),0) and a saddle at (0,0)
        """
        x, y = x[0], x[1]
        return (x**2 - eps)**2 + y**2

    def f_grad_numpy(x, eps=0.1):
        x, y = x[0], x[1]
        dfdx = 4 * x * (x**2 - eps)
        dfdy = 2 * y
        return np.array([dfdx, dfdy])

    def f_hessian_numpy(x, eps=0.1):
        x, y = x[0], x[1]
        d2fdx2 = 12 * x**2 - 4 * eps
        d2fdxdy = 0
        d2fdy2 = 2
        return np.array([[d2fdx2, d2fdxdy],
                        [d2fdxdy, d2fdy2]])

    # Set up search
    BOUNDS = (-1.0, 1.0)        # better not change too much. normalise inputs instead.
    DIMENSION = 2               # number of parameters (dimension)
    ATTEMPTS = 64               # number of generated random initial conditions (must be power of 2)
    MIN_DISTANCE = 0.01         # tolerance for keeping new critical point 

    # Find critical points:
    minima, maxima, saddle_points = find_critical_points(f_numpy, 
                                                         f_grad_numpy, 
                                                         f_hessian_numpy,
                                                         bounds=BOUNDS,
                                                         dimension=DIMENSION,
                                                         num_attempts=ATTEMPTS,
                                                         min_distance=MIN_DISTANCE,
                                                         known_minima=None,
                                                        )


    print(f"\nFound {len(minima)} potential minima:")
    for i, (point, val, index) in enumerate(minima, 1):
        print(f"Minimum {i}: Position {point}, ||∇f||² = {val:.2e}")

    print(f"\nFound {len(saddle_points)} potential saddle points:")
    for i, (point, val, index) in enumerate(saddle_points, 1):
        print(f"Saddle {i}: Position {point}, ||∇f||² = {val:.2e}, index={index:d}")

    # output to file
    from pandas import DataFrame
    f_values = []
    for point_type, coord_list in zip(['minimum', 'saddle'], [minima, saddle_points]):
        vardict = lambda x: {f'x{i+1:d}': x[i] for i in range(DIMENSION)}
        f_values += [{**vardict(x), 'f_value': f_numpy(x), 'type': point_type, 'index': index} for x, fval, index in coord_list]
    results_df = DataFrame(f_values)
    print(results_df)
    results_df.to_csv(f"critical_points_{DIMENSION:d}D.csv", float_format='%.10f')