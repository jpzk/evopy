from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut

from evopy.problems.sphere_problem import SphereProblem
from evopy.strategies.without_constraint_metamodel import WithoutConstraintMetaModel
from evopy.strategies.svc_best_sliding_weighted import SVCBestSlidingWeighted
from evopy.strategies.svc_cv_best_sliding_weighted import SVCCVBestSlidingWeighted
from evopy.metamodel.cv.svc_cv_sklearn_grid import SVCCVSkGrid
from evopy.scaling.scaling_dummy import ScalingDummy

#method = WithoutConstraintMetaModel(\
#    SphereProblem(), 15, 100, 0.5, 1) 
"""
method = SVCBestSlidingWeighted(\
    SphereProblem(),
    mu = 15,
    lambd = 100,
    alpha = 0.5,
    sigma = 1,
    beta = 0.9,
    window_size = 25,
    append_to_window = 25,
    parameter_C = 1.0,
    parameter_gamma = 0.0)
 """

sklearn_cv = SVCCVSkGrid(\
    gamma_range = [2 ** i for i in range(-15, 3, 2)],
    C_range = [2 ** i for i in range(-5, 15, 2)],
    cv_method = KFold(50, 5))

method = SVCCVBestSlidingWeighted(\
    SphereProblem(),
    mu = 15,
    lambd = 100,
    alpha = 0.5,
    sigma = 1,
    beta = 0.9,
    window_size = 25,
    append_to_window = 25,
    crossvalidation = sklearn_cv,
    scaling = ScalingDummy())
     
method.run() 
