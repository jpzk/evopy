''' 
This file is part of evopy.

Copyright 2012, Jendrik Poloczek

evopy is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

evopy is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
evopy.  If not, see <http://www.gnu.org/licenses/>.
'''

from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut

# problems
from evopy.problems.sphere_problem import SphereProblem
from evopy.problems.sa_sphere_problem import SASphereProblem

# strategies
from evopy.strategies.without_constraint_metamodel import WithoutConstraintMetaModel
from evopy.strategies.svc_best_sliding_weighted import SVCBestSlidingWeighted
from evopy.strategies.svc_cv_best_sliding_weighted import SVCCVBestSlidingWeighted
from evopy.strategies.svc_cv_sa_best_weighted import SVCCVSABestSlidingWeighted
from evopy.strategies.svc_cv_ds_best_sliding_weighted import SVCCVDSBestSlidingWeighted

# operators
from evopy.operators.scaling.scaling_dummy import ScalingDummy
from evopy.operators.mutation.gauss_sigma import GaussSigma
from evopy.operators.combination.intermediate import Intermediate
from evopy.operators.combination.sa_intermediate import SAIntermediate
from evopy.operators.selection.smallest_fitness import SmallestFitness
from evopy.operators.selection.smallest_fitness_new_first import SmallestFitnessNewFirst
from evopy.operators.selfadaption.selfadaption import Selfadaption

# views, etc. 
from evopy.views.default_view import DefaultView
from evopy.views.cv_view import CVView
from evopy.views.cv_ds_view import CVDSView
from evopy.metamodel.cv.svc_cv_sklearn_grid import SVCCVSkGrid

"""
method = WithoutConstraintMetaModel(\
    SphereProblem(), 15, 100, 0.5, 1,\
    Intermediate(),\
    GaussSigma(),\
    SmallestFitness(), 
    DefaultView()) 

method.run()

method = SVCBestSlidingWeighted(\
    SphereProblem(),\
    mu = 15,\
    lambd = 100,\
    alpha = 0.5,\
    sigma = 1,\
    combination = Intermediate(),\
    mutation = GaussSigma(),\
    selection = SmallestFitness(),
    view = DefaultView(),
    beta = 0.9,\
    window_size = 25,\
    append_to_window = 25,\
    parameter_C = 1.0,\
    parameter_gamma = 0.0)\

method.run()
"""
sklearn_cv = SVCCVSkGrid(\
    gamma_range = [2 ** i for i in range(-15, 3, 2)],
    C_range = [2 ** i for i in range(-5, 15, 2)],
    cv_method = KFold(50, 5))

method = SVCCVDSBestSlidingWeighted(\
    SASphereProblem(),
    mu = 15,
    lambd = 100,
    alpha = 0.5,
    sigma = 1,
    theta = 0.7,
    pi = 70, 
    epsilon = 1.0,
    combination = SAIntermediate(),\
    mutation = GaussSigma(),\
    selection = SmallestFitnessNewFirst(),
    view = CVDSView(),
    beta = 0.9,
    window_size = 25,
    append_to_window = 25,
    crossvalidation = sklearn_cv,
    scaling = ScalingDummy(),
    selfadaption = Selfadaption())
     
method.run() 
