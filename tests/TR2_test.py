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

from numpy import matrix
from evopy.strategies.cmaes import CMAES
from evopy.strategies.cmaes_rsvc import CMAESRSVC
from evopy.strategies.cmaes_svc import CMAESSVC
from evopy.metamodel.svc_linear_meta_model import SVCLinearMetaModel
from evopy.metamodel.cma_svc_linear_meta_model import CMASVCLinearMetaModel
from evopy.problems.tr_problem import TRProblem
from evopy.simulators.simulator import Simulator
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear

from sklearn.cross_validation import KFold

def get_method_cmaessvc():
    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 5, 2)],
        cv_method = KFold(20, 5))

    meta_model = CMASVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        repair_mode = 'mirror')

    method = CMAESSVC(\
        mu = 15,
        lambd = 100,
        xmean = matrix([[5.0, 5.0]]),
        sigma = 1.0,
        beta = 0.9,
        meta_model = meta_model) 

    return method

def get_method_cmaesrsvc():
    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(20, 5))

    meta_model = SVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        repair_mode = 'mirror')

    method = CMAESRSVC(\
        mu = 15,
        lambd = 100,
        xmean = matrix([[5.0, 5.0]]),
        sigma = 1.0,
        beta = 0.80,
        meta_model = meta_model)

    return method

def get_method_cmaes():
    method = CMAES(\
        mu = 15,
        lambd = 100,
        xmean = matrix([[5.0, 5.0]]),
        sigma = 1.0)
    return method

def CMAES_TR2_simulation_test():
    sim = Simulator(get_method_cmaes(), TRProblem(), pow(10, -12))
    results = sim.simulate()
    assert True  

def CMAESSVC_TR2_simulation_test():
    sim = Simulator(get_method_cmaessvc(), TRProblem(), pow(10, -12))
    results = sim.simulate()
    assert True  

def CMAESRSVC_TR2_simulation_test():
    sim = Simulator(get_method_cmaesrsvc(), TRProblem(), pow(10, -12))
    results = sim.simulate()
    assert True  

 
