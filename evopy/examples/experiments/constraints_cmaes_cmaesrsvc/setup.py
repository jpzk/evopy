''' 
This file is part of evopy.

Copyright 2012 - 2013, Jendrik Poloczek

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

from sys import path
path.append("../../../..")

from copy import deepcopy
from numpy import matrix, log10

from evopy.strategies.cmaes import CMAES
from evopy.strategies.cmaes_svc import CMAESSVC
from evopy.strategies.cmaes_rsvc import CMAESRSVC
from evopy.simulators.simulator import Simulator

from evopy.problems.sphere_problem_origin_r1 import SphereProblemOriginR1
from evopy.problems.sphere_problem_origin_r2 import SphereProblemOriginR2
from evopy.problems.schwefels_problem_26 import SchwefelsProblem26
from evopy.problems.tr_problem import TRProblem
from evopy.metamodel.rsvc_linear_meta_model import RSVCLinearMetaModel

from sklearn.cross_validation import KFold
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.operators.scaling.scaling_dummy import ScalingDummy
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear

from evopy.operators.termination.or_combinator import ORCombinator
from evopy.operators.termination.accuracy import Accuracy
from evopy.operators.termination.generations import Generations
from evopy.operators.termination.convergence import Convergence 

def get_method_SphereProblemR1():
    method = CMAES(\
        mu = 15,
        lambd = 100,
        xmean = matrix([[10.0, 10.0]]),        
        sigma = 5.0)

    return method

def get_method_SphereProblemR2():
    method = CMAES(\
        mu = 15,
        lambd = 100,
        xmean = matrix([[10.0, 10.0]]),        
        sigma = 5.0)

    return method

def get_method_TR():
    method = CMAES(\
        mu = 15,
        lambd = 100,
        xmean = matrix([[10.0, 10.0]]),        
        sigma = 4.5)

    return method

def get_method_Schwefel26():
    method = CMAES(\
        mu = 15,
        lambd = 100,
        xmean = matrix([[100.0, 100.0]]),        
        sigma = 36.0)

    return method

def get_method_SphereProblemR1_svc():
    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-3, 14, 2)],
        cv_method = KFold(20, 5))

    meta_model = RSVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        repair_mode = 'none')

    method = CMAESRSVC(\
        mu = 15,
        lambd = 100,
        xmean = matrix([[10.0, 10.0]]),
        sigma = 5.0,
        beta = 0.80,
        meta_model = meta_model) 

    return method

def get_method_SphereProblemR2_svc():
    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-3, 14, 2)],
        cv_method = KFold(20, 5))

    meta_model = RSVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        repair_mode = 'none')

    method = CMAESRSVC(\
        mu = 15,
        lambd = 100,
        xmean = matrix([[10.0, 10.0]]),
        sigma = 5.0,
        beta = 0.80,
        meta_model = meta_model) 
    
    return method
    
def get_method_TR_svc():
    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-3, 14, 2)],
        cv_method = KFold(20, 5))

    meta_model = RSVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        repair_mode = 'none')

    method = CMAESRSVC(\
        mu = 15,
        lambd = 100,
        xmean = matrix([[10.0, 10.0]]),
        sigma = 4.5,
        beta = 0.80,
        meta_model = meta_model) 
    
    return method
 
def get_method_Schwefel26_svc():
    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-3, 14, 2)],
        cv_method = KFold(20, 5))

    meta_model = RSVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        repair_mode = 'none')

    method = CMAESRSVC(\
        mu = 15,
        lambd = 100,
        xmean = matrix([[100.0, 100.0]]),
        sigma = 36.0,
        beta = 0.80,
        meta_model = meta_model) 
    
    return method

def create_problem_optimizer_map(typeofelements):
    t = typeofelements    
    return {\
    TRProblem: {get_method_TR: deepcopy(t), get_method_TR_svc: deepcopy(t)},
    SphereProblemOriginR1: {get_method_SphereProblemR1: deepcopy(t), get_method_SphereProblemR1_svc: deepcopy(t)},
    SphereProblemOriginR2: {get_method_SphereProblemR2: deepcopy(t), get_method_SphereProblemR2_svc: deepcopy(t)},
    SchwefelsProblem26: {get_method_Schwefel26: deepcopy(t), get_method_Schwefel26_svc: deepcopy(t)}}

samples = 100
termination = Generations(100)

problems = [TRProblem, SphereProblemOriginR1,\
    SphereProblemOriginR2, SchwefelsProblem26]

optimizers = {\
    TRProblem: [get_method_TR, get_method_TR_svc],
    SphereProblemOriginR1: [get_method_SphereProblemR1, get_method_SphereProblemR1_svc],
    SphereProblemOriginR2: [get_method_SphereProblemR2, get_method_SphereProblemR2_svc],
    SchwefelsProblem26: [get_method_Schwefel26, get_method_Schwefel26_svc]
}

simulators = {\
    TRProblem: {},
    SphereProblemOriginR1: {},
    SphereProblemOriginR2: {},
    SchwefelsProblem26: {}
}

cfcs = create_problem_optimizer_map([])
