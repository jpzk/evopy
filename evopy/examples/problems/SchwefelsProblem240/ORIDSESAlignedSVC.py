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

from sys import path
path.append("../../../..")

from numpy import matrix
from sklearn.cross_validation import KFold

from evopy.strategies.ori_dses_aligned_svc import ORIDSESAlignedSVC
from evopy.problems.schwefels_problem_240 import SchwefelsProblem240
from evopy.simulators.simulator import Simulator
from evopy.metamodel.dses_svc_linear_meta_model import DSESSVCLinearMetaModel
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear
from evopy.operators.termination.accuracy import Accuracy

def get_method():

    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-1, 14, 2)],
        cv_method = KFold(20, 5))

    meta_model = DSESSVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        repair_mode = 'mirror')

    method = ORIDSESAlignedSVC(\
        mu = 15,
        lambd = 100,
        theta = 0.3,
        pi = 70,
        initial_sigma = matrix([[4.5, 4.5, 4.5, 4.5, 4.5]]),
        delta = 4.5,
        tau0 = 0.5, 
        tau1 = 0.6,
        initial_pos = matrix([[10.0, 10.0, 10.0, 10.0, 10.0]]),
        beta = 1.0,
        meta_model = meta_model) 

    return method

if __name__ == "__main__":
    problem = SchwefelsProblem240() 
    optimizer = get_method()
    print optimizer.description
    print problem.description
    optfit = problem.optimum_fitness()
    sim = Simulator(optimizer, problem, Accuracy(optfit, 10**(-6)))
    results = sim.simulate()
