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
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear

from evopy.strategies.cmaes import CMAES
from evopy.problems.tr_problem import TRProblem
from evopy.simulators.simulator import Simulator

def get_method():
    method = CMAES(\
        mu = 15,
        lambd = 100,
        xmean = matrix([[5.0, 5.0]]),        
        sigma = 1.0)

    return method

if __name__ == "__main__":
    sim = Simulator(get_method(), TRProblem(), pow(10, -12))
    results = sim.simulate()
