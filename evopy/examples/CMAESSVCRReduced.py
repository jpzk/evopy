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
path.append("../..")

from sklearn.cross_validation import KFold
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear
from evopy.strategies.cmaes_svc_repair_reduced import CMAESSVCRReduced
from evopy.metamodel.cma_svc_linear_meta_model import CMASVCLinearMetaModel
from evopy.metamodel.svc_linear_meta_model import SVCLinearMetaModel
from evopy.problems.tr_problem import TRProblem
from evopy.simulators.simulator import Simulator

def get_method():

    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(20, 5))

    meta_model = CMASVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        repair_mode = 'mirror')

    meta_model_dr = SVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        repair_mode = 'mirror')

    method = CMAESSVCRReduced(\
        mu = 15,
        lambd = 100,
        xmean = [5.0, 5.0],
        sigma = 1.0,
        beta = 0.80,
        meta_model = meta_model)

    return method

if __name__ == "__main__":
    sim = Simulator(get_method(), TRProblem(), pow(10, -12))
    sim.simulate()
