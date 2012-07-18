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

from multiprocessing import cpu_count
from evopy.external.playdoh import map as pmap

from sklearn.cross_validation import KFold
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear
from evopy.metamodel.cma_svc_linear_meta_model import CMASVCLinearMetaModel

from evopy.simulators.csv_writer import CSVWriter
from evopy.simulators.experiment_simulator import ExperimentSimulator
from evopy.strategies.cmaes_rsvc import CMAESRSVC
from evopy.strategies.cmaes import CMAES
from evopy.problems.tr_problem import TRProblem

def get_cmaes_rsvc_method():
    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(20, 5))

    meta_model = CMASVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        repair_mode = 'mirror')

    method = CMAESRSVC(\
        mu = 15,
        lambd = 100,
        xmean = [5.0, 5.0],
        sigma = 1.0,
        beta = 0.80,
        meta_model = meta_model)

    return method

def get_cmaes_method():
    method = CMAES(\
        mu = 15,
        lambd = 100,
        xmean = [5.0, 5.0],
        sigma = 1.0)

    return method

writer = CSVWriter("test")

def process(simulator):
    writer.write(simulator.simulate())

simulators = []
for i in range(0, 50):
    simulator1 = ExperimentSimulator(get_cmaes_rsvc_method(), TRProblem(), 10**-12)
    simulator2 = ExperimentSimulator(get_cmaes_method(), TRProblem(), 10**-12)
    simulators.append(simulator1)
    simulators.append(simulator2)

pmap(process, simulators, cpu = cpu_count)
#map(process, simulators)
