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
path.append("../../")

from numpy import matrix

from evopy.strategies.ori_dses_svc import ORIDSESSVC
from evopy.problems.tr_problem import TRProblem
from evopy.simulators.simulator import Simulator

from evopy.metamodel.dses_svc_linear_meta_model import DSESSVCLinearMetaModel
from sklearn.cross_validation import KFold
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.operators.scaling.scaling_dummy import ScalingDummy
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear

from evopy.termination.or_combinator import ORCombinator
from evopy.termination.accuracy import Accuracy
from evopy.termination.generations import Generations
from evopy.termination.convergence import Convergence 

from evopy.helper.timeseries_aggregator import TimeseriesAggregator

from multiprocessing import cpu_count
from evopy.external.playdoh import map as pmap

from pylab import * 

def get_method():

    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 5, 2)],
        cv_method = KFold(20, 5))

    meta_model = DSESSVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        repair_mode = 'mirror')

    method = ORIDSESSVC(\
        mu = 15,
        lambd = 100,
        theta = 0.3,
        pi = 70,
        initial_sigma = matrix([[4.5, 4.5]]),
        delta = 4.5,
        tau0 = 0.5, 
        tau1 = 0.6,
        initial_pos = matrix([[10.0, 10.0]]),
        beta = 0.9,
        meta_model = meta_model) 

    return method

def process(simulator):
    return simulator.simulate()

simulators_with_s = []
simulators_without_s = []

for i in range(0, 10):
    optimizer = get_method()
    problem = TRProblem()
    conditions = [Accuracy(problem.optimum_fitness(), 10**-4), Convergence(10**-6)]
    simulators_with_s.append(Simulator(optimizer, problem, ORCombinator(conditions)))

map(process, simulators_with_s)

parameterCs_with_s = []

for simulator in simulators_with_s:
    parameterCs_with_s.append(\
        simulator.optimizer.meta_model.logger.all()['best_parameter_C'])

parameterCs_with_s, errors_with_s =\
    TimeseriesAggregator(parameterCs_with_s).get_aggregate()

generations_with_s = range(0, len(parameterCs_with_s))

b1 = errorbar(generations_with_s,\
    parameterCs_with_s,\
    fmt="g-",\
    label="mit Skalierung",\
    yerr=errors_with_s)

xlabel('Generation')
ylabel('Bester Parameter C')

show()

