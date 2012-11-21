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

from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pylab import * 

from numpy import matrix, log2, sqrt
from sklearn.cross_validation import KFold

from evopy.strategies.ori_dses_svc import ORIDSESSVC
from evopy.simulators.simulator import Simulator
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear
from evopy.metamodel.dses_svc_linear_meta_model import DSESSVCLinearMetaModel
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.operators.scaling.scaling_normalization import ScalingNormalization
from evopy.operators.scaling.scaling_dummy import ScalingDummy
from evopy.operators.termination.or_combinator import ORCombinator
from evopy.operators.termination.accuracy import Accuracy
from evopy.operators.termination.generations import Generations
from evopy.operators.termination.convergence import Convergence 
from evopy.helper.timeseries_aggregator import TimeseriesAggregator

# problems
from evopy.problems.tr_problem import TRProblem
from evopy.problems.schwefels_problem_12 import SchwefelsProblem12
from evopy.problems.schwefels_problem_26 import SchwefelsProblem26
from evopy.problems.schwefels_problem_240 import SchwefelsProblem240

from multiprocessing import cpu_count
from evopy.external.playdoh import map as pmap

def get_method_TR(scaling):

    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(20, 5))

    meta_model = DSESSVCLinearMetaModel(\
        window_size = 10,
        scaling = scaling(),
        crossvalidation = sklearn_cv,
        repair_mode = 'mirror')

    method = ORIDSESSVC(\
        mu = 15,
        lambd = 100,
        theta = 0.3,
        pi = 15,
        initial_sigma = matrix([[4.5, 4.5]]),
        delta = 4.5,
        tau0 = 0.5, 
        tau1 = 0.6,
        initial_pos = matrix([[10.0, 10.0]]),
        beta = 0.5,
        meta_model = meta_model) 

    return method

def get_method_Schwefel240(scaling):

    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(20, 5))

    meta_model = DSESSVCLinearMetaModel(\
        window_size = 10,
        scaling = scaling(),
        crossvalidation = sklearn_cv,
        repair_mode = 'mirror')

    method = ORIDSESSVC(\
        mu = 15,
        lambd = 100,
        theta = 0.3,
        pi = 15,
        initial_sigma = matrix([[4.5, 4.5, 4.5, 4.5, 4.5]]),
        delta = 4.5,
        tau0 = sqrt(2*5), 
        tau1 = sqrt(2*sqrt(5)),
        initial_pos = matrix([[10.0, 10.0, 10.0, 10.0, 10.0]]),
        beta = 0.5,
        meta_model = meta_model) 

    return method

def get_method_Schwefel26(scaling):

    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(20, 5))

    meta_model = DSESSVCLinearMetaModel(\
        window_size = 10,
        scaling = scaling(),
        crossvalidation = sklearn_cv,
        repair_mode = 'mirror')

    method = ORIDSESSVC(\
        mu = 15,
        lambd = 100,
        theta = 0.3,
        pi = 15,
        initial_sigma = matrix([[4.5, 4.5]]),
        delta = 4.5,
        tau0 = 0.5, 
        tau1 = 0.6,
        initial_pos = matrix([[100.0, 100.0]]),
        beta = 0.5,
        meta_model = meta_model) 

    return method

scaling = [ScalingStandardscore, ScalingNormalization]
problems = [TRProblem, SchwefelsProblem26]
optimizers = [get_method_TR, get_method_Schwefel26]
termination = Generations(50)
simulators_std = []
simulators_nor = []
max_C_std = []
min_C_std =[]
max_C_nor = []
min_C_nor = []

for problem in problems:
    index = problems.index(problem)
    simulators_std_for_problem = []        
    simulators_nor_for_problem = []
    for i in range(0, 25):
        optimizer = optimizers[index]
        simulator = Simulator(optimizer(scaling[0]), problem(), termination)
        simulators_std_for_problem.append(simulator)
        simulator = Simulator(optimizer(scaling[1]), problem(), termination)
        simulators_nor_for_problem.append(simulator)
    simulators_std.append(simulators_std_for_problem)
    simulators_nor.append(simulators_nor_for_problem)

for simulators_for_problem in simulators_std:
    for simulator in simulators_for_problem:
        simulator.simulate()
for simulators_for_problem in simulators_nor:
    for simulator in simulators_for_problem:
        simulator.simulate()

for simulators_for_problem in simulators_std:
    index = simulators_std.index(simulators_for_problem)
    parameterCs_best = []
    for simulator in simulators_for_problem:
        bestc = simulator.optimizer.meta_model.logger.all()['best_parameter_C']
        parameterCs_best.append(bestc)
    min_C_std.append(TimeseriesAggregator(parameterCs_best).get_minimum())
    max_C_std.append(TimeseriesAggregator(parameterCs_best).get_maximum())

for simulators_for_problem in simulators_nor:
    index = simulators_nor.index(simulators_for_problem)
    parameterCs_best = []
    for simulator in simulators_for_problem:
        bestc = simulator.optimizer.meta_model.logger.all()['best_parameter_C']
        parameterCs_best.append(bestc)
    min_C_nor.append(TimeseriesAggregator(parameterCs_best).get_minimum())
    max_C_nor.append(TimeseriesAggregator(parameterCs_best).get_maximum())

logit = lambda x : log2(x) 
filternone = lambda c : type(c) != type(None) 

log_min_C_std = []
log_max_C_std = []
log_min_C_nor = []
log_max_C_nor = []

for l in min_C_std:
    log_min_C_std.append(min(map(logit, filter(filternone, l))))
for l in max_C_std:
    log_max_C_std.append(min(map(logit, filter(filternone, l))))
for l in min_C_nor:
    log_min_C_nor.append(min(map(logit, filter(filternone, l))))
for l in max_C_nor:
    log_max_C_nor.append(min(map(logit, filter(filternone, l))))

results = file("parameter_c_each_results.tex","w")

lines = [
    "\\begin{tabularx}{\\textwidth}{l X X X}\n", 
    "\\toprule\n", 
    "\\textbf{Skalierung} & Problem & Maximum & Minimum \\\\\n",
    "\midrule\n",
    "Normalisierung & TR2 & $2^{%i}$ & $2^{%i}$ \\\\\n" % (log_max_C_nor[0] , log_min_C_nor[0]), 
    "& Schwefels 2.60 & $2^{%i}$ & $2^{%i}$ \\\\\n" % (log_max_C_nor[1], log_min_C_nor[1]),
    "\midrule\n",
    "z-Transformation & TR2 & $2^{%i}$ & $2^{%i}$ \\\\\n" % (log_max_C_std[0] , log_min_C_std[0]), 
    "& Schwefels 2.60 & $2^{%i}$ & $2^{%i}$ \\\\\n" % (log_max_C_std[1], log_min_C_std[1]),

    "\\bottomrule\n", 
    "\end{tabularx}\n"]

results.writelines(lines)
results.close()
