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
from scipy.stats import wilcoxon

from numpy import matrix, log2, sqrt
from sklearn.cross_validation import KFold
from copy import deepcopy

from evopy.strategies.ori_dses_aligned_svc import ORIDSESAlignedSVC
from evopy.strategies.ori_dses import ORIDSES
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

def get_method_TR():
    method = ORIDSES(\
        mu = 15,
        lambd = 100,
        theta = 0.3,
        pi = 70,
        initial_sigma = matrix([[4.5, 4.5]]),
        delta = 4.5,
        tau0 = 0.5, 
        tau1 = 0.6,
        initial_pos = matrix([[10.0, 10.0]])) 

    return method

def get_method_TR_ali():
    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(20, 5))

    meta_model = DSESSVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        repair_mode = 'none')

    method = ORIDSESAlignedSVC(\
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

def get_method_Schwefel26():
    method = ORIDSES(\
        mu = 15,
        lambd = 100,
        theta = 0.3,
        pi = 70,
        initial_sigma = matrix([[4.5, 4.5]]),
        delta = 4.5,
        tau0 = 0.5, 
        tau1 = 0.6,
        initial_pos = matrix([[100.0, 100.0]])) 

    return method

def get_method_Schwefel26_ali():
    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(20, 5))

    meta_model = DSESSVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        repair_mode = 'none')

    method = ORIDSESAlignedSVC(\
        mu = 15,
        lambd = 100,
        theta = 0.3,
        pi = 70,
        initial_sigma = matrix([[4.5, 4.5]]),
        delta = 4.5,
        tau0 = 0.5, 
        tau1 = 0.6,
        initial_pos = matrix([[100.0, 100.0]]),
        beta = 0.9,
        meta_model = meta_model) 

    return method

def create_problem_optimizer_map(typeofelements):
    t = typeofelements    
    return {\
    TRProblem: {get_method_TR: deepcopy(t), get_method_TR_ali: deepcopy(t)},
    SchwefelsProblem26: {get_method_Schwefel26: deepcopy(t), get_method_Schwefel26_ali: deepcopy(t)}}

samples = 25 
termination = Generations(50)
problems = [TRProblem, SchwefelsProblem26]

optimizers = {\
    TRProblem: [get_method_TR, get_method_TR_ali],
    SchwefelsProblem26: [get_method_Schwefel26, get_method_Schwefel26_ali]
}
simulators = {\
    TRProblem: {},
    SchwefelsProblem26: {}
}

cfc_per_generation = create_problem_optimizer_map([])
cfc_sum = create_problem_optimizer_map([])
cfc_agg = create_problem_optimizer_map({})

# create simulators
for problem in problems:
    for optimizer in optimizers[problem]:
        simulators_op = []
        for i in range(0, samples):
            simulator = Simulator(optimizer(), problem(), termination)
            simulators_op.append(simulator)
        simulators[problem][optimizer] = simulators_op

# run simulators, get important info
for problem in problems:
    for optimizer, simulators_ in simulators[problem].iteritems():
        for simulator in simulators_:
            simulator.simulate()
            ccfc = simulator.logger.all()['count_cfc']
            cfcsum = sum(ccfc)
            cfc_per_generation[problem][optimizer].append(ccfc)
            cfc_sum[problem][optimizer].append(cfcsum)

# aggregate cfc_series 
for problem in problems:
    for optimizer in optimizers[problem]:
        cfclist = cfc_per_generation[problem][optimizer]
        cfcs_serie, cfcs_error = TimeseriesAggregator(cfclist).get_aggregate()
        cfc_agg[problem][optimizer]['cfcs_serie'] = cfcs_serie
        cfc_agg[problem][optimizer]['cfcs_error'] = cfcs_error

# statistics
variable_names = ['min', 'max', 'mean', 'var']
variables = {}
for variable in variable_names:
    variables[variable] = create_problem_optimizer_map(0.0)

for problem in problems:
    for optimizer in optimizers[problem]:
        agg_serie = cfc_agg[problem][optimizer]['cfcs_serie']
        agg_error = cfc_agg[problem][optimizer]['cfcs_error']
        variables['min'][problem][optimizer] = min(agg_serie)
        variables['max'][problem][optimizer] = max(agg_serie)
        variables['mean'][problem][optimizer] = array(agg_serie).mean()
        variables['var'][problem][optimizer] = array(agg_error).mean()

results = file("dses_ndalignment_with_without_each_results.tex","w")

zTR, pTR = wilcoxon(\
    cfc_agg[TRProblem][get_method_TR]['cfcs_serie'],
    cfc_agg[TRProblem][get_method_TR_ali]['cfcs_serie'])

zs26, ps26 = wilcoxon(\
    cfc_agg[SchwefelsProblem26][get_method_Schwefel26]['cfcs_serie'],
    cfc_agg[SchwefelsProblem26][get_method_Schwefel26]['cfcs_serie'])

lines = [
    "\\begin{tabularx}{\\textwidth}{l X X X X X X}\n", 
    "\\toprule\n", 
    "\\textbf{Problem} & Anpassung & Minimum & Mittel & Maximum & Varianz  \\\\\n",
    "\midrule\n",
    "TR2 & nein & $%i$ & $%i$ & $%i$ & $%f$ \\\\\n"\
        % (variables['min'][TRProblem][get_method_TR],\
        variables['mean'][TRProblem][get_method_TR],\
        variables['max'][TRProblem][get_method_TR],\
        variables['var'][TRProblem][get_method_TR]),\
    "& ja & $%i$ & $%i$ & $%i$ & $%f$ \\\\\n"\
        % (variables['min'][TRProblem][get_method_TR_ali],\
        variables['mean'][TRProblem][get_method_TR_ali],\
        variables['max'][TRProblem][get_method_TR_ali],\
        variables['var'][TRProblem][get_method_TR_ali]),\
    "2.6 & nein & $%i$ & $%i$ & $%i$ & $%f$ \\\\\n"\
        % (variables['min'][SchwefelsProblem26][get_method_Schwefel26],\
        variables['mean'][SchwefelsProblem26][get_method_Schwefel26],\
        variables['max'][SchwefelsProblem26][get_method_Schwefel26],\
        variables['var'][SchwefelsProblem26][get_method_Schwefel26]),\
    "& ja & $%i$ & $%i$ & $%i$ & $%f$ \\\\\n"\
        % (variables['min'][SchwefelsProblem26][get_method_Schwefel26_ali],\
        variables['mean'][SchwefelsProblem26][get_method_Schwefel26_ali],\
        variables['max'][SchwefelsProblem26][get_method_Schwefel26_ali],\
        variables['var'][SchwefelsProblem26][get_method_Schwefel26_ali]),\
    "\\bottomrule\n", 
    "\end{tabularx}\n"]

results.writelines(lines)
results.close()

