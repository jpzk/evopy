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
from matplotlib import ticker
from scipy.stats import wilcoxon

from evopy.strategies.ori_dses_aligned_svc import ORIDSESAlignedSVC
from evopy.strategies.ori_dses_svc_repair import ORIDSESSVCR
from evopy.strategies.ori_dses_svc import ORIDSESSVC
from evopy.strategies.ori_dses import ORIDSES
from evopy.problems.tr_problem import TRProblem
from evopy.simulators.simulator import Simulator
from evopy.problems.schwefels_problem_26 import SchwefelsProblem26

from evopy.metamodel.dses_svc_linear_meta_model import DSESSVCLinearMetaModel
from sklearn.cross_validation import KFold
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.operators.scaling.scaling_dummy import ScalingDummy
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear

from evopy.operators.termination.or_combinator import ORCombinator
from evopy.operators.termination.accuracy import Accuracy
from evopy.operators.termination.generations import Generations
from evopy.operators.termination.convergence import Convergence 

from evopy.helper.timeseries_aggregator import TimeseriesAggregator

from multiprocessing import cpu_count
from evopy.external.playdoh import map as pmap

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pylab import * 

def get_method_TR():
    method = ORIDSES(\
        mu = 15,
        lambd = 100,
        theta = 0.3,
        pi = 15,
        initial_sigma = matrix([[4.5, 4.5]]),
        delta = 4.5,
        tau0 = 0.5, 
        tau1 = 0.6,
        initial_pos = matrix([[10.0, 10.0]])) 

    return method

def get_method_TR_svc():
    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-3, 9, 2)],
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
        pi = 15,
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
        pi = 15,
        initial_sigma = matrix([[4.5, 4.5]]),
        delta = 4.5,
        tau0 = 0.5, 
        tau1 = 0.6,
        initial_pos = matrix([[100.0, 100.0]])) 

    return method

def get_method_Schwefel26_svc():
    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-3, 9, 2)],
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
        pi = 15,
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
    TRProblem: {get_method_TR: deepcopy(t), get_method_TR_svc: deepcopy(t)},
    SchwefelsProblem26: {get_method_Schwefel26: deepcopy(t), get_method_Schwefel26_svc: deepcopy(t)}}

samples = 100 
termination = Generations(50)
problems = [TRProblem, SchwefelsProblem26]

optimizers = {\
    TRProblem: [get_method_TR, get_method_TR_svc],
    SchwefelsProblem26: [get_method_Schwefel26, get_method_Schwefel26_svc]
}

simulators = {\
    TRProblem: {},
    SchwefelsProblem26: {}
}

logit = lambda value, optimum : log10(value - optimum)

best_fitness = create_problem_optimizer_map([])
mean_fitness = create_problem_optimizer_map([])
agg_mean_fitness = create_problem_optimizer_map({})

# create simulators
for problem in problems:
    for optimizer in optimizers[problem]:
        simulators_op = []
        for i in range(0, samples):
            simulator = Simulator(optimizer(), problem(), termination)
            simulators_op.append(simulator)
        simulators[problem][optimizer] = simulators_op

# run simulators 
for problem in problems:
    for optimizer, simulators_ in simulators[problem].iteritems():
        for simulator in simulators_:
            simulator.simulate()
            fitness = simulator.optimizer.logger.all()['best_fitness'][-1]
            means = simulator.optimizer.logger.all()['mean_fitness']
            best_fitness[problem][optimizer].append(fitness)
            mean_fitness[problem][optimizer].append(means) 

# statistics
variable_names = ['min', 'max', 'mean', 'var']
variables = {}
for variable in variable_names:
    variables[variable] = create_problem_optimizer_map(0.0)

for problem in problems:
    for optimizer in optimizers[problem]:
        best_fitnesses = best_fitness[problem][optimizer]
        variables['min'][problem][optimizer] = min(best_fitnesses)
        variables['max'][problem][optimizer] = max(best_fitnesses)
        variables['mean'][problem][optimizer] = array(best_fitnesses).mean()
        variables['var'][problem][optimizer] = array(best_fitnesses).var()

# aggregate to mean best fitnesses
for problem in problems:
    for optimizer in optimizers[problem]:
        means = mean_fitness[problem][optimizer]
        series, error = TimeseriesAggregator(means).get_aggregate()
        agg_mean_fitness[problem][optimizer]['series'] = series
        agg_mean_fitness[problem][optimizer]['error'] = error 
        
figure_cs = plt.figure(figsize=(8,6), dpi=10, facecolor="w", edgecolor="k")
ax = figure_cs.add_subplot(111)
ax.yaxis.set_major_formatter((ticker.FormatStrFormatter('$2^{%d}$')))

optimum_fitness = problem().optimum_fitness()
log_series = map(logit, series, [optimum_fitness] * len(series))

# TR plot
#TR_without_svc = agg_mean_fitness[TRProblem][get_method_TR]
#b1 = plot(range(0, agg_mean_fitness), 
#    map(log_series, 

# table output
zTR, pTR = wilcoxon(\
    best_fitness[TRProblem][get_method_TR], 
    best_fitness[TRProblem][get_method_TR_svc])

zs26, ps26 = wilcoxon(\
    best_fitness[SchwefelsProblem26][get_method_Schwefel26], 
    best_fitness[SchwefelsProblem26][get_method_Schwefel26_svc])

results = file("fitness_with_without_each_results.tex","w")
lines = [
    "\\begin{tabularx}{\\textwidth}{l  X X X X X}\n", 
    "\\toprule\n", 
    "\\textbf{Problem} & Anpassung & Minimum & Mittel & Maximum & Varianz\\\\\n",
    "\midrule\n",
    "TR2 & nein & $%0.10f$ & $%0.10f$ & $%0.10f$ & %1.2e \\\\\n"\
        % (variables['min'][TRProblem][get_method_TR],\
        variables['mean'][TRProblem][get_method_TR],\
        variables['max'][TRProblem][get_method_TR],\
        variables['var'][TRProblem][get_method_TR]),\
    "& ja & $%0.10f$ & $%0.10f$ & $%0.10f$ & %1.2e \\\\\n"\
        % (variables['min'][TRProblem][get_method_TR_svc],\
        variables['mean'][TRProblem][get_method_TR_svc],\
        variables['max'][TRProblem][get_method_TR_svc],\
        variables['var'][TRProblem][get_method_TR_svc]),\
    "2.60 & nein & $%0.10f$ & $%0.10f$ & $%0.10f$ & %1.2e \\\\\n"\
        % (variables['min'][SchwefelsProblem26][get_method_Schwefel26],\
        variables['mean'][SchwefelsProblem26][get_method_Schwefel26],\
        variables['max'][SchwefelsProblem26][get_method_Schwefel26],\
        variables['var'][SchwefelsProblem26][get_method_Schwefel26]),\
    "& ja & $%0.10f$ & $%0.10f$ & $%0.10f$ & %1.2e \\\\\n"\
        % (variables['min'][SchwefelsProblem26][get_method_Schwefel26_svc],\
        variables['mean'][SchwefelsProblem26][get_method_Schwefel26_svc],\
        variables['max'][SchwefelsProblem26][get_method_Schwefel26_svc],\
        variables['var'][SchwefelsProblem26][get_method_Schwefel26_svc]),\
    "\\bottomrule\n", 
    "\end{tabularx}\n"]

results.writelines(lines)
results.close()

