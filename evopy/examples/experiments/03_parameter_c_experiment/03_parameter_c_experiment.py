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

from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pylab import * 

from numpy import matrix, log2
from sklearn.cross_validation import KFold

from evopy.strategies.ori_dses_svc import ORIDSESSVC
from evopy.problems.tr_problem import TRProblem
from evopy.simulators.simulator import Simulator
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear
from evopy.metamodel.dses_svc_linear_meta_model import DSESSVCLinearMetaModel
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.operators.scaling.scaling_dummy import ScalingDummy
from evopy.operators.termination.or_combinator import ORCombinator
from evopy.operators.termination.accuracy import Accuracy
from evopy.operators.termination.generations import Generations
from evopy.operators.termination.convergence import Convergence 
from evopy.helper.timeseries_aggregator import TimeseriesAggregator

from multiprocessing import cpu_count
from evopy.external.playdoh import map as pmap

def get_method():

    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(20, 5))

    meta_model = DSESSVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingDummy(),
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
        beta = 0.5,
        meta_model = meta_model) 

    return method

def process(simulator):
    return simulator.simulate()

simulators_with_s = []

for i in range(0, 25):
    optimizer = get_method()
    problem = TRProblem()
    termination = Generations(50)
    conditions = [Accuracy(problem.optimum_fitness(), 10**-4), Convergence(10**-6)]
    simulators_with_s.append(Simulator(optimizer, problem, ORCombinator(conditions)))

map(process, simulators_with_s)

parameterCs_with_s = []

for simulator in simulators_with_s:
    parameterCs_with_s.append(\
        simulator.optimizer.meta_model.logger.all()['best_parameter_C'])

minimum_parameterCs = TimeseriesAggregator(parameterCs_with_s).get_minimum()
maximum_parameterCs = TimeseriesAggregator(parameterCs_with_s).get_maximum()

parameterCs_with_s, errors_with_s =\
    TimeseriesAggregator(parameterCs_with_s).get_aggregate()

generations_with_s = range(0, len(parameterCs_with_s))

figure_c = plt.figure(figsize=(8,6), dpi=10, facecolor="w", edgecolor="k")
plt.xlabel('Generation')
plt.ylabel('Bester Parameter C')
#plt.xlim([0,50])
plt.ylim([-6, 20])

logit = lambda x : log2(x) if type(x) != type(None) else 0
#parameterCs_with_s = map(logit, parameterCs_with_s)
#errors_with_s = map(logit, errors_with_s)
maximum_parameterCs = map(logit, maximum_parameterCs)
minimum_parameterCs = map(logit, minimum_parameterCs)

ax = figure_c.add_subplot(111)
ax.yaxis.set_major_formatter((ticker.FormatStrFormatter('$2^{%d}$')))

#b1 = errorbar(generations_with_s,\
#   parameterCs_with_s,\
#   marker=".",
#   color="#004997",
#   ecolor="#CCCCCC",
#   linestyle="none",
#   label="mit Skalierung",\
#   yerr=errors_with_s)

b2 = plot(generations_with_s,\
    minimum_parameterCs,\
    marker="x",
    color="g",
    linestyle="none")

b2 = plot(generations_with_s,\
    maximum_parameterCs,\
    marker=".",
    color="#004997",
    linestyle="none")

print minimum_parameterCs
print maximum_parameterCs

xlabel('Generation')
ylabel('Parameter C')

pp = PdfPages("parameterC.pdf")
plt.savefig(pp, format='pdf')
pp.close()

min_min = min(minimum_parameterCs)
max_min = max(minimum_parameterCs)
min_max = min(maximum_parameterCs)
max_max = max(maximum_parameterCs)

def get_method_with_s():

    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 15, 2)],
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
        beta = 0.5,
        meta_model = meta_model) 

    return method

def process(simulator):
    return simulator.simulate()

simulators_with_s = []

for i in range(0, 25):
    optimizer = get_method_with_s()
    problem = TRProblem()
    termination = Generations(50)
    conditions = [Accuracy(problem.optimum_fitness(), 10**-4), Convergence(10**-6)]
    simulators_with_s.append(Simulator(optimizer, problem, ORCombinator(conditions)))

map(process, simulators_with_s)

parameterCs_with_s = []

for simulator in simulators_with_s:
    parameterCs_with_s.append(\
        simulator.optimizer.meta_model.logger.all()['best_parameter_C'])

minimum_parameterCs = TimeseriesAggregator(parameterCs_with_s).get_minimum()
maximum_parameterCs = TimeseriesAggregator(parameterCs_with_s).get_maximum()

parameterCs_with_s, errors_with_s =\
    TimeseriesAggregator(parameterCs_with_s).get_aggregate()

generations_with_s = range(0, len(parameterCs_with_s))

figure_cs = plt.figure(figsize=(8,6), dpi=10, facecolor="w", edgecolor="k")
plt.xlabel('Generation')
plt.ylabel('Bester Parameter C')
#plt.xlim([0,50])
plt.ylim([-6, 20])

logit = lambda x : log2(x) if type(x) != type(None) else 0
#parameterCs_with_s = map(logit, parameterCs_with_s)
#errors_with_s = map(logit, errors_with_s)
maximum_parameterCs = map(logit, maximum_parameterCs)
minimum_parameterCs = map(logit, minimum_parameterCs)

ax = figure_cs.add_subplot(111)
ax.yaxis.set_major_formatter((ticker.FormatStrFormatter('$2^{%d}$')))

#b1 = errorbar(generations_with_s,\
#   parameterCs_with_s,\
#   marker=".",
#   color="#004997",
#   ecolor="#CCCCCC",
#   linestyle="none",
#   label="mit Skalierung",\
#   yerr=errors_with_s)

b2 = plot(generations_with_s,\
    minimum_parameterCs,\
    marker="x",
    color="g",
    linestyle="none")

b2 = plot(generations_with_s,\
    maximum_parameterCs,\
    marker=".",
    color="#004997",
    linestyle="none")

print minimum_parameterCs
print maximum_parameterCs

xlabel('Generation')
ylabel('Parameter C')

pp = PdfPages("parameterC_with_s.pdf")
plt.savefig(pp, format='pdf')
pp.close()

min_min_s = min(minimum_parameterCs)
max_min_s = max(minimum_parameterCs)
min_max_s = min(maximum_parameterCs)
max_max_s = max(maximum_parameterCs)

results = file("parameter_c_results.tex","w")

lines = [
    "\\begin{tabularx}{\\textwidth}{l X X}\n", 
    "\\toprule\n", 
    "\\textbf{Ergebnisse}\\\\\n",
    "\midrule\n",
    "\\textbf{ohne Skalierung} & Maximum & Minimum\\\\\n",
    "max. $C$ & $2^{%i}$ & $2^{%i}$ \\\\\n" % (max_max, min_max), 
    "min. $C$ & $2^{%i}$ & $2^{%i}$ \\\\\n" % (max_min, min_min),
    "\midrule\n",
    "\\textbf{mit z-Transformation} & Maximum & Minimum\\\\\n",
    "max. $C$ & $2^{%i}$ & $2^{%i}$ \\\\\n" % (max_max_s, min_max_s), 
    "min. $C$ & $2^{%i}$ & $2^{%i}$ \\\\\n" % (max_min_s, min_min_s),
    "\\bottomrule\n", 
    "\end{tabularx}\n"]

results.writelines(lines)
results.close()
