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
from sys import argv
path.append("../../../..")

from numpy import matrix, array
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pylab import * 
from scipy.stats import wilcoxon

from evopy.strategies.ori_dses import ORIDSES 
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut

from evopy.strategies.ori_dses_svc_repair import ORIDSESSVCR
from evopy.strategies.ori_dses_svc import ORIDSESSVC
from evopy.problems.tr_problem import TRProblem
from evopy.problems.schwefels_problem_26 import SchwefelsProblem26
from evopy.simulators.simulator import Simulator
from evopy.simulators.experiment_simulator import ExperimentSimulator
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear
from evopy.metamodel.dses_svc_linear_meta_model import DSESSVCLinearMetaModel
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.operators.scaling.scaling_dummy import ScalingDummy
from evopy.operators.termination.or_combinator import ORCombinator
from evopy.operators.termination.accuracy import Accuracy
from evopy.operators.termination.generations import Generations
from evopy.operators.termination.convergence import Convergence 
from evopy.helper.timeseries_aggregator import TimeseriesAggregator

def get_method_without_SVC():
    method = ORIDSES(\
        mu = 15,
        lambd = 100,
        theta = 0.3,
        pi = 70,
        initial_sigma = matrix([[4.5, 4.5]]),
        delta = 4.5,
        tau0 = 0.5, 
        tau1 = 0.6,
        initial_pos = matrix([[200.0, 200.0]])) 

    return method

def get_method_with_SVC():

    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-1, 14, 2)],
        cv_method = KFold(20,5))

    meta_model = DSESSVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        repair_mode = 'none')

    method = ORIDSESSVC(\
        mu = 15,
        lambd = 100,
        theta = 0.3,
        pi = 70,
        initial_sigma = matrix([[4.5, 4.5]]),
        delta = 4.5,
        tau0 = 0.5, 
        tau1 = 0.6,
        initial_pos = matrix([[200.0, 200.0]]),
        beta = 1.0,
        meta_model = meta_model) 

    return method

method_names = ["DSES", "DSES-SVC"]
methods = [get_method_without_SVC, get_method_with_SVC]
method_marker = ['.','x']
method_color = ['#004997', 'g']

simulators = []
cfcs_sums = []
cfcs = []
cfcs_series = []
cfcs_errors = []
cum_cfcs_series = []
cum_cfcs_errors = []

for method in method_names:
    simulators_for_method = []
    index = method_names.index(method)
    for i in range(0, 25):
        optimizer = methods[index]()
        problem = SchwefelsProblem26(dimensions=2)
        conditions = [Generations(50)]
        simulators_for_method.append(\
            Simulator(optimizer, problem, ORCombinator(conditions)))
    simulators.append(simulators_for_method)

for method in method_names:
    index = method_names.index(method)
    cfc_sum_for_method = []
    cfc_for_method = []
   
    for simulator in simulators[index]:
        simulator.simulate()
        cfc_sum_for_method.append(sum(simulator.logger.all()['count_cfc'])) 
        cfc_for_method.append(simulator.logger.all()['count_cfc'])

    cfcs_sums.append(cfc_sum_for_method)
    cfcs.append(cfc_for_method)             

# boxplots
figure_boxplot = plt.figure(figsize=(8,6), dpi=10, facecolor="w", edgecolor="k")

plt.ylabel("Kumulierte Restriktionsaufrufe nach Generation")
plt.xticks(range(0, len(method_names)), method_names)
box = plt.boxplot(cfcs_sums) 

setp(box['boxes'], color="#004997")
setp(box['medians'], color="g")

pp = PdfPages("boxplot.pdf")
plt.savefig(pp, format='pdf')
pp.close()

# cfc time series
for method in method_names:
    index = method_names.index(method)
    cfcs_serie, cfcs_error = TimeseriesAggregator(cfcs[index]).get_aggregate()
    cfcs_series.append(cfcs_serie)
    cfcs_errors.append(cfcs_error)

figure_cfcs = plt.figure(figsize=(8,6), dpi=10, facecolor="w", edgecolor="k")
plt.xlim([0, 50])
plt.xlabel("Generation")
plt.ylabel("Mittlere Restriktionsaufrufe pro Generation")

for method in method_names:
    index = method_names.index(method)
    generations = range(0, len(cfcs_series[index]))
    errorbar(generations,\
        cfcs_series[index],\
        yerr=cfcs_errors[index],\
        linestyle="none",\
        marker=method_marker[index],\
        color=method_color[index],\
        ecolor='#CCCCCC')

pp = PdfPages("cfcs.pdf")
plt.savefig(pp, format='pdf')
pp.close()

# cum cfc time series
for method in method_names:
    index = method_names.index(method)
    for i in range(0, len(cfcs[index])):
        cfcs[index][i] = array(cfcs[index][i]).cumsum().tolist()
    cfcs_serie, cfcs_error = TimeseriesAggregator(cfcs[index]).get_aggregate()
    cum_cfcs_series.append(cfcs_serie)
    cum_cfcs_errors.append(cfcs_error)

figure_cfcs = plt.figure(figsize=(8,6), dpi=10, facecolor="w", edgecolor="k")
plt.xlabel("Generation")
plt.ylabel("Kumulierte Restriktionsaufrufe nach Generation")
plt.xlim([0, 50])

for method in method_names:
    index = method_names.index(method)
    generations = range(0, len(cum_cfcs_series[index]))
    errorbar(generations,\
        cum_cfcs_series[index],\
        yerr=cum_cfcs_errors[index],\
        linestyle="none",\
        marker=method_marker[index],\
        color=method_color[index],\
        ecolor='#CCCCCC')

pp = PdfPages("cum_cfcs.pdf")
plt.savefig(pp, format='pdf')
pp.close()

# wilcoxon test for cfcs per generation
z, p = wilcoxon(cfcs_series[0], cfcs_series[1])
print "cfcs per generations: z: %f, p: %f" % (z,p)

# wilcoxon test for cumulated cfcs
z, p = wilcoxon(cfcs_sums[0], cfcs_sums[1])
print "cumulated cfcs wilcoxon test: z: %f, p: %f" % (z,p)

