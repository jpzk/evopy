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

from numpy import matrix, polyfit, poly1d, linspace
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pylab import * 

from evopy.strategies.ori_dses_svc import ORIDSESSVC
from evopy.problems.tr_problem import TRProblem
from evopy.simulators.simulator import Simulator

from evopy.metamodel.dses_svc_linear_meta_model import DSESSVCLinearMetaModel
from sklearn.cross_validation import KFold
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.operators.scaling.scaling_dummy import ScalingDummy
from evopy.operators.scaling.scaling_normalization import ScalingNormalization
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear
from evopy.operators.termination.or_combinator import ORCombinator
from evopy.operators.termination.accuracy import Accuracy
from evopy.operators.termination.generations import Generations
from evopy.operators.termination.convergence import Convergence 
from evopy.helper.timeseries_aggregator import TimeseriesAggregator

from multiprocessing import cpu_count
from evopy.external.playdoh import map as pmap

def get_method_without_scaling():

    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 5, 2)],
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
        beta = 0.9,
        meta_model = meta_model) 

    return method

def get_method_with_normalization():
    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 5, 2)],
        cv_method = KFold(20, 5))

    meta_model = DSESSVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingNormalization(),
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

def get_method_with_scaling():

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
simulators_with_n = []

samples = 25 

for i in range(0, samples):
    optimizer = get_method_with_normalization()
    problem = TRProblem()
    simulators_with_n.append(Simulator(optimizer, problem, Generations(50)))

for i in range(0, samples):
    optimizer = get_method_with_scaling()
    problem = TRProblem()
    simulators_with_s.append(Simulator(optimizer, problem, Generations(50)))

for i in range(0, samples):
    optimizer = get_method_without_scaling()
    problem = TRProblem()
    simulators_without_s.append(Simulator(optimizer, problem, Generations(50)))

map(process, simulators_with_s)
map(process, simulators_with_n)
map(process, simulators_without_s)

accuracies_with_n = []
accuracies_with_s = []
accuracies_without_s = []

for simulator in simulators_with_n:
    accuracies_with_n.append(\
        simulator.optimizer.meta_model.logger.all()['best_acc'])

for simulator in simulators_with_s:
    accuracies_with_s.append(\
        simulator.optimizer.meta_model.logger.all()['best_acc'])

for simulator in simulators_without_s:
    accuracies_without_s.append(\
        simulator.optimizer.meta_model.logger.all()['best_acc'])

accuracies_with_n, errors_with_n =\
    TimeseriesAggregator(accuracies_with_n).get_aggregate()

generations_with_n = range(0, len(accuracies_with_n))

accuracies_with_s, errors_with_s =\
    TimeseriesAggregator(accuracies_with_s).get_aggregate()

generations_with_s = range(0, len(accuracies_with_s))

accuracies_without_s, errors_without_s =\
    TimeseriesAggregator(accuracies_without_s).get_aggregate()

generations_without_s = range(0, len(accuracies_without_s))

figure_accs = plt.figure(figsize=(8,6), dpi=10, facecolor="w", edgecolor="k")
plt.xlabel("Generation")
plt.ylabel("CV Genauigkeit")
plt.ylim([0.0, 1.0])
plt.xlim([0, 50])
plt.ylim([0.0, 1.0])

b0 = errorbar(generations_with_n,\
    accuracies_with_n,\
    marker="+",
    color="k",\
    ecolor="#CCCCCC",\
    linestyle="none",
    label="mit Normalisierung",\
    yerr=errors_with_n)

b0z = polyfit(generations_with_n[20:], accuracies_with_n[20:], 0)
b0p = poly1d(b0z)
b0ls = linspace(generations_with_n[0], generations_with_n[-1], 100)
b0p = plot(b0ls, b0p(b0ls), color="k", linestyle="-")

b1 = errorbar(generations_with_s,\
    accuracies_with_s,\
    marker="x",
    color="g",\
    ecolor="#CCCCCC",\
    linestyle="none",
    label="mit Skalierung",\
    yerr=errors_with_s)

b1z = polyfit(generations_with_s[20:], accuracies_with_s[20:], 0)
b1p = poly1d(b1z)
b1ls = linspace(generations_with_s[0], generations_with_s[-1], 100)
b1p = plot(b1ls, b1p(b1ls), color="g", linestyle="-")

b2 = errorbar(generations_without_s,\
    accuracies_without_s,\
    marker=".",\
    color="#004997",\
    ecolor="#CCCCCC",\
    linestyle="none",\
    label="ohne Skalierung",\
    yerr=errors_without_s)

b2z = polyfit(generations_without_s[20:], accuracies_without_s[20:], 0)
b2p = poly1d(b2z)
b2ls = linspace(generations_without_s[0], generations_without_s[-1], 100)
b2p = plot(b2ls, b2p(b2ls), color="#004997", linestyle="-")

#legend([b1, b2], ["mit Skalierung", "ohne Skalierung"])

pp = PdfPages("acc.pdf")
plt.savefig(pp, format='pdf')
pp.close()
