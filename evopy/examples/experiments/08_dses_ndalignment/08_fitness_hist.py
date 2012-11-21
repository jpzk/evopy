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
from numpy import matrix, log10, array
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
import pdb 

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

def get_method_TR_ali():
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

def create_problem_optimizer_map(typeofelements):
    t = typeofelements    
    return {\
    TRProblem: {get_method_TR: deepcopy(t), get_method_TR_ali: deepcopy(t)}}

samples = 100 
termination = Generations(50)
problems = [TRProblem]

optimizers = {\
    TRProblem: [get_method_TR, get_method_TR_ali]
}

simulators = {\
    TRProblem: {}
}

logit = lambda value, optimum : log10(value - optimum)
best_fitness = create_problem_optimizer_map([])

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
            best_fitness[problem][optimizer].append(fitness)

optimum_fitness = TRProblem().optimum_fitness()
best_tr2 = best_fitness[TRProblem][get_method_TR]
log_best_tr2 = map(logit, best_tr2, [optimum_fitness] * len(best_tr2))
best_tr2_svc = best_fitness[TRProblem][get_method_TR_ali]
log_best_tr2_svc = map(logit, best_tr2_svc, [optimum_fitness] * len(best_tr2))

figure_hist = plt.figure(figsize=(8,6), dpi=10, facecolor="w", edgecolor="k")
#ax = figure_hist.add_subplot(111)
#ax.xaxis.set_major_formatter((ticker.FormatStrFormatter('$2^{%d}$')))

def gauss(u):
    return (1.0 / sqrt(2 * pi)) * exp((-(1.0/2.0) * (u**2)))
 
def nadaraya(x, data, labels, h):
    labels = [0] + labels.tolist() + [0]   
    data = data.tolist()
    data = [data[0] - (data[1] - data[0])] + data 
    print data, labels
    bottom = sum(map(lambda sample : (1/h)*gauss((x - sample)/h), data))
    top = sum(map(lambda sample, label : label * (1/h)* gauss((x - sample)/h), data, labels))
    return float(top)/float(bottom)

figure_hist = plt.figure(figsize=(8,6), dpi=10, facecolor="w", edgecolor="k")

plt.xlabel('Genauigkeit')
plt.ylabel('Wahrscheinlichkeit')

ax = figure_hist.add_subplot(111)
ax.xaxis.set_major_formatter((ticker.FormatStrFormatter('$2^{%d}$')))

pdfs1, bins1, patches1 = hist(log_best_tr2, normed=True, alpha=0.5,\
    histtype='step', edgecolor="g")

h = 1.06 * array(log_best_tr2).std() * (len(log_best_tr2)**(-1.0/5.0))
x = linspace(-10, -1, 100)
y = map(lambda x : nadaraya(x, bins1, pdfs1, h), x)
plot(x,y, linestyle="--", color='g')

pdfs2, bins2, pachtes2 = hist(log_best_tr2_svc, normed=True, alpha=0.5,\
    histtype='step', edgecolor="#004779")

h = 1.06 * array(log_best_tr2_svc).std() * (len(log_best_tr2_svc)**(-1.0/5.0))
x = linspace(-10, -1, 100)
y = map(lambda x : nadaraya(x, bins2, pdfs2, h), x)
plot(x,y, color="#004779")

pp = PdfPages("dses_ndaligned_hist.pdf")
plt.savefig(pp, format='pdf')
pp.close()


