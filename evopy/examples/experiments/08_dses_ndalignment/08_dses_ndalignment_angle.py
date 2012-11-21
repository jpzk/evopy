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

import pdb 
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
        initial_sigma = matrix([[4.5, 4.5, 4.5]]),
        delta = 4.5,
        tau0 = 0.5, 
        tau1 = 0.6,
        initial_pos = matrix([[10.0, 10.0, 10.0]]),
        beta = 1.0,
        meta_model = meta_model) 

    return method

samples = 50 
termination = Generations(50)
problem = TRProblem(dimensions=3)

angles_list = []
first_angles_list = []
second_angles_list = []

simulators = []
for i in range(0, samples):
    simulator = Simulator(get_method_TR_ali(), problem, termination)
    simulators.append(simulator)

for simulator in simulators:
    simulator.simulate()
    angles_list.append(simulator.optimizer.logger.all()['angles'])

for angles in angles_list:
    first_angles_list.append(map(lambda angles : angles[0], angles))
    second_angles_list.append(map(lambda angles : angles[1], angles))

st_angles_serie, st_angles_error =\
    TimeseriesAggregator(first_angles_list).get_aggregate()

nd_angles_serie, nd_angles_error =\
    TimeseriesAggregator(second_angles_list).get_aggregate()

figure_accs = plt.figure(figsize=(8,6), dpi=10, facecolor="w", edgecolor="k")
plt.xlabel("Generation")
plt.ylabel("Winkel in einer Generation")
plt.xlim([0, 50])
plt.ylim([0.0, 200.0])

generations = range(0, len(st_angles_serie))

b0 = errorbar(generations,\
    st_angles_serie,\
    marker=".",
    color="#044997",\
    ecolor="#CCCCCC",\
    linestyle="none",
    label="mit Normalisierung",\
    yerr=st_angles_error)

b0 = errorbar(generations,\
    nd_angles_serie,\
    marker="x",
    color="g",\
    ecolor="#CCCCCC",\
    linestyle="none",
    label="mit Normalisierung",\
    yerr=nd_angles_error)

pp = PdfPages("angle.pdf")
plt.savefig(pp, format='pdf')
pp.close()

