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
from sys import argv
path.append("../../../..")

from numpy import matrix, array
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pylab import * 

from evopy.strategies.ori_dses import ORIDSES 
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeaveOneOut

from evopy.strategies.ori_dses_svc_repair import ORIDSESSVCR
from evopy.strategies.ori_dses_svc import ORIDSESSVC
from evopy.problems.tr_problem import TRProblem
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

def get_method_with_SVC(beta):

    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 14, 2)],
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
        initial_pos = matrix([[10.0, 10.0]]),
        beta = beta,
        meta_model = meta_model) 

    return method

betas = map(lambda i : i / 100.0, range(0, 110))
simulators = []
cfcs = []
means = []

for beta in betas:
    simulators_for_beta = []
    index = betas.index(beta)
    for i in range(0, 25):
        simulators_for_beta = []
        optimizer = get_method_with_SVC(beta)
        problem = TRProblem()
        optimum_fitness = problem.optimum_fitness()
        accuracy = Accuracy(optimum_fitness, 10**(-6))
        convergence = Convergence(10**-12)
        generationst = Generations(50)
        termination = ORCombinator([accuracy, generationst])
        simulators_for_beta.append(\
            Simulator(optimizer, problem, termination))
    simulators.append(simulators_for_beta)

for beta in betas:
    index = betas.index(beta)
    cfc_for_method = []

    for simulator in simulators[index]:
        simulator.simulate()
        cfc_for_method.append(\
            sum(simulator.logger.all()['count_cfc']))
    cfcs.append(cfc_for_method) 

for beta in betas:
    index = betas.index(beta)
    means.append(array(cfcs[index]).mean())

figure_betas = plt.figure(figsize=(8,6), dpi=10, facecolor="w", edgecolor="k")

plt.xlabel("Einflussparameter $\\beta \cdot 10^2$")
plt.ylabel("Mittlere kum. Restriktionsaufrufe pro Generation")
plt.xlim([0, 100])

m = array(means).tolist()

plt.plot(\
    range(0, len(m)),    
    m, 
    color="#004997",
    marker=".",
    linestyle="none")

pp = PdfPages("cumbeta.pdf")
plt.savefig(pp, format='pdf')
pp.close()

