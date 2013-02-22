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

from pickle import load 
from copy import deepcopy
from numpy import matrix, log10, array
from scipy.stats import wilcoxon 
from itertools import chain
from pylab import errorbar 
from matplotlib.backends.backend_pdf import PdfPages

from evopy.strategies.ori_dses_svc_repair import ORIDSESSVCR
from evopy.strategies.ori_dses_svc import ORIDSESSVC
from evopy.strategies.ori_dses import ORIDSES
from evopy.simulators.simulator import Simulator

from evopy.problems.sphere_problem_origin_r1 import SphereProblemOriginR1
from evopy.problems.sphere_problem_origin_r2 import SphereProblemOriginR2
from evopy.problems.schwefels_problem_26 import SchwefelsProblem26
from evopy.problems.tr_problem import TRProblem

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

import matplotlib.pyplot as plt
from setup import * 

fitnesses_file = file("output/fitnesses_file.save", "r")
fitnesses = load(fitnesses_file)

o_colors = {
    TRProblem: "#044977",\
    SphereProblemOriginR1: "k",\
    SphereProblemOriginR2: "g",\
    SchwefelsProblem26: "g"}

o_markers = {
    TRProblem: ".",\
    SphereProblemOriginR1: "x",\
    SphereProblemOriginR2: "+",\
    SchwefelsProblem26: "s"}

figure_accs = plt.figure(figsize=(8,6), dpi=10, facecolor="w", edgecolor="k")
plt.xlabel("Generation")
plt.ylabel("Fitnessgenauigkeit")
plt.xlim([0, 200])
plt.ylim([-100, 0])

for problem in problems:
    fitnesses_agg, errors_agg =\
        TimeseriesAggregator(fitnesses[problem]).get_aggregate()
        
    eb = errorbar(range(0, len(fitnesses_agg)),\
            fitnesses_agg,\
            marker=o_markers[problem],
            color=o_colors[problem],\
            ecolor="#CCCCCC",\
            linestyle="none",
            yerr=errors_agg)

pp = PdfPages("output/fitnesses.pdf")
plt.savefig(pp, format='pdf')
pp.close()
 
