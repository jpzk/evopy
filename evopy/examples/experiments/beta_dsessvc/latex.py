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

cfcsfile = file("output/cfcs_file.save", "r")
cfcs = load(cfcsfile)

problems_order = [SphereProblemOriginR1, SphereProblemOriginR2,\
    TRProblem, SchwefelsProblem26]

problem_titles = {
    TRProblem : "TR2",\
    SphereProblemOriginR1: "Kugel. R. 1",\
    SphereProblemOriginR2: "Kugel. R. 2",\
    SchwefelsProblem26: "2.6 mit R."
    }

lines_p = []

for problem in problems_order:
    means = []
    stds = []
    for beta in betas:
        means.append(array(cfcs[problem][beta]).mean())
        stds.append(array(cfcs[problem][beta]).std())
    lines_p.append("%s & %1.2f & %1.2f & %1.2f & %1.2f & %1.2f & %1.2f & %1.2f & %1.2f & %1.2f & %1.2f\\\\\n"\
    % (problem_titles[problem], means[0], means[1], means[2], means[3], means[4],\
    means[5], means[6], means[7], means[8], means[9]))

results = file("output/results.tex", "w")
lines = [
    "\\begin{tabularx}{\\textwidth}{l X X X X X X X X X X}\n", 
    "\\toprule\n", 
    "\\textbf{Problem} & 0.1 & 0.2 & 0.3 & 0.4 & 0.5 & 0.6 & 0.7 & 0.8 & 0.9 & 1.0 \\\\\n",
    "\midrule\n"]

endlines = [
    "\\bottomrule\n",\
    "\end{tabularx}\n"]

lines = lines + lines_p + endlines

results.writelines(lines)
results.close()

