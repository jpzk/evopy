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

durationsfile = file("output/durations.save", "r")
durations = load(durationsfile)

problems_order = [SphereProblemOriginR1, SphereProblemOriginR2,\
    TRProblem, SchwefelsProblem26]

problem_titles = {
    TRProblem : "TR2 Problem",\
    SphereProblemOriginR1: "Kugel. R. 1",\
    SphereProblemOriginR2: "Kugel. R. 2",\
    SchwefelsProblem26: "2.6 mit R."
    }

lines_p = []

titles = {
    True: "Mit Parallelisierung",
    False: "Ohne Parallelisierung"
}

all_durations = []
for use_par in parallel_options: 
    durations_str = "%s " % titles[use_par]
    for samples in sample_sizes:    
        durations_str += "& %1.2f" % ((durations[use_par][samples] / 1000.0)/60.0) + " "
    durations_str += "\\\\\n"                        
    all_durations.append(durations_str)

results = file("output/results.tex", "w")
lines = [
    "\\begin{tabularx}{\\textwidth}{l X X X X X X X X X X}\n", 
    "\\toprule\n", 
    "\\textbf{Simulator} & 10 & 20 & 30 & 40 & 50 & 60 & 70 & 80 & 90 & 100 \\\\\n",
    "\midrule\n"]

endlines = [
    "\\bottomrule\n",\
    "\end{tabularx}\n"]

lines = lines + all_durations + endlines
results.writelines(lines)
results.close()

