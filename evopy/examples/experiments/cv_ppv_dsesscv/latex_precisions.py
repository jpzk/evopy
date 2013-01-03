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

from setup import * 

precisionfile = file("output/precision_file.save", "r")
precisions = load(precisionfile)

variable_names = ['min', 'max', 'mean', 'var', 'h']
variables = {}
for variable in variable_names:
    variables[variable] = create_problem_optimizer_map(0.0)

for problem in problems:
    for optimizer in optimizers[problem]:
        precisionses = precisions[problem][optimizer]
        precisionses = list(chain.from_iterable(precisionses))

        variables['min'][problem][optimizer] = min(precisionses)
        variables['max'][problem][optimizer] = max(precisionses)
        variables['mean'][problem][optimizer] = array(precisionses).mean()
        variables['var'][problem][optimizer] = array(precisionses).var()

results = file("output/results_precisions.tex", "w")
lines = [
    "\\begin{tabularx}{\\textwidth}{l l X X X X}\n", 
    "\\toprule\n", 
    "\\textbf{Problem} & Skalierung & Minimum & Mittel & Maximum & Varianz \\\\\n",
    "\midrule\n",
    "Kugel R. 1 & Ohne Skalierung& %1.2f & %1.2f & %1.2f & %1.2e \\\\\n"\
    % (variables['min'][SphereProblemOriginR1][get_method_SphereProblemR1_none],\
    variables['mean'][SphereProblemOriginR1][get_method_SphereProblemR1_none],\
    variables['max'][SphereProblemOriginR1][get_method_SphereProblemR1_none],\
    variables['var'][SphereProblemOriginR1][get_method_SphereProblemR1_none]),\
    "& Standardisierung & %1.2f & %1.2f & %1.2f & %1.2e \\\\\n"\
    % (variables['min'][SphereProblemOriginR1][get_method_SphereProblemR1_ssc],\
    variables['mean'][SphereProblemOriginR1][get_method_SphereProblemR1_ssc],\
    variables['max'][SphereProblemOriginR1][get_method_SphereProblemR1_ssc],\
    variables['var'][SphereProblemOriginR1][get_method_SphereProblemR1_ssc]),\
    "& Normalisierung & %1.2f & %1.2f & %1.2f & %1.2e \\\\\n"\
    % (variables['min'][SphereProblemOriginR1][get_method_SphereProblemR1_nor],\
    variables['mean'][SphereProblemOriginR1][get_method_SphereProblemR1_nor],\
    variables['max'][SphereProblemOriginR1][get_method_SphereProblemR1_nor],\
    variables['var'][SphereProblemOriginR1][get_method_SphereProblemR1_nor]),\
    "Kugel R. 2 & Ohne Skalierung & %1.2f & %1.2f & %1.2f & %1.2e \\\\\n"\
    % (variables['min'][SphereProblemOriginR2][get_method_SphereProblemR2_none],\
    variables['mean'][SphereProblemOriginR2][get_method_SphereProblemR2_none],\
    variables['max'][SphereProblemOriginR2][get_method_SphereProblemR2_none],\
    variables['var'][SphereProblemOriginR2][get_method_SphereProblemR2_none]),\
    "& Standardisierung & %1.2f & %1.2f & %1.2f & %1.2e \\\\\n"\
    % (variables['min'][SphereProblemOriginR2][get_method_SphereProblemR2_ssc],\
    variables['mean'][SphereProblemOriginR2][get_method_SphereProblemR2_ssc],\
    variables['max'][SphereProblemOriginR2][get_method_SphereProblemR2_ssc],\
    variables['var'][SphereProblemOriginR2][get_method_SphereProblemR2_ssc]),\
    "& Normalisierung & %1.2f & %1.2f & %1.2f & %1.2e \\\\\n"\
    % (variables['min'][SphereProblemOriginR2][get_method_SphereProblemR2_nor],\
    variables['mean'][SphereProblemOriginR2][get_method_SphereProblemR2_nor],\
    variables['max'][SphereProblemOriginR2][get_method_SphereProblemR2_nor],\
    variables['var'][SphereProblemOriginR2][get_method_SphereProblemR2_nor]),\
    "TR2 & Ohne & %1.2f & %1.2f & %1.2f & %1.2e \\\\\n"\
    % (variables['min'][TRProblem][get_method_TR_none],\
    variables['mean'][TRProblem][get_method_TR_none],\
    variables['max'][TRProblem][get_method_TR_none],\
    variables['var'][TRProblem][get_method_TR_none]),\
    "& Standardisierung & %1.2f & %1.2f & %1.2f & %1.2e \\\\\n"\
    % (variables['min'][TRProblem][get_method_TR_ssc],\
    variables['mean'][TRProblem][get_method_TR_ssc],\
    variables['max'][TRProblem][get_method_TR_ssc],\
    variables['var'][TRProblem][get_method_TR_ssc]),\
    "& Normalisierung & %1.2f & %1.2f & %1.2f & %1.2e \\\\\n"\
    % (variables['min'][TRProblem][get_method_TR_nor],\
    variables['mean'][TRProblem][get_method_TR_nor],\
    variables['max'][TRProblem][get_method_TR_nor],\
    variables['var'][TRProblem][get_method_TR_nor]),\
    "2.60 mit R. & Ohne Skalierung & %1.2f & %1.2f & %1.2f & %1.2e\\\\\n"\
    % (variables['min'][SchwefelsProblem26][get_method_Schwefel26_none],\
    variables['mean'][SchwefelsProblem26][get_method_Schwefel26_none],\
    variables['max'][SchwefelsProblem26][get_method_Schwefel26_none],\
    variables['var'][SchwefelsProblem26][get_method_Schwefel26_none]),\
    "& Standardisierung & %1.2f & %1.2f & %1.2f & %1.2e \\\\\n"\
    % (variables['min'][SchwefelsProblem26][get_method_Schwefel26_ssc],\
    variables['mean'][SchwefelsProblem26][get_method_Schwefel26_ssc],\
    variables['max'][SchwefelsProblem26][get_method_Schwefel26_ssc],\
    variables['var'][SchwefelsProblem26][get_method_Schwefel26_ssc]),\
    "& Normalisierung & %1.2f & %1.2f & %1.2f & %1.2e \\\\\n"\
    % (variables['min'][SchwefelsProblem26][get_method_Schwefel26_nor],\
    variables['mean'][SchwefelsProblem26][get_method_Schwefel26_nor],\
    variables['max'][SchwefelsProblem26][get_method_Schwefel26_nor],\
    variables['var'][SchwefelsProblem26][get_method_Schwefel26_nor]),\
    "\\bottomrule\n",\
    "\end{tabularx}\n"]

results.writelines(lines)
results.close()

