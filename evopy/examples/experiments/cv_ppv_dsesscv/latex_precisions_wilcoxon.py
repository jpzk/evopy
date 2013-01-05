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
from scipy.stats import wilcoxon, ranksums 
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
import pdb

accuraciefile = file("output/precision_file.save", "r")
precisions = load(accuraciefile)

none_filter = lambda accuracy : type(accuracy) != type(None)

SR1_X_nonessc = filter(none_filter, list(chain.from_iterable(precisions[SphereProblemOriginR1][get_method_SphereProblemR1_none])))
SR1_Y_nonessc = filter(none_filter, list(chain.from_iterable(precisions[SphereProblemOriginR1][get_method_SphereProblemR1_ssc])))
SR1_nonessc_p = ranksums(SR1_X_nonessc, SR1_Y_nonessc)

SR2_X_nonessc = filter(none_filter, list(chain.from_iterable(precisions[SphereProblemOriginR2][get_method_SphereProblemR2_none])))
SR2_Y_nonessc = filter(none_filter, list(chain.from_iterable(precisions[SphereProblemOriginR2][get_method_SphereProblemR2_ssc])))
SR2_nonessc_p = ranksums(SR2_X_nonessc, SR2_Y_nonessc)

TR_X_nonessc = filter(none_filter, list(chain.from_iterable(precisions[TRProblem][get_method_TR_none])))
TR_Y_nonessc = filter(none_filter, list(chain.from_iterable(precisions[TRProblem][get_method_TR_ssc])))
TR_nonessc_p = ranksums(TR_X_nonessc, TR_Y_nonessc)

Sch_X_nonessc = filter(none_filter, list(chain.from_iterable(precisions[SchwefelsProblem26][get_method_Schwefel26_none])))
Sch_Y_nonessc = filter(none_filter, list(chain.from_iterable(precisions[SchwefelsProblem26][get_method_Schwefel26_ssc])))
Sch_nonessc_p = ranksums(Sch_X_nonessc, Sch_Y_nonessc)

#####

SR1_X_norssc = filter(none_filter, list(chain.from_iterable(precisions[SphereProblemOriginR1][get_method_SphereProblemR1_nor])))
SR1_Y_norssc = filter(none_filter, list(chain.from_iterable(precisions[SphereProblemOriginR1][get_method_SphereProblemR1_ssc])))
SR1_norssc_p = ranksums(SR1_X_norssc, SR1_Y_norssc)

SR2_X_norssc = filter(none_filter, list(chain.from_iterable(precisions[SphereProblemOriginR2][get_method_SphereProblemR2_nor])))
SR2_Y_norssc = filter(none_filter, list(chain.from_iterable(precisions[SphereProblemOriginR2][get_method_SphereProblemR2_ssc])))
SR2_norssc_p = ranksums(SR2_X_norssc, SR2_Y_norssc)

TR_X_norssc = filter(none_filter, list(chain.from_iterable(precisions[TRProblem][get_method_TR_nor])))
TR_Y_norssc = filter(none_filter, list(chain.from_iterable(precisions[TRProblem][get_method_TR_ssc])))
TR_norssc_p = ranksums(TR_X_norssc, TR_Y_norssc)

Sch_X_norssc = filter(none_filter, list(chain.from_iterable(precisions[SchwefelsProblem26][get_method_Schwefel26_nor])))
Sch_Y_norssc = filter(none_filter, list(chain.from_iterable(precisions[SchwefelsProblem26][get_method_Schwefel26_ssc])))
Sch_norssc_p = ranksums(Sch_X_norssc, Sch_Y_norssc)

results = file("output/results_precisions_wilcoxon.tex", "w")
lines = [
    "\\begin{tabularx}{\\textwidth}{l l X X X X}\n", 
    "\\toprule\n", 
    "\\textbf{Skalierung X} & \\textbf{Skalierung Y} & Problem & p-Wert \\\\\n",
    "\midrule\n",
    "Ohne Skalierung & Standardisierung & Kugel. R. 1 & %1.2e \\\\\n"\
    % SR1_nonessc_p[1],\
    "&& Kugel. R. 2 & %1.2e \\\\\n"\
    % SR2_nonessc_p[1],\
    "&& TR2 Problem & %1.2e \\\\\n"\
    % TR_nonessc_p[1],\
    "&& Schwefels Problem 2.6 & %1.2e \\\\\n"\
    % Sch_nonessc_p[1],\
    "Normalisierung & Standardisierung & Kugel R. 1& %1.2e \\\\\n"\
    % SR1_norssc_p[1],\
    "&& Kugel. R. 2 & %1.2e \\\\\n"\
    % SR2_norssc_p[1],\
    "&& TR2 Problem & %1.2e \\\\\n"\
    % TR_norssc_p[1],\
    "&& Schwefels Problem 2.6 & %1.2e \\\\\n"\
    % Sch_norssc_p[1],\
    "\\bottomrule\n",\
    "\end{tabularx}\n"]

results.writelines(lines)
results.close()

