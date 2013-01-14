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

from multiprocessing import cpu_count
from evopy.external.playdoh import map as pmap

from setup import * 

bff = file("output/best_fitness_file.save", "r")
best_fitness = load(bff)

# statistics
variable_names = ['min', 'max', 'mean', 'var', 'h']
variables = {}
for variable in variable_names:
    variables[variable] = create_problem_optimizer_map(0.0)

for problem in problems:
    for optimizer in optimizers[problem]:
        import pdb
        pdb.set_trace()
        best_fitnesses = best_fitness[problem][optimizer]
        variables['min'][problem][optimizer] = min(best_fitnesses)
        variables['max'][problem][optimizer] = max(best_fitnesses)
        variables['mean'][problem][optimizer] = array(best_fitnesses).mean()
        variables['var'][problem][optimizer] = array(best_fitnesses).var()
        variables['h'][problem][optimizer] =\
            1.06 * array(best_fitnesses).std() * (len(best_fitnesses)**(-1.0/5.0))

pvalues = {}
for problem in problems:
    x = best_fitness[problem][optimizers[problem][0]]
    y = best_fitness[problem][optimizers[problem][1]]
    z, pvalues[problem] = ranksums(x,y)

results = file("output/results.tex","w")
lines = [
    "\\begin{tabularx}{\\textwidth}{l X X X X X X X }\n", 
    "\\toprule\n", 
    "\\textbf{Problem} & p-Wert & SVK & Minimum & Mittel & Maximum & Varianz & h\\\\\n",
    "\midrule\n",
    "Kugel R. 1 & %1.2f & nein & %1.2e & %1.2e & %1.2e & %1.2e & %1.2e \\\\\n"\
        % (pvalues[SphereProblemOriginR1],\
        variables['min'][SphereProblemOriginR1][get_method_SphereProblemR1],\
        variables['mean'][SphereProblemOriginR1][get_method_SphereProblemR1],\
        variables['max'][SphereProblemOriginR1][get_method_SphereProblemR1],\
        variables['var'][SphereProblemOriginR1][get_method_SphereProblemR1],\
        variables['h'][SphereProblemOriginR1][get_method_SphereProblemR1]),\
    "&& ja & %1.2e & %1.2e & %1.2e & %1.2e & %1.2e \\\\\n"\
        % (variables['min'][SphereProblemOriginR1][get_method_SphereProblemR1_svc],\
        variables['mean'][SphereProblemOriginR1][get_method_SphereProblemR1_svc],\
        variables['max'][SphereProblemOriginR1][get_method_SphereProblemR1_svc],\
        variables['var'][SphereProblemOriginR1][get_method_SphereProblemR1_svc],\
        variables['h'][SphereProblemOriginR1][get_method_SphereProblemR1_svc]),\
    "Kugel R. 2 & %1.2f & nein & %1.2e & %1.2e & %1.2e & %1.2e & %1.2e \\\\\n"\
        % (pvalues[SphereProblemOriginR2],\
        variables['min'][SphereProblemOriginR2][get_method_SphereProblemR2],\
        variables['mean'][SphereProblemOriginR2][get_method_SphereProblemR2],\
        variables['max'][SphereProblemOriginR2][get_method_SphereProblemR2],\
        variables['var'][SphereProblemOriginR2][get_method_SphereProblemR2],\
        variables['h'][SphereProblemOriginR2][get_method_SphereProblemR2_svc]),\
    "&& ja & %1.2e & %1.2e & %1.2e & %1.2e & %1.2e \\\\\n"\
        % (variables['min'][SphereProblemOriginR2][get_method_SphereProblemR2_svc],\
        variables['mean'][SphereProblemOriginR2][get_method_SphereProblemR2_svc],\
        variables['max'][SphereProblemOriginR2][get_method_SphereProblemR2_svc],\
        variables['var'][SphereProblemOriginR2][get_method_SphereProblemR2_svc],\
        variables['h'][SphereProblemOriginR2][get_method_SphereProblemR2_svc]),\
    "TR2 & %1.2f & nein & %1.2e & %1.2e & %1.2e & %1.2e & %1.2e \\\\\n"\
        % (pvalues[TRProblem],\
        variables['min'][TRProblem][get_method_TR]-2.0,\
        variables['mean'][TRProblem][get_method_TR]-2.0,\
        variables['max'][TRProblem][get_method_TR]-2.0,\
        variables['var'][TRProblem][get_method_TR],\
        variables['h'][TRProblem][get_method_TR]),\
    "&& ja & %1.2e & %1.2e & %1.2e & %1.2e & %1.2e \\\\\n"\
        % (variables['min'][TRProblem][get_method_TR_svc]-2.0,\
        variables['mean'][TRProblem][get_method_TR_svc]-2.0,\
        variables['max'][TRProblem][get_method_TR_svc]-2.0,\
        variables['var'][TRProblem][get_method_TR_svc],\
        variables['h'][TRProblem][get_method_TR_svc]),\
    "2.60 mit R. & %1.2f & nein & %1.2e & %1.2e & %1.2e & %1.2e & %1.2e\\\\\n"\
        % (pvalues[SchwefelsProblem26],\
        variables['min'][SchwefelsProblem26][get_method_Schwefel26]-99.0,\
        variables['mean'][SchwefelsProblem26][get_method_Schwefel26]-99.0,\
        variables['max'][SchwefelsProblem26][get_method_Schwefel26]-99.0,\
        variables['var'][SchwefelsProblem26][get_method_Schwefel26],\
        variables['h'][SchwefelsProblem26][get_method_Schwefel26]),\
    "&& ja & %1.2e & %1.2e & %1.2e & %1.2e & %1.2e \\\\\n"\
        % (variables['min'][SchwefelsProblem26][get_method_Schwefel26_svc]-99.0,\
        variables['mean'][SchwefelsProblem26][get_method_Schwefel26_svc]-99.0,\
        variables['max'][SchwefelsProblem26][get_method_Schwefel26_svc]-99.0,\
        variables['var'][SchwefelsProblem26][get_method_Schwefel26_svc],\
        variables['h'][SchwefelsProblem26][get_method_Schwefel26_svc]),\
    "\\bottomrule\n",\
    "\end{tabularx}\n"]

results.writelines(lines)
results.close()

