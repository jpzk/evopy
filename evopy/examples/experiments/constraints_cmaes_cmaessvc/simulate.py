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

from pickle import dump
from copy import deepcopy
from numpy import matrix, log10

from evopy.strategies.ori_dses_svc_repair import ORIDSESSVCR
from evopy.strategies.ori_dses_svc import ORIDSESSVC
from evopy.strategies.ori_dses import ORIDSES

from evopy.simulators.simulator import Simulator
from evopy.external.playdoh import map as pmap

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

from setup import *  

# create simulators
for problem in problems:
    for optimizer in optimizers[problem]:
        simulators_op = []
        for i in range(0, samples):
            opt = problem().optimum_fitness()
            termination = Accuracy(opt, pow(10, -15))
            simulator = Simulator(optimizer(), problem(), termination)
            simulators_op.append(simulator)
        simulators[problem][optimizer] = simulators_op

simulate = lambda simulator : simulator.simulate()

# run simulators 
for problem in problems:
    for optimizer, simulators_ in simulators[problem].iteritems():
        resulting_simulators = pmap(simulate, simulators_)
        for simulator in resulting_simulators:
            cfc = simulator.logger.all()['count_cfc']
            cfcs[problem][optimizer].append(cfc)
       
cfc_file = open("output/cfcs_file.save", "w")
dump(cfcs, cfc_file)
cfc_file.close()
