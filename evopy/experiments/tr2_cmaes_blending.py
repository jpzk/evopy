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

---

Problem: TR2 (dimension = 2) 
Fitness-accuracy termination: 10^-12
Strategies: CMAES, CMAES-SVC-Blending
'''

from sys import path
path.append("../..")

from sklearn.cross_validation import KFold
from evopy.experiments.experiment import Experiment
from evopy.problems.simple_sa_sphere_problem import SimpleSASphereProblem
from evopy.problems.sa_sphere_problem import SASphereProblem
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.operators.mutation.gauss_sigma_aligned_nd import GaussSigmaAlignedND
from evopy.operators.combination.sa_intermediate import SAIntermediate
from evopy.operators.selection.smallest_fitness import SmallestFitness
from evopy.operators.selfadaption.selfadaption import Selfadaption
from evopy.views.universal_view import UniversalView
from evopy.views.cv_ds_linear_view import CVDSLinearView
from evopy.views.cv_ds_r_linear_view import CVDSRLinearView
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear
from evopy.strategies.cmaes import CMAES
from evopy.strategies.cmaes_svc_repair import CMAESSVCR

def _run_cmaes():
    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(50, 5))

    method = CMAES(\
        SASphereProblem(dimensions = 2, accuracy = -12),
        mu = 15,
        lambd = 100,
        combination = SAIntermediate(),\
        mutation = GaussSigmaAlignedND(),\
        selection = SmallestFitness(),
        xmean = [5.0, 5.0],
        sigma = 1.0,
        view = UniversalView())
     
    method.run()
    return method

def _run_cmaes_svc():
    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(50, 5))

    method = CMAESSVCR(\
        SASphereProblem(dimensions = 2, accuracy = -12),
        mu = 15,
        lambd = 100,
        combination = SAIntermediate(),\
        mutation = GaussSigmaAlignedND(),\
        selection = SmallestFitness(),
        xmean = [5.0, 5.0],
        sigma = 1.0,
        view = UniversalView(),
        beta = 0.90,
        window_size = 25,
        append_to_window = 25,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv, 
        repair_mode = 'mirror')
     
    method.run()     
    return method

class TR2CMAESBlendingExperiment(Experiment):

    def __init__(self):
        super(TR2CMAESBlendingExperiment, self).__init__(\
            "TR2", "tr2-cmaes-blending")

    def run(self):
        ''' no documentation yet '''

        print __doc__

        cases = [_run_cmaes, _run_cmaes_svc]
        self.run_cases(cases, 2)


if __name__ == "__main__":
    experiment = TR2CMAESBlendingExperiment()
    experiment.run()
