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

Problem: TR4 (dimension = 4) 
Fitness-accuracy termination: 10^-4
Strategies: DSES, DSES-rbfSVC, DSES-linSVC-repair-aligned

'''

from sys import stdout
from math import floor
from os import makedirs

from sklearn.cross_validation import KFold

from evopy.problems.sa_sphere_problem import SASphereProblem
from evopy.operators.mutation.gauss_sigma import GaussSigma
from evopy.operators.mutation.gauss_sigma_aligned_nd import GaussSigmaAlignedND
from evopy.operators.combination.sa_intermediate import SAIntermediate
from evopy.operators.selection.smallest_fitness import SmallestFitness
from evopy.operators.selfadaption.selfadaption import Selfadaption
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear
from evopy.metamodel.cv.svc_cv_sklearn_grid_rbf import SVCCVSkGridRBF
from evopy.views.dses_view import DSESView
from evopy.views.universal_view import UniversalView
from evopy.views.cv_ds_linear_view import CVDSLinearView
from evopy.views.cv_ds_r_linear_view import CVDSRLinearView
from evopy.views.cv_ds_rbf_view import CVDSRBFView
from evopy.views.dses_view import DSESView
from evopy.strategies.dses import DSES
from evopy.strategies.dses_svc_repair import DSESSVCR
from evopy.strategies.dses_svc import DSESSVC

from experiment import Experiment

def _run_dses():
    ''' no documentation yet '''

    dses = DSES(\
        problem = SASphereProblem(dimensions = 4, accuracy = -4),
        mu = 15, 
        lambd = 100,
        pi = 50,
        theta = 0.7,
        epsilon = 1.0,
        combination = SAIntermediate(),
        mutation = GaussSigma(),
        selection = SmallestFitness(),
        view = DSESView(mute = False),
        selfadaption = Selfadaption(tau0 = 1.0, tau1 = 0.1))

    dses.run()
    return dses

def _run_dsessvc():
    ''' no documentation yet '''

    sklearn_cv = SVCCVSkGridRBF(\
        gamma_range = [2 ** i for i in range(-15, 3, 2)],
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(50, 5))

    dsessvc = DSESSVC(\
        SASphereProblem(dimensions = 4, accuracy = -4),
        mu = 15,
        lambd = 100,
        theta = 0.7,
        pi = 70,
        epsilon = 1.0,
        combination = SAIntermediate(),\
        mutation = GaussSigma(),\
        selection = SmallestFitness(),
        view = CVDSRBFView(mute = False),
        beta = 0.9,
        window_size = 25,
        append_to_window = 25,
        crossvalidation = sklearn_cv,
        scaling = ScalingStandardscore(),
        selfadaption = Selfadaption(tau0 = 1.0, tau1 = 0.1))

    dsessvc.run()
    return dsessvc

def _run_dsessvcr():
    ''' no documentation yet '''

    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(50, 5))

    dsessvcm = DSESSVCR(\
        SASphereProblem(dimensions = 4, accuracy = -4),
        mu = 15,
        lambd = 100,
        theta = 0.7,
        pi = 10,
        epsilon = 1.0,
        combination = SAIntermediate(),\
        mutation = GaussSigmaAlignedND(),\
        selection = SmallestFitness(),
        view = UniversalView(),
        beta = 0.9,
        window_size = 25,
        append_to_window = 25,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        selfadaption = Selfadaption(tau0 = 1.0, tau1 = 0.1),
        repair_mode = 'mirror')

    dsessvcm.run()
    return dsessvcm

class TR4RepairAlignedExperiment(Experiment):

    def __init__(self):
        super(TR4RepairAlignedExperiment, self).__init__(\
            "TR4", "tr4-repair-aligned")

    def run(self):
        ''' no documentation yet '''

        print __doc__
        samples = 1

        cases = [_run_dsessvcr, _run_dsessvc, _run_dses]
        self.run_cases(cases, 50)
