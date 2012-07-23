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

from sys import stdout
from csv import writer
from math import floor
from sklearn.cross_validation import KFold

from evopy.problems.sa_sphere_problem import SASphereProblem

from evopy.operators.mutation.gauss_sigma import GaussSigma
from evopy.operators.combination.sa_intermediate import SAIntermediate
from evopy.operators.selection.smallest_fitness import SmallestFitness
from evopy.operators.selfadaption.selfadaption import Selfadaption
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore

from evopy.metamodel.cv.svc_cv_sklearn_grid_rbf import SVCCVSkGridRBF
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear

from evopy.views.dses_view import DSESView
from evopy.views.cv_ds_rbf_view import CVDSRBFView
from evopy.views.cv_ds_linear_view import CVDSLinearView

from evopy.strategies.dses import DSES
from evopy.strategies.dses_svc_mirror import DSESSVCM
from evopy.strategies.dses_svc import DSESSVC

def _run_dsessvc():
    ''' no documentation yet '''

    sklearn_cv = SVCCVSkGridRBF(\
        gamma_range = [2 ** i for i in range(-15, 3, 2)],
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(50, 5))

    dsessvc = DSESSVC(\
        SASphereProblem(),
        mu = 15,
        lambd = 100,
        theta = 0.7,
        pi = 70, 
        epsilon = 1.0,
        tau0 = 1.0,
        tau1 = 0.1,
        combination = SAIntermediate(),\
        mutation = GaussSigma(),\
        selection = SmallestFitness(),
        view = CVDSRBFView(mute = True),
        beta = 0.9,
        window_size = 25,
        append_to_window = 25,
        crossvalidation = sklearn_cv,
        scaling = ScalingStandardscore(),
        selfadaption = Selfadaption())

    dsessvc.run()
    return dsessvc

def _run_dsessvcm():
    ''' no documentation yet '''

    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(50, 5))

    dsessvcm = DSESSVCM(\
        SASphereProblem(),
        mu = 15,
        lambd = 100,
        theta = 0.7,
        pi = 70, 
        epsilon = 1.0,
        tau0 = 1.0,
        tau1 = 0.1,
        combination = SAIntermediate(),\
        mutation = GaussSigma(),\
        selection = SmallestFitness(),
        view = CVDSLinearView(mute = True),
        beta = 0.9,
        window_size = 25,
        append_to_window = 25,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv, 
        selfadaption = Selfadaption())

    dsessvcm.run()
    return dsessvcm

def _run_dses():
    ''' no documentation yet '''

    dses = DSES(\
        problem = SASphereProblem(),
        mu = 15, 
        lambd = 100,
        pi = 50,
        theta = 0.7,
        epsilon = 1.0,
        tau0 = 1.0,
        tau1 = 0.1,
        combination = SAIntermediate(),
        mutation = GaussSigma(),
        selection = SmallestFitness(),
        view = DSESView(mute = True),
        selfadaption = Selfadaption())

    dses.run()
    return dses

class Experiment():

    def __init__(self):
        self._file_call = 'experiments/sa-tr2/experiment_calls.csv'
        self._file_fitnesses = 'experiments/sa-tr2/experiment_fitnesses.csv'
        self._file_acc = 'experiments/sa-tr2/experiment_acc.csv'

        self._writer_calls = writer(open(self._file_call, 'wb'), delimiter=';')
        self._writer_fitnesses = writer(open(self._file_fitnesses, 'wb'), \
            delimiter=';')

        self._writer_acc = writer(open(self._file_acc, 'wb'), delimiter=';')
        self._problem = "TR2"

        self._writer_calls.writerow(\
            ["problem",
            "method", 
            "sample",
            "train-function-calls",
            "constraint-calls",
            "metamodel-calls",
            "fitness-function-calls",
            "generations"])

        self._writer_fitnesses.writerow(\
            ["problem"
            "method", 
            "sample",
            "generation", 
            "worst-fitness", 
            "avg-fitness", 
            "best-fitness"])

        self._writer_acc.writerow(\
            ["problem",
            "method",
            "sample", 
            "generation", 
            "best-acc"])

    def done(self, i, n, msg):
        s = "["
        percentage = int(floor((float(i)/float(n)*10)))
        for i in range(0, percentage):
            s += "="
        for i in range(percentage, 10):
            s += " "
        s += "] " + msg
        return s

    def update_progress(self, i, n, msg):
        stdout.write('\r'*(12+len(msg)))
        stdout.flush()
        msg = self.done(i, n, msg)      
        stdout.write(msg)
        stdout.flush()

    def run(self):
        ''' no documentation yet '''

        n = 10

        for i in range(0, n):
            dses = _run_dses()
            self._write_stats(\
                dses._strategy_name,
                i,
            dses.get_statistics())
            self.update_progress(i+1, n, "dses")

        for i in range(0, n):
            dses = _run_dsessvc()
            self._write_stats(\
                dses._strategy_name,
                i,
                dses.get_statistics())
            self.update_progress(i+1, n, "dses-svc")

        for i in range(0, n):
            dses = _run_dsessvcm()
            self._write_stats(\
                dses._strategy_name,
                i,
                dses.get_statistics())
            self.update_progress(i+1, n, "dses-svc-m")

    def _write_stats(self, methodname, sample, stats):
        '''no documentation yet'''

        self._writer_calls.writerow(\
            [sample, methodname,
#            stats["train-function-calls"],
            stats["constraint-calls"],
#            stats["metamodel-calls"],
            stats["fitness-function-calls"],
            stats["generations"]])

        best_fitnesses = stats["best-fitness"]
        worst_fitnesses = stats["worst-fitness"]
        avg_fitnesses = stats["avg-fitness"]

        for generation in range(0, int(stats["generations"])):
            
            worst_fitness = worst_fitnesses[generation]
            avg_fitness = avg_fitnesses[generation]
            best_fitness = best_fitnesses[generation]

            self._writer_fitnesses.writerow([self._problem, methodname, 
                sample, methodname, \
                generation, worst_fitness, avg_fitness, best_fitness])

 #           self._writer_acc.writerow([self._problem, methodname, sample, \
 #               generation, stats["best-acc"][generation]])

