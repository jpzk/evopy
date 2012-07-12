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

import sys
import matplotlib
import threading

matplotlib.use('GTK')
from matplotlib.backends.backend_gtk import FigureCanvasGTK, NavigationToolbar
import pygtk
pygtk.require('2.0')

import gtk
import gtk.glade
import time

from sys import path
path.append("../..")

from sklearn.cross_validation import KFold
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
from evopy.strategies.cmaes_svc_repair import CMAESSVCR
from evopy.metamodel.cma_svc_linear_meta_model import CMASVCLinearMetaModel
from evopy.problems.tr_problem import TRProblem

class appGui():
   
    class Evopy(threading.Thread):

        def run(self):                  
            sklearn_cv = SVCCVSkGridLinear(\
            C_range = [2 ** i for i in range(-5, 15, 2)],
            cv_method = KFold(50, 5))

            meta_model = CMASVCLinearMetaModel(\
                window_size = 25,
                scaling = ScalingStandardscore(),
                crossvalidation = sklearn_cv,
                repair_mode = 'mirror')

            self.problem = TRProblem()
            self.accuracy = pow(10, -12)

            self.optimizer = CMAESSVCR(\
                mu = 50,
                lambd = 100,
                xmean = [5.0, 5.0],
                sigma = 1.0,
                beta = 0.9,
                meta_model = meta_model) 

            self.gui_closed = False
            while(not self.gui_closed):
                # Simulator and optimizer handling constraints
                all_feasible = False
                while(not all_feasible):
                    # ASK for solutions (feasbile and infeasible) 
                    solutions = self.optimizer.ask_pending_solutions()

                    # CHECK solutions for feasibility 
                    feasibility = lambda solution : (solution, self.problem.is_feasible(solution))
                    feasibility_information = map(feasibility, solutions)

                    # TELL feasibility, returns True if all feasible, returns False if extra checks
                    all_feasible = self.optimizer.tell_feasibility(feasibility_information)

                # ASK for valid solutions (feasible)
                valid_solutions = self.optimizer.ask_valid_solutions()

                # CHECK fitness
                fitness = lambda solution : (solution, self.problem.fitness(solution))
                fitnesses = map(fitness, valid_solutions)

                # TELL fitness, return optimum
                optimum, optimum_fitness = self.optimizer.tell_fitness(fitnesses)

                # GUI update
                stats = self.optimizer.get_last_statistics()
                self.gui.update_fitness(\
                    stats['best_fitness'], stats['avg_fitness'], stats['worst_fitness'])
 
                self.gui.plot_datapoints(stats['selected_children'])
                                           
                time.sleep(0.5)

                # TERMINATION
                if(optimum_fitness <= self.problem.optimum_fitness() + self.accuracy):
                    break

    
    def plot_datapoints(self, values):

        self.hist_axis.cla()
        self.hist_axis.grid(True) 

        xv = lambda value : value[0]
        yv = lambda value : value[1]
        X = map(xv, values)
        Y = map(yv, values)

        self.hist_axis.scatter(X,Y, color='green')
        self.hist_canvas.draw_idle()


    def destroy(self, a):
        self.evopy.gui_closed = True
        gtk.main_quit()

    def update_fitness(self, best, average, worst):
        self.evopy.gui_locking = True

        self.best_fitness_trajectory.append(best)
        self.average_fitness_trajectory.append(average)
        self.worst_fitness_trajectory.append(worst)
        
        generations = range(0, len(self.best_fitness_trajectory))
        self.data_axis.cla()
        self.data_axis.plot(generations[-10:], self.best_fitness_trajectory[-10:], color='green')
        #self.data_axis.plot(generations[-10:], self.worst_fitness_trajectory[-10:], color='blue')
        #self.data_axis.plot(generations[-10:], self.average_fitness_trajectory[-10:], color='black')
        self.data_canvas.draw_idle()

    def __init__(self):
        gladefile = "cmaes.xml"
        builder = gtk.Builder()
        builder.add_from_file(gladefile)
        self.window = builder.get_object("window1")
        builder.connect_signals(self)
        self.window.connect("destroy", self.destroy)

        self.data_figure = matplotlib.figure.Figure()
        self.data_figure.suptitle('fitness', fontsize=12)
        self.data_axis = self.data_figure.add_subplot(111)
        self.data_axis.grid(True)
        self.data_canvas = FigureCanvasGTK(self.data_figure)
        self.data_canvas.show()

        self.hist_figure = matplotlib.figure.Figure()
        self.hist_figure.suptitle('selected children', fontsize=12)
        self.hist_axis = self.hist_figure.add_subplot(111)
        self.hist_axis.grid(True)
        self.hist_canvas = FigureCanvasGTK(self.hist_figure)
        self.hist_canvas.show()

        self.vbox = builder.get_object("vbox1")
        self.vbox.pack_start(self.data_canvas, True, True)
        self.vbox.pack_start(self.hist_canvas, True, True)

        self.best_fitness_trajectory = []
        self.average_fitness_trajectory = []
        self.worst_fitness_trajectory = []

    def main(self):
        self.window.show()
        self.evopy = self.Evopy()
        self.evopy.gui = self
        self.evopy.start()
        gtk.main()

if __name__ == "__main__":
    gtk.gdk.threads_init()
    app = appGui()   

    gtk.gdk.threads_enter()
    app.main() 
    gtk.gdk.threads_leave()
