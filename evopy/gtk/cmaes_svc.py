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

class appGui():
   
    class Evopy(threading.Thread):

        def run(self):
            sklearn_cv = SVCCVSkGridLinear(\
                C_range = [2 ** i for i in range(-5, 15, 2)],
                cv_method = KFold(50, 5))

            method = CMAESSVCR(\
                SASphereProblem(dimensions = 2, accuracy = -12),
                mu = 50,
                lambd = 100,
                combination = SAIntermediate(),\
                mutation = GaussSigmaAlignedND(),\
                selection = SmallestFitness(),
                xmean = [5.0, 5.0],
                sigma = 1.0,
                view = self.view,
                beta = 0.9,
                window_size = 25,
                append_to_window = 25,
                scaling = ScalingStandardscore(),
                crossvalidation = sklearn_cv, 
                repair_mode = 'mirror')             
           
            method.run()

    generations = 0
    best_fitness = []

    def plot_datapoints(self, freduced, funpacked, ireduced, iunpacked):

        self.data_axis.cla()
        self.data_axis.grid(True) 

        combinations =\
            [(freduced, "g", "s"), (funpacked, "g", "o"),\
            (ireduced, "r", "s"), (iunpacked, "r", "o")]

        for (data, color, marker) in combinations:
            x = map(lambda child : child[0], data)
            y = map(lambda child : child[1], data)
            self.data_axis.scatter(x,y, c=color, marker=marker)

        self.data_canvas.draw_idle()

        self.hist_axis.cla()
        self.data_axis.grid(True)

        reduce_to_x = lambda child : child[0]
        combinations = [(map(reduce_to_x,freduced), "g"), (map(reduce_to_x, ireduced), "r")]        

        for (data, color) in combinations:
            self.hist_axis.hist(data, color=color, histtype='step')

        self.hist_canvas.draw_idle()

    def view(\
        self, generations = None, best_fitness = None, best_acc = None,\
        parameter_C = None, epsilon = None, DSES_infeasibles = None,\
        wrong_meta_infeasibles = None, angles = None, sigmasmean = None):
        return

    def on_quit1_activate(self, a):
        gtk.main_quit()
        sys.exit()

    def __init__(self):
        gladefile = "cmaes.xml"
        builder = gtk.Builder()
        builder.add_from_file(gladefile)
        self.window = builder.get_object("window1")
        builder.connect_signals(self)

        self.data_figure = matplotlib.figure.Figure()
        self.data_axis = self.data_figure.add_subplot(111)
        self.data_axis.grid(True)
        self.data_canvas = FigureCanvasGTK(self.data_figure)
        self.data_canvas.show()

        self.hist_figure = matplotlib.figure.Figure()
        self.hist_axis = self.hist_figure.add_subplot(111)
        self.hist_axis.grid(True)
        self.hist_canvas = FigureCanvasGTK(self.hist_figure)
        self.hist_canvas.show()

        self.vbox = builder.get_object("vbox1")
        self.vbox.pack_start(self.data_canvas, True, True)
        self.vbox.pack_start(self.hist_canvas, True, True)

    def main(self):
        self.window.show()
        self.evopy = self.Evopy()
        self.evopy.view = self
        self.evopy.start()
        gtk.main()

if __name__ == "__main__":
    gtk.threads_init()
    app = appGui()   
    app.main() 
