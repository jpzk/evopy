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

matplotlib.use('GTK')
from matplotlib.backends.backend_gtk import FigureCanvasGTK, NavigationToolbar
import pygtk
pygtk.require('2.0')

import gtk
import gtk.glade
import gobject
import time

from sys import path
path.append("../..")

import evopy.examples.CMAESSVCR as CMAESSVCR
from evopy.problems.tr_problem import TRProblem

from evopy.gtk.fitness_plot import FitnessPlot
from evopy.gtk.searchspace_plot import SearchspacePlot
from evopy.gtk.simulator import Simulator

class appGui():
      
    def on_update_plots(self, stats):
        for plot in self.plots:
            plot.on_update(stats)

    def on_optimizer_pulldown_show(self, a):
        return 

    def on_reset_button_clicked(self, widget):
        self.simulator.stop = True
        
    def on_play_button_clicked(self, widget):
        
        optimizer = CMAESSVCR.get_method()
        problem = TRProblem()
        accuracy = pow(10, -12)

        self.simulator = Simulator()
        self.simulator.configure(optimizer, problem, accuracy)            

        self.simulator.gui = self
        self.simulator.start()

    def on_destroy(self, a):
        self.simulator.gui_closed = True
        gtk.main_quit()

    def __init__(self):
        gladefile = "gui.xml"
        builder = gtk.Builder()
        builder.add_from_file(gladefile)
        self.window = builder.get_object("window1")
        builder.connect_signals(self)

        self.plots = []

        self.window.connect("destroy", self.on_destroy)
        self.optimizer_pulldown = builder.get_object("optimizer_pulldown")
        self.optimizer_pulldown.set_title("optimizer")

        liststore = gtk.ListStore(str)
        self.optimizer_pulldown.set_model(liststore)
        cell = gtk.CellRendererText()

        self.optimizer_pulldown.pack_start(cell, True)
        self.optimizer_pulldown.add_attribute(cell, 'text', 0) 
        self.optimizer_pulldown.append_text("CMA-ES-SVC")
        self.optimizer_pulldown.set_active(0)

        self.fitness_plot = FitnessPlot() 
        self.plots.append(self.fitness_plot)
        self.fitness_plot.show()
        
        self.searchspace_plot = SearchspacePlot()
        self.plots.append(self.searchspace_plot)
        self.searchspace_plot.show()

        self.vbox = builder.get_object("vbox1")
        self.vbox.pack_start(self.fitness_plot, True, True)
        self.vbox.pack_start(self.searchspace_plot, True, True)

    def main(self):
        self.window.show()

        gtk.main()

if __name__ == "__main__":
    gtk.gdk.threads_init()
    app = appGui()   

    gtk.gdk.threads_enter()
    app.main() 
    gtk.gdk.threads_leave()
