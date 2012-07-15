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

from evopy.gtk.metamodel_combobox import MetamodelComboBox
from evopy.gtk.problem_combobox import ProblemComboBox
from evopy.gtk.optimizer_combobox import OptimizerComboBox
from evopy.gtk.mutation_plot import MutationPlot
from evopy.gtk.fitness_plot import FitnessPlot
from evopy.gtk.searchspace_plot import SearchspacePlot
from evopy.gtk.simulator import Simulator

class appGui():
     
    def __init__(self):

        self.pages = ['configuration', 'search_space', 'fitness', 'mutation',\
            'constraints', 'metamodel']

        self.notebook_pages, self.notebook_plots = {}, {}
        for index, page in enumerate(self.pages):
            self.notebook_pages[page] = index
            self.notebook_plots[page] = []

        gladefile = "gui.xml"
        builder = gtk.Builder()
        builder.add_from_file(gladefile)

        self.window = builder.get_object("window1")
        self.pause_button = builder.get_object("pause_button")
        self.play_button = builder.get_object("play_button")
        self.reset_button = builder.get_object("reset_button")
        self.notebook = builder.get_object("tabs")

        optimizer_config_table = builder.get_object("optimizer_config_table")
        problem_config_table = builder.get_object("problem_config_table")

        builder.connect_signals(self)

        self.plots = []
        self.window.connect("destroy", self.on_destroy)
        
        self.optimizer_combobox = OptimizerComboBox()
        self.optimizer_combobox.show()

        self.metamodel_combobox = MetamodelComboBox()
        self.metamodel_combobox.show()

        optimizer_config_table.attach(self.optimizer_combobox, 1, 2, 0, 1)
        optimizer_config_table.attach(self.metamodel_combobox, 1, 2, 1, 2)
        
        self.problem_combobox = ProblemComboBox()
        self.problem_combobox.show()
 
        problem_config_table.attach(self.problem_combobox, 1, 2, 0, 1)

        self.fitness_plot = FitnessPlot() 
        self.plots.append(self.fitness_plot)
        self.notebook_plots['fitness'].append(self.fitness_plot)
        self.fitness_plot.show()
        self.vbox = builder.get_object("fitness_vbox")
        self.vbox.pack_start(self.fitness_plot, True, True)
       
        self.searchspace_plot = SearchspacePlot()
        self.notebook_plots['search_space'].append(self.searchspace_plot)
        self.plots.append(self.searchspace_plot)
        self.searchspace_plot.show()
        self.searchspace_vbox = builder.get_object("searchspace_vbox")
        self.searchspace_vbox.pack_start(self.searchspace_plot, True, True)

        self.mutation_plot = MutationPlot()
        self.plots.append(self.mutation_plot)
        self.mutation_plot.show()
        self.notebook_plots['mutation'].append(self.mutation_plot)
        self.mutation_vbox = builder.get_object("mutation_vbox")
        self.mutation_vbox.pack_start(self.mutation_plot, True, True)

    def on_update_plots(self, stats):        
        for plot in self.plots:
            plot.on_update(stats)

    def on_mutation_activate(self, widget):
        self.notebook.set_current_page(\
            self.notebook_pages['mutation'])
 
    def on_meta_model_activate(self, widget):
        self.notebook.set_current_page(\
            self.notebook_pages['metamodel'])
            
    def on_constraints_activate(self, widget):
        self.notebook.set_current_page(\
            self.notebook_pages['constraints'])
 
    def on_fitness_activate(self, widget):
        self.notebook.set_current_page(\
            self.notebook_pages['fitness'])    

    def on_search_space_activate(self, widget):
        self.notebook.set_current_page(\
            self.notebook_pages['search_space'])

    def on_configuration_activate(self, widget):
        self.notebook.set_current_page(\
            self.notebook_pages['configuration'])

    def on_optimizer_pulldown_show(self, a):
        return 

    def on_reset_button_clicked(self, widget):
        self.simulator.stop = True
        for plot in self.plots:
            plot.on_reset()

        self.play_button.set_sensitive(True)
        self.pause_button.set_sensitive(False)
        self.reset_button.set_sensitive(False)          

    def on_pause_button_clicked(self, widget):
        self.play_button.set_sensitive(True)
        self.pause_button.set_sensitive(False)
        self.reset_button.set_sensitive(True)          

        return

    def on_config_button_clicked(self, widget):
        self.notebook.set_current_page(\
            self.notebook_pages['configuration'])

    def on_play_button_clicked(self, widget):
        
        optimizer = CMAESSVCR.get_method()
        problem = TRProblem()
        accuracy = pow(10, -12)

        self.simulator = Simulator()
        self.simulator.configure(optimizer, problem, accuracy)            
        self.simulator.gui = self
        self.simulator.start()

        self.play_button.set_sensitive(False)
        self.pause_button.set_sensitive(True)
        self.reset_button.set_sensitive(True)

        if(self.notebook.get_current_page() ==\
            self.notebook_pages['configuration']):
            self.notebook.set_current_page(\
                self.notebook_pages['search_space'])

    def on_destroy(self, a):
        self.simulator.gui_closed = True
        gtk.main_quit()


    def main(self):
        self.window.show()

        gtk.main()

if __name__ == "__main__":
    gtk.gdk.threads_init()
    app = appGui()   

    gtk.gdk.threads_enter()
    app.main() 
    gtk.gdk.threads_leave()
