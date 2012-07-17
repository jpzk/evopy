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

import evopy.examples.CMAES as CMAES
import evopy.examples.CMAESSVCRDR as CMAESSVCRDR

from evopy.problems.tr_problem import TRProblem

from evopy.gtk.components.cfc_plot import CFCPlot
from evopy.gtk.components.cumulated_cfc_plot import CumulatedCFCPlot
from evopy.gtk.components.parameter_C_plot import ParameterCPlot
from evopy.gtk.components.accuracy_plot import AccuracyPlot
from evopy.gtk.components.metamodel_combobox import MetamodelComboBox
from evopy.gtk.components.problem_combobox import ProblemComboBox
from evopy.gtk.components.optimizer_combobox import OptimizerComboBox
from evopy.gtk.components.mutation_plot import MutationPlot
from evopy.gtk.components.infeasibles_plot import InfeasiblesPlot
from evopy.gtk.components.cumulated_infeasibles_plot import CumulatedInfeasiblesPlot
from evopy.gtk.components.fitness_plot import FitnessPlot
from evopy.gtk.components.searchspace_plot import SearchspacePlot
from evopy.gtk.simulator import Simulator

class appGui():
     
    def __init__(self):

        self.pages = ['configuration', 'search_space', 'fitness', 'mutation',\
            'constraints', 'constraint_calls', 'metamodel']

        self.notebook_pages, self.notebook_plots = {}, {}
        self.optimizer_plots, self.metamodel_plots, self.simulator_plots =\
            [],[],[]

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

        self.accuracy_plot = AccuracyPlot()
        self.metamodel_plots.append(self.accuracy_plot)
        self.notebook_plots['metamodel'].append(\
            self.accuracy_plot)
        self.accuracy_plot.show()
        self.vbox = builder.get_object("metamodel_vbox")
        self.vbox.pack_start(self.accuracy_plot, True, True)

        self.parameter_C_plot = ParameterCPlot()
        self.metamodel_plots.append(self.parameter_C_plot)
        self.notebook_plots['metamodel'].append(\
            self.parameter_C_plot)
        self.parameter_C_plot.show()
        self.vbox = builder.get_object("metamodel_vbox")
        self.vbox.pack_start(self.parameter_C_plot, True, True)

        self.cumulated_infeasibles_plot = CumulatedInfeasiblesPlot()
        self.optimizer_plots.append(self.cumulated_infeasibles_plot)
        self.notebook_plots['constraints'].append(\
            self.cumulated_infeasibles_plot)
        self.cumulated_infeasibles_plot.show()
        self.vbox = builder.get_object("constraint_vbox")
        self.vbox.pack_start(self.cumulated_infeasibles_plot, True, True)

        self.infeasibles_plot = InfeasiblesPlot() 
        self.optimizer_plots.append(self.infeasibles_plot)
        self.notebook_plots['constraints'].append(self.infeasibles_plot)
        self.infeasibles_plot.show()
        self.vbox = builder.get_object("constraint_vbox")
        self.vbox.pack_start(self.infeasibles_plot, True, True)

        self.fitness_plot = FitnessPlot() 
        self.optimizer_plots.append(self.fitness_plot)
        self.notebook_plots['fitness'].append(self.fitness_plot)
        self.fitness_plot.show()
        self.vbox = builder.get_object("fitness_vbox")
        self.vbox.pack_start(self.fitness_plot, True, True)
       
        self.searchspace_plot = SearchspacePlot()
        self.notebook_plots['search_space'].append(self.searchspace_plot)
        self.optimizer_plots.append(self.searchspace_plot)
        self.searchspace_plot.show()
        self.searchspace_vbox = builder.get_object("searchspace_vbox")
        self.searchspace_vbox.pack_start(self.searchspace_plot, True, True)

        self.mutation_plot = MutationPlot()
        self.optimizer_plots.append(self.mutation_plot)
        self.mutation_plot.show()
        self.notebook_plots['mutation'].append(self.mutation_plot)
        self.mutation_vbox = builder.get_object("mutation_vbox")
        self.mutation_vbox.pack_start(self.mutation_plot, True, True)

        self.cumulated_cfc_plot = CumulatedCFCPlot()
        self.simulator_plots.append(self.cumulated_cfc_plot)
        self.cumulated_cfc_plot.show()
        self.notebook_plots['constraint_calls'].append(self.cumulated_cfc_plot)
        self.constraint_calls_vbox = builder.get_object("constraint_calls_vbox")
        self.constraint_calls_vbox.pack_start(self.cumulated_cfc_plot, True, True)

        self.cfc_plot = CFCPlot()
        self.simulator_plots.append(self.cfc_plot)
        self.cfc_plot.show()
        self.notebook_plots['constraint_calls'].append(self.cfc_plot)
        self.constraint_calls_vbox = builder.get_object("constraint_calls_vbox")
        self.constraint_calls_vbox.pack_start(self.cfc_plot, True, True)

    def on_update_plots(self, optimizer_stats, simulator_stats,\
        metamodel_stats=False):

        current_page = self.notebook.get_current_page()
        draw_page = self.notebook_plots[self.pages[current_page]]

        for plot in self.optimizer_plots:
            plot.on_update(optimizer_stats)

        for plot in self.simulator_plots:
            plot.on_update(simulator_stats)

        if(metamodel_stats != False):
            for plot in self.metamodel_plots:
                plot.on_update(metamodel_stats)

        for plot in draw_page:
            plot.on_draw()

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
        for plot in self.optimizer_plots:
            plot.on_reset()
        for plot in self.metamodel_plots:
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
        
        optimizer = CMAES.get_method()
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
