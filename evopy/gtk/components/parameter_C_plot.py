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

from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk import FigureCanvasGTK, NavigationToolbar

class ParameterCPlot(FigureCanvasGTK):
    def __init__(self):
        self.parameter_C_trajectory = []
        self.figure = Figure(dpi=75, facecolor='#e1e1e1')
        self.figure.suptitle('meta model parameter C', fontsize=12)
        self.axis = self.figure.add_subplot(111)
        self.axis.grid(True)

        super(ParameterCPlot, self).__init__(self.figure)            

    def on_reset(self):
        self.parameter_C_trajectory = []
        self.axis.cla()
        self.axis.grid(True)
        self.draw_idle()

    def on_draw(self):
        self.axis.cla()
        self.axis.grid(True)
        self.axis.plot(self.generations, 
            self.parameter_C_trajectory, color='green', marker="o")
        self.draw_idle()

    def on_update(self, stats):
        best_C = stats['best_parameter_C']
        self.parameter_C_trajectory.append(best_C)
        self.generations = range(0, len(self.parameter_C_trajectory))
