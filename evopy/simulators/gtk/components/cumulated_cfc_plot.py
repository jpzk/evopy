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

class CumulatedCFCPlot(FigureCanvasGTK):
    def __init__(self):
        self.cumulated = 0
        self.generations = 0
        self.cumulated_cfc_trajectory = []
        self.figure = Figure(dpi=75, facecolor='#e1e1e1')
        self.figure.suptitle('cumulated constraint function calls', fontsize=12)
        self.axis = self.figure.add_subplot(111)        
        self._setup()

        super(CumulatedCFCPlot, self).__init__(self.figure)            

    def _setup(self):
        self.axis.cla()
        self.axis.set_xlabel("generations")
        self.axis.set_ylabel("cum. constraint calls") 
        self.axis.grid(True)

    def on_reset(self):
        self.cumulated = 0
        self.generations = 0
        self.cumulated_cfc_trajectory = []
        self.draw_idle()

    def on_draw(self):
        self._setup()
        generations = range(0, self.generations)
        self.axis.plot(generations, 
            self.cumulated_cfc_trajectory, color='green', marker="o")
        self.draw_idle()

    def on_update(self, stats):
        cfc = stats['cfc']

        self.generations += 1
        self.cumulated += cfc
        self.cumulated_cfc_trajectory.append(self.cumulated) 
