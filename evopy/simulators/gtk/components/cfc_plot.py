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

class CFCPlot(FigureCanvasGTK):
    def __init__(self):
        self.cfc_trajectory = []
        self.figure = Figure(dpi=75, facecolor='#e1e1e1')
        self.figure.suptitle('constraint function calls', fontsize=12)
        self.axis = self.figure.add_subplot(111)
        self._setup()

        super(CFCPlot, self).__init__(self.figure)            

    def on_reset(self):
        self.cfc_trajectory = []
        self.axis.cla()
        self.axis.grid(True)
        self.draw_idle()

    def _setup(self):
        self.axis.cla()
        self.axis.set_xlabel("generations")
        self.axis.set_ylabel("constraint calls per gen.") 
        self.axis.grid(True)

    def on_draw(self):
        self._setup() 
        self.axis.plot(self.generations, 
            self.cfc_trajectory, color='green', marker="o")
        self.draw_idle()

    def on_update(self, stats):
        cfc = stats['cfc']
        self.cfc_trajectory.append(cfc)       
        self.generations = range(0, len(self.cfc_trajectory))
