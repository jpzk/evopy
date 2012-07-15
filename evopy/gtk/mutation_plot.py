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

from numpy import array, max, fabs, arange 

from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk import FigureCanvasGTK, NavigationToolbar
from matplotlib.ticker import NullFormatter

# http://matplotlib.sourceforge.net/examples/pylab_examples/scatter_hist.html
class MutationPlot(FigureCanvasGTK):
    def __init__(self):

        self.figure = Figure(dpi=75, facecolor='#e1e1e1')
        #self.figure.suptitle('selected children', fontsize=12)
 
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left+width+0.02

        nullfmt = NullFormatter() 

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]

        self.axScatter = self.figure.add_axes(rect_scatter)
        self.axHistx = self.figure.add_axes(rect_histx)
        self.axHisty = self.figure.add_axes(rect_histy)

        self.axScatter.grid(True)
        self.axHistx.grid(True)
        self.axHisty.grid(True)

        self.axHistx.xaxis.set_major_formatter(nullfmt)
        self.axHisty.yaxis.set_major_formatter(nullfmt)

        super(MutationPlot, self).__init__(self.figure)

    def on_reset(self):      
        self.axScatter.cla()
        self.axHistx.cla()
        self.axHisty.cla()
        self.axScatter.grid(True)
        self.axHistx.grid(True)
        self.axHisty.grid(True)
        self.draw_idle()

    def on_update(self, stats):  
        values = stats['selected_children']
    
        dimensions = len(values[0])

        self._mean = []
        self._std = []
        for d in range(0, dimensions):
            vals = map(lambda u : u[d], values)
            self._mean.append(array(vals).mean())
            self._std.append(array(vals).std())

        scaled_values = []
        for value in values:
            scaled_value = []
            for d in range(0, dimensions):
                old_value = value[d]
                scaled_value.append((old_value - self._mean[d])/ self._std[d])
            scaled_values.append(scaled_value)                            

        xv = lambda value : value[0]
        yv = lambda value : value[1]
        X = map(xv, scaled_values)
        Y = map(yv, scaled_values)

        self.axScatter.cla()
        self.axHistx.cla()
        self.axHisty.cla()
        self.axScatter.grid(True)
        self.axHistx.grid(True)
        self.axHisty.grid(True)
        self.draw_idle()

        binwidth = 0.25
        xymax = max( [max(fabs(X)), max(fabs(Y))] )
        lim = ( int(xymax/binwidth) + 1) * binwidth

        self.axScatter.set_xlim( (-lim, lim) )
        self.axScatter.set_ylim( (-lim, lim) )

        bins = arange(-lim, lim + binwidth, binwidth)
        self.axHistx.hist(X, bins=bins, color='green')
        self.axHisty.hist(Y, bins=bins, orientation='horizontal', color='green')

        self.axHistx.set_xlim( self.axScatter.get_xlim() )
        self.axHisty.set_ylim( self.axScatter.get_ylim() )

        self.axScatter.scatter(X, Y, color='green')
        self.draw_idle()

