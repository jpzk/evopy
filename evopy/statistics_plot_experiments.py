''' 
This file is part of evolutionary-algorithms-sandbox.

evolutionary-algorithms-sandbox is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

evolutionary-algorithms-sandbox is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
evolutionary-algorithms-sandbox.  If not, see <http://www.gnu.org/licenses/>.
'''

from sys import argv
from numpy import mean, var, std, unique
from matplotlib.mlab import rec2txt, csv2rec, rec_groupby
from pylab import boxplot, show, xticks, setp, figure, subplots_adjust

data = csv2rec(argv[1], delimiter = ';')

def boxplots(data, methods, attribute):
    fig = figure(figsize=(10,5))
    fig.canvas.set_window_title('constraint-calls')
    ax1 = fig.add_subplot(111)
    subplots_adjust(left=0.175, right=0.95, top=0.9, bottom=0.30)
    
    boxplot_data_lists = []
    for method in methods:
        boxplot_data_lists.append(\
            data[data['method'] == method][attribute])
    boxplot(boxplot_data_lists)
    xticks(range(1,len(methods)+1), methods)
   
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)

    ax1.set_title(attribute, fontsize = 'large', weight = 'bold')
    ax1.set_xlabel('method')
    ax1.set_ylabel(attribute)
    
    xtickNames = setp(ax1, xticklabels = methods)
    setp(xtickNames, rotation=45, fontsize=8)

methods = unique(data['method']) 
boxplots(data, methods, argv[2])
show()
