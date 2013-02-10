''' 
This file is part of evopy.

Copyright 2012 - 2013, Jendrik Poloczek

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

from sys import path
path.append("../../../..")

from pickle import load 
from copy import deepcopy
from numpy import matrix, log10, array
from scipy.stats import wilcoxon 
from scipy import polyfit, polyval 
from itertools import chain
from pylab import plot 
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
from setup import * 

durationsfile = file("output/durations.save", "r")
durations = load(durationsfile)

figure_accs = plt.figure(figsize=(8,6), dpi=10, facecolor="w", edgecolor="k")
plt.xlabel("Anzahl der Stichproben" )
plt.ylabel("Dauer in Minuten")
plt.xlim([10, 100])
#plt.ylim([100.0, 200.0])

import pdb

colors = {
    True: "#044977",
    False: "g"
}

markers = {
    True: ".",
    False: "."
}

for use_par in [True, False]:
    X = range(10, 110, 10)
    y = lambda x : durations[use_par][x]
    Y = map(lambda s : (s / 1000.0) / 60, map(y, X))
   
    plot(X, Y, linestyle = "none",\
        color=colors[use_par], marker=markers[use_par])

    (ar, br) = polyfit(X, Y, 1)
    YR = polyval([ar,br], X)
    plot(X, YR, color=colors[use_par],\
        marker=markers[use_par], linestyle="--") 
    
    #for sample_size in range(10, 100, 10):          
    #    durations[use_par]

pp = PdfPages("output/par.pdf")
plt.savefig(pp, format='pdf')
pp.close()
   
