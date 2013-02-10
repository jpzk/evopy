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

import csv
from copy import deepcopy
from numpy import matrix, log10, array
from scipy.stats import wilcoxon 
from scipy import polyfit, polyval 
from itertools import chain
from pylab import plot 
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt

f = open("output/cpus2.csv")
reader = csv.reader(f)

cpus_ids = range(0,8)
cpus = {}
for i in cpus_ids:
    cpus[i] = [] 

indices = [0,6,12,18,24,30,36,42]
for row in reader:
    for cpu in cpus_ids:
        cpus[cpu].append(row[indices[cpu]])

figure_accs = plt.figure(figsize=(8,6), dpi=10, facecolor="w", edgecolor="k")
plt.xlabel("Dauer in Sekunden" )
plt.ylabel("CPU Auslastung")
plt.xlim([0, 250])
time_range = range(0, len(cpus[0]))
#for cpu in cpus_ids:
   # plot(time_range, cpus[cpu], linestyle="-", color="#CCCCCC")
for cpu in cpus_ids:
    plot(time_range, cpus[cpu], linestyle="none", marker=".", color="#044977")

pp = PdfPages("output/par2.pdf")
plt.savefig(pp, format='pdf')
pp.close()
 
