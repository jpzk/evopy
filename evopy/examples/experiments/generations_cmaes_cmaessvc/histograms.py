# Using the magic encoding
# -*- coding: utf-8 -*-

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

from scipy import stats
from copy import deepcopy
from numpy import matrix, log10, array, linspace, sqrt, pi, exp
from pickle import load 
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pylab import hist, plot
from setup import *

gens = file("output/generations_file.save", "r")
generations = load(gens)


for problem in problems:
    figure_hist = plt.figure(figsize=(8,6), dpi=10, facecolor="w", edgecolor="k")

    plt.xlabel('Generationen')
    plt.ylabel('relative H' + u'ä' + 'ufigkeit')

    x1 = map(lambda l : l[-1], generations[problem][optimizers[problem][0]])
    x2 = map(lambda l : l[-1], generations[problem][optimizers[problem][1]])

    minimum = min(x1 + x2)
    maximum = max(x1 + x2)

    x1, x2 = map(float, x1), map (float, x2)

    minimum, maximum = minimum - 10, maximum + 10 
    plt.xlim([minimum, maximum])
    
    pdfs1, bins1, patches1 = hist(x1, normed=True, alpha=0.5,\
        histtype='step', bins = range(minimum, maximum + 5, 5),\
        edgecolor="g")

    kernel1 = stats.gaussian_kde(x1)

    # scipy 0.10.1 requires setting the bandwith manually
    # @deprecated bw_method attribute in scipy 0.11    
    kernel1.covariance_factor = stats.gaussian_kde.silverman_factor

    X1 = linspace(minimum, maximum, 1000)
    Y1 = kernel1(X1)    
    plot(X1,Y1, linestyle="--", color="g")

    pdfs2, bins2, patches2 = hist(x2, normed=True, alpha=0.5,\
        histtype='step', bins = range(minimum, maximum + 5, 5),\
        edgecolor="#004779")

    kernel2 = stats.gaussian_kde(x2)

    # scipy 0.10.1 requires setting the bandwith manually
    # @deprecated bw_method attribute in scipy 0.11    
    kernel2.covariance_factor = stats.gaussian_kde.silverman_factor

    X2 = linspace(minimum, maximum, 1000)
    Y2 = kernel2(X2)    
    plot(X2,Y2, linestyle="-", color="#004779")

    pp = PdfPages("output/%s.pdf" % str(problem).replace('.', '-'))
    plt.savefig(pp, format='pdf')
    pp.close()
    plt.clf()


