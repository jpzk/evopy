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

from copy import deepcopy
from numpy import matrix, log10, array, linspace, sqrt, pi, exp

from pickle import load
from scipy import stats
from evopy.strategies.ori_dses_svc_repair import ORIDSESSVCR
from evopy.strategies.ori_dses_svc import ORIDSESSVC
from evopy.strategies.ori_dses import ORIDSES
from evopy.simulators.simulator import Simulator

from evopy.problems.sphere_problem_origin_r1 import SphereProblemOriginR1
from evopy.problems.sphere_problem_origin_r2 import SphereProblemOriginR2
from evopy.problems.schwefels_problem_26 import SchwefelsProblem26
from evopy.problems.tr_problem import TRProblem

from evopy.metamodel.dses_svc_linear_meta_model import DSESSVCLinearMetaModel
from sklearn.cross_validation import KFold
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.operators.scaling.scaling_dummy import ScalingDummy
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear

from evopy.operators.termination.or_combinator import ORCombinator
from evopy.operators.termination.accuracy import Accuracy
from evopy.operators.termination.generations import Generations
from evopy.operators.termination.convergence import Convergence 

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pylab import hist, plot
from setup import *

bff = file("output/best_fitness_file.save", "r")
best_fitness = load(bff)

for problem in problems:
    figure_hist = plt.figure(figsize=(8,6), dpi=10, facecolor="w", edgecolor="k")
    logit = lambda value, optimum : log10(value - optimum)
    opt = problem().optimum_fitness()

    plt.xlabel('Fitnessgenauigkeit')
    plt.ylabel('relative H' + u'ä' + 'ufigkeit')

    x1 = best_fitness[problem][optimizers[problem][0]]
    x2 = best_fitness[problem][optimizers[problem][1]]  
    x1_log = map(logit, x1, len(x1) * [opt])
    x2_log = map(logit, x2, len(x2) * [opt])

    minimum = min(x1_log + x2_log)
    maximum = max(x1_log + x2_log)

    minimum, maximum =  minimum - 2, maximum + 2
    plt.xlim([minimum, maximum])

    pdfs1, bins1, patches1 = hist(x1_log, normed=True, alpha=0.5,\
        histtype='step', edgecolor="g")

    kernel1 = stats.gaussian_kde(x1_log)

    # scipy 0.10.1 requires setting the bandwith manually
    # @deprecated bw_method attribute in scipy 0.11    
    kernel1.covariance_factor = stats.gaussian_kde.silverman_factor

    X1 = linspace(minimum, maximum, 100)
    Y1 = kernel1(X1)    
    plot(X1,Y1, linestyle="--", color="g")

    pdfs2, bins2, patches2 = hist(x2_log, normed=True, alpha=0.5,\
        histtype='step', edgecolor="#004779")

    kernel2 = stats.gaussian_kde(x2_log)

    # scipy 0.10.1 requires setting the bandwith manually
    # @deprecated bw_method attribute in scipy 0.11    
    kernel2.covariance_factor = stats.gaussian_kde.silverman_factor

    X2 = linspace(minimum, maximum, 100)
    Y2 = kernel2(X2)    
    plot(X2,Y2, linestyle="-", color="#004779")

    pp = PdfPages("output/%s.pdf" % str(problem).replace('.', '-'))
    plt.savefig(pp, format='pdf')
    pp.close()
    plt.clf()


