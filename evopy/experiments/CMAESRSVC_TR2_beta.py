from sys import path
path.append("../..")

from multiprocessing import cpu_count
from evopy.external.playdoh import map as pmap
from evopy.simulators.csv_writer import CSVWriter
from evopy.simulators.experiment_simulator import ExperimentSimulator

from sklearn.cross_validation import KFold
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear
from evopy.strategies.cmaes_rsvc import CMAESRSVC
from evopy.metamodel.cma_svc_linear_meta_model import CMASVCLinearMetaModel
from evopy.metamodel.svc_linear_meta_model import SVCLinearMetaModel
from evopy.problems.tr_problem import TRProblem

def get_method(beta):

    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(20, 5))

    meta_model = CMASVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        repair_mode = 'mirror')

    method = CMAESRSVC(\
        mu = 15,
        lambd = 100,
        xmean = [5.0, 5.0],
        sigma = 1.0,
        beta = beta,
        meta_model = meta_model)

    return method

simulators = []

for beta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for i in range(0, 50):
        simulator1 = ExperimentSimulator(get_method(beta), TRProblem(dimensions=2), 10**-12)
        simulators.append(simulator1)
        writer = CSVWriter("beta" + str(beta))
        results = pmap(process, simulators, cpu = cpu_count)
        for result in results:
            writer.write(result)
