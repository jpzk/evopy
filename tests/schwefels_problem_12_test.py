from evopy.strategies.cmaes import CMAES
from evopy.problems.schwefels_problem_12 import SchwefelsProblem12
from evopy.simulators.simulator import Simulator
from sklearn.cross_validation import KFold
from evopy.operators.scaling.scaling_standardscore import ScalingStandardscore
from evopy.metamodel.cv.svc_cv_sklearn_grid_linear import SVCCVSkGridLinear
from evopy.strategies.cmaes_rsvc import CMAESRSVC
from evopy.metamodel.svc_linear_meta_model import SVCLinearMetaModel
from evopy.problems.schwefels_problem_12 import SchwefelsProblem12
from evopy.simulators.simulator import Simulator
from evopy.strategies.cmaes_svc import CMAESSVC
from evopy.metamodel.cma_svc_linear_meta_model import CMASVCLinearMetaModel

def get_method_cmaessvc():

    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 5, 2)],
        cv_method = KFold(20, 5))

    meta_model = CMASVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        repair_mode = 'mirror')

    method = CMAESSVC(\
        mu = 15,
        lambd = 100,
        xmean = [100.0, 100.0],
        sigma = 1.0,
        beta = 0.9,
        meta_model = meta_model) 

    return method

def get_method_cmaesrsvc():

    sklearn_cv = SVCCVSkGridLinear(\
        C_range = [2 ** i for i in range(-5, 15, 2)],
        cv_method = KFold(20, 5))

    meta_model = SVCLinearMetaModel(\
        window_size = 10,
        scaling = ScalingStandardscore(),
        crossvalidation = sklearn_cv,
        repair_mode = 'mirror')

    method = CMAESRSVC(\
        mu = 15,
        lambd = 100,
        xmean = [100.0, 100.0],
        sigma = 1.0,
        beta = 0.80,
        meta_model = meta_model)

    return method

def get_method_cmaes():
    method = CMAES(\
        mu = 15,
        lambd = 100,
        xmean = [100.0, 100.0],
        sigma = 1.0)

    return method

def CMAES_schwefels_problem_12_test():
    sim = Simulator(get_method_cmaes(), SchwefelsProblem12(), pow(10, -12))
    results = sim.simulate()

def CMAESRSVC_schwefels_problem_12_test():
    sim = Simulator(get_method_cmaesrsvc(), SchwefelsProblem12(), pow(10, -12))
    results = sim.simulate()

def CMAESSVC_schwefels_problem_12_test():
    sim = Simulator(get_method_cmaessvc(), SchwefelsProblem12(), pow(10, -12))
    results = sim.simulate()
