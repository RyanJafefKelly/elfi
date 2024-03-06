import numpy as np
from numpyctypes import c_ndarray
import elfi
import matplotlib.pyplot as plt
from elfi.examples import owl
from elfi.methods.bsl.estimate_whitening_matrix import \
    estimate_whitening_matrix
from elfi.methods.bsl.select_penalty import select_penalty
from ctypes import cdll

# import rasterio
import pandas as pd
import datetime
import logging

import time

logger = logging.getLogger(__name__)

def run_owl():
    logging.basicConfig(filename="logs/owls_" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S") + ".log", level=logging.DEBUG)
    ind_data_start_time = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_start_time.txt")
    ind_data_start_day = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_start_day.txt")
    ind_data_centroid = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_centroid.csv")

    # true_params = np.array([2, 1.5, 1, 5])  # TODO
    m = owl.get_model(seed_obs=123, parallelise=True, num_processes=64, observed=False)
    batch_size = 50
    owl_bsl_sampler = elfi.BSL(m['SL'], batch_size=batch_size, seed=123)

    # owl_bsl_sampler.plot_summary_statistics(batch_size=100, theta_point=true_params)
    # plt.savefig("owl_new_summaries_100.png")

    # TODO! MAKE DICT POSSIBLE
    logit_transform_bound = np.array([
                                    [0.1, 4],
                                    [0.01, 30],
                                    [0.1, 5],
                                    [0.1, 3.5]
                                   ])

    logger.info("TEST!")
    params0 = {
        'k': 1.5,
        'lmda_r': 5,
        'rho': 2,
        'tau': 1
    }
    bsl_res = owl_bsl_sampler.sample(100,
                                     burn_in=0,
                                     params0=params0,
                                     sigma_proposals=0.2*np.eye(4),
                                     logit_transform_bound=logit_transform_bound
                                     )

    bsl_res.plot_traces()
    plt.savefig("owl_traces_semiBsl.png")

    bsl_res.plot_marginals(bins=30)

    plt.savefig("owl_marginals_semiBsl.png")

    bsl_res.plot_pairs(bins=30)
    plt.savefig("owl_pairs_semiBsl.png")

    est_cov_mat = bsl_res.get_sample_covariance()
    print('est_cov_mat', est_cov_mat)

if __name__ == '__main__':
    run_owl()
