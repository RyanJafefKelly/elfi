import numpy as np
from numpyctypes import c_ndarray
import elfi
import matplotlib.pyplot as plt
from elfi.examples import owl
from elfi.methods.bsl import pre_sample_methods, pdf_methods
# from elfi.clients.multiprocessing import Client as MultiprocessingClient
from ctypes import cdll
import multiprocessing as mp
import pickle as pkl

# import rasterio
import pandas as pd
import datetime
import logging

import os
import time

logger = logging.getLogger(__name__)

def run_owl():
    np.random.seed(123)
    logs_dir = 'logs'

    # Check if the directory exists
    if not os.path.exists(logs_dir):
        # If it doesn't exist, create it
        os.makedirs(logs_dir)

    logging.basicConfig(filename="logs/owls_" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S") + ".log", level=logging.DEBUG)
    ind_data_start_time = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_start_time.txt")
    ind_data_start_day = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_start_day.txt")
    ind_data_centroid = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_centroid.csv")

    # true_params = np.array([2, 1.5, 1, 5])  # TODO

    # elfi.set_client(MultiprocessingClient(num_processes=6))  # TODO: Hardcoded


    m = owl.get_model(seed_obs=123, observed=False)
    batch_size = 50

    feature_names = 'S'

    # NOTE: could do presampling stuff here

    # TODO: how many simulations to run?
    # TODO: what does the distribution look like?

    likelihood = pdf_methods.semiparametric_likelihood()

    n_sim_round = 300
    # batch_size = 300
    semi_bsl = elfi.BSL(m, n_sim_round=n_sim_round,
                        batch_size=batch_size, seed=1, likelihood=likelihood)

    rho = 2.0
    k = 1.5
    tau = 1.0
    lmda_r = 5.0
    params = {'k': k, 'lmda_r': lmda_r, 'rho': rho, 'tau': tau}

    mcmc_iterations = 20  # sample size
    # est_post_cov = np.array([[0.02, 0.01], [0.01, 0.02]])  # covariance matrix for the proposal distribution
    logit_transform_bound = np.array([
                                    [0.1, 4],  # k
                                    [0.01, 30],  # lmdr_r
                                    [0.1, 5],  # rho
                                    [0.1, 3.5]  # tau
                                   ])

    # TODO! DOUBLE CHECK ROWS MATCH UP WITH BOUNDS
    sigma_proposals = np.array([[ 1.82484162,  0.04122204, -0.22807996, -0.05305404],
       [ 0.04122204,  3.41680703, -0.15229068,  0.17158555],
       [-0.22807996, -0.15229068,  2.46904036,  0.0493929 ],
       [-0.05305404,  0.17158555,  0.0493929 ,  2.54673614]])


    # pre-sampling
    # check summary statistics shape
    # nsim = 10000
    # seed = 1
    # pre_sample_methods.plot_features(m, params, nsim, feature_names, seed=seed)
    # plt.savefig("owl_features.png")

    # TODO: should also check log SL stdev

    tic = time.time()
    res = semi_bsl.sample(mcmc_iterations,
                          sigma_proposals=sigma_proposals,  # TODO: check if works with transform...
                          params0=np.array([k, lmda_r, rho, tau]),
                          param_names=['k', 'lmda_r', 'rho', 'tau'],
                          logit_transform_bound=logit_transform_bound,
                          )
    toc = time.time()
    print("BSL sampling time: ", toc - tic)
    print(res)
    with open("owl_semibsl.pkl", "wb") as f:
        pkl.dump(res, f)
    # print(res.compute_ess())
    # res.plot_traces();
    # plt.savefig("owl_traces.png")

    # res.plot_pairs(reference_values=params,
    #                bins=30);
    # plt.savefig("owl_pairs.png")
    # owl_bsl_sampler = elfi.BSL(m['SL'], batch_size=batch_size, seed=123)

    # owl_bsl_sampler.plot_summary_statistics(batch_size=100, theta_point=true_params)
    # plt.savefig("owl_new_summaries_100.png")

    # TODO! MAKE DICT POSSIBLE

    # logger.info("TEST!")
    # params0 = {
    #     'k': 1.5,
    #     'lmda_r': 5,
    #     'rho': 2,
    #     'tau': 1
    # }
    # bsl_res = owl_bsl_sampler.sample(100,
    #                                  burn_in=0,
    #                                  params0=params0,
    #                                  sigma_proposals=0.2*np.eye(4),
    #                                  logit_transform_bound=logit_transform_bound
    #                                  )

    # bsl_res.plot_traces()
    # plt.savefig("owl_traces_semiBsl.png")

    # bsl_res.plot_marginals(bins=30)

    # plt.savefig("owl_marginals_semiBsl.png")

    # bsl_res.plot_pairs(bins=30)
    # plt.savefig("owl_pairs_semiBsl.png")

    # est_cov_mat = bsl_res.get_sample_covariance()
    # print('est_cov_mat', est_cov_mat)

if __name__ == '__main__':
    run_owl()
