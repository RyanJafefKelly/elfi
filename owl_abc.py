import numpy as np
from numpyctypes import c_ndarray
import elfi
import matplotlib.pyplot as plt
from elfi.examples import owl
from elfi.methods.bsl import pre_sample_methods, pdf_methods
from elfi.clients.multiprocessing import Client as MultiprocessingClient
from ctypes import cdll
import multiprocessing as mp
import pickle as pkl
from elfi.examples.owl import summary_stats

# import rasterio
import pandas as pd
import datetime
import logging

import os
import time

logger = logging.getLogger(__name__)


def run_owl():
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
    elfi.set_client(MultiprocessingClient(num_processes=4))

    m = owl.get_model(seed_obs=123, observed=False)
    observed_summaries = summary_stats(m.observed['OWL'])
    # rej = elfi.Rejection(m['d'], batch_size=1, seed=123)
    # schedule = [2.0e+3, 5e+2, 1e+2]
    smc_abc = elfi.AdaptiveThresholdSMC(m['d'], batch_size=1, seed=123, q_threshold=0.99)

    tic = time.time()
    sample_smc_abc = smc_abc.sample(300, max_iter=3)
    toc = time.time()
    print("SMC ABC sampling time: ", toc - tic)
    print(sample_smc_abc)
    # print(sample_smc_abc.compute_ess())

    with open("owl_smc_abc.pkl", "wb") as f:
        pkl.dump(sample_smc_abc, f)

    # sample_smc_abc.plot_traces()
    # plt.savefig("owl_traces.png")

    rho = 2.0
    k = 1.5
    tau = 1.0
    lmda_r = 5.0

    params = {'k': k, 'lmda_r': lmda_r, 'rho': rho, 'tau': tau}

    sample_smc_abc.plot_marginals(reference_values=params,
                                  bins=30)

    sample_smc_abc.plot_pairs(reference_values=params,
                              bins=30)
    plt.savefig("owl_pairs.png")

if __name__ == '__main__':
    run_owl()
