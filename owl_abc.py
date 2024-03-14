import numpy as np
from numpyctypes import c_ndarray
import elfi
import matplotlib.pyplot as plt
from elfi.examples import owl
from elfi.methods.bsl import pre_sample_methods, pdf_methods
from ctypes import cdll

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
    m = owl.get_model(seed_obs=123, observed=False)

    rej = elfi.Rejection(m['d'], batch_size=1, seed=123)
    tic = time.time()
    res = rej.sample(n_samples=100, quantile=0.01)
    toc = time.time()
    print("Rejection sampling time: ", toc - tic)
    print(res)

if __name__ == '__main__':
    run_owl()
