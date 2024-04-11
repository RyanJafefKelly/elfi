"""Little owls..."""

import os
import warnings

import numpy as np
# from numpyctypes import c_ndarray
from sklearn.linear_model import LinearRegression, QuantileRegressor
import rasterio
from numpyctypes import c_ndarray
import ctypes
# from contextlib import redirect_stdout
import multiprocessing as mp
from functools import partial

from scipy.stats import skew
from scipy.spatial import ConvexHull
from scipy.spatial.distance import  
import elfi
import pandas as pd
import time
import uuid

# TODO: yucky global stuff
# dat = np.zeros((15092, 16730))
# i = 0
# with rasterio.open("habitatSuitability_scaled_20.tif") as src:
#     profile = src.profile.copy()
#     print(src.profile)
#     for ji, window in src.block_windows(1):
#         window_dat = src.read(window=window)
#         window_dat[np.where(window_dat < 0)] = 0
#         dat[:, i] = window_dat.flatten()
#         i += 1

# TODO: really need to sort out this ugly global shit ...
# dat = np.loadtxt("habsuitcalib_new.txt")
# dat = np.transpose(dat)  # TODO! TESTING
# dat = np.ones((5829, 6489))

# environment_c = c_ndarray(dat, dtype=np.double, ndim=2)
# lib = ctypes.cdll.LoadLibrary('./runsim.so')  # TODO! UPDATE

# def moderate_negatively_skewed(x):
#     """Moderately negatively skewed data to more normal"""
#     return np.sqrt(max())

def load_and_process_data(filepath):
    data = np.zeros((15092, 16730))
    i = 0
    with rasterio.open(filepath) as src:
        print(src.profile)
        for ji, window in src.block_windows(1):
            window_data = src.read(window=window)
            window_data[np.where(window_data < 0)] = 0
            data[:, i] = window_data.flatten()
            i += 1
    return data

def load_shared_library(lib_path):
    lib = ctypes.cdll.LoadLibrary(lib_path)
    return lib

def prepare_inputs(*inputs, **kwinputs):
    """Prepare the inputs for the simulator

    """
    rho, k, tau, lmda_r = inputs
    if 'meta' in kwinputs:
        meta = kwinputs['meta']
        filename = '{model_name}_{batch_index}_{submission_index}.txt'.format(**meta)
    else:
        filename = str(uuid.uuid4()) + '.txt'
        # filename = ' a_temp_file.txt'
    # Organize the parameters to an array. The broadcasting works nicely with constant
    # arguments.

    try:
        random_state = kwinputs['random_state']
        seed = random_state.integers(low=0, high=1e+9)
    except AttributeError:
        random_state = np.random
        seed = random_state.randint(1e+9)

    kwinputs['seed'] = seed
    # param_array = np.row_stack(np.broadcast(rho, k, tau, lmda_r))

    # Prepare a unique filename for parallel settings
    # np.savetxt(filename, param_array, fmt='%.4f %.4f %.4f %d')

    # Add the filenames to kwinputs
    # TODO! HARDCODED ONE FOR NOW
    ind_data = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_data.csv",
                    header=None,
                    names=["timestamp", "xObserved", "yObserved",
                            "stepDistanceObserved", "tuningAngleObserved",
                            "habitatSuitabilityObserved", "rsc"])
    ind_data_start_time = 0
    with open("individual_data/individual_data/calibration_2011VA0533_start_time.txt") as f:
        lines = f.readlines()
        ind_data_start_time = float(lines[0].strip())

    ind_data_start_day = 0
    with open("individual_data/individual_data/calibration_2011VA0533_start_day.txt") as f:
        lines = f.readlines()
        ind_data_start_day = int(lines[0].strip())

    # ind_data_start_time = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_start_time.txt")
    # ind_data_start_day = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_start_day.txt")
    ind_data_centroid = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_centroid.csv", header=None)
    ind_data = np.array(ind_data)

    # str_timestamp = [str(int(time_val)) for time_val in ind_data[:, 0]]
    # str_timestamp = ' '.join(str_timestamp)
    times = np.array(ind_data[:, 0], dtype='uintc')



    kwinputs['s_time'] = ind_data_start_time
    kwinputs['s_day'] = ind_data_start_day
    kwinputs['sol_lat'] = ind_data_centroid.values[0][0]
    kwinputs['sol_long'] = ind_data_centroid.values[1][0]

    kwinputs['upper_left_left'] = 3436195  # TODO
    kwinputs['upper_left_top'] = 5474136  # TODO

    kwinputs['observation_error'] = 2.4  # TODO! HARDCODED OBSERVATION ERROR FROM AccMetric R - LOSIM/code/calibrationScript.R
    kwinputs['environment_c'] = c_ndarray(kwinputs['environment_c'], dtype=np.double, ndim=2)

    # if 'seed' not in kwinputs:
    #     seed =  rng.integers(low=0, high=1e+9)  # TODO? want bigger...
    #     kwinputs['seed'] = seed

    kwinputs['times'] = times

    kwinputs['filename'] = filename
    kwinputs['output_filename'] = filename[:-4] + '_out.txt'

    return inputs, kwinputs


def process_result(completed_process, *inputs, **kwinputs):
    """Process the result of the owl simulation.

    The signature follows that given in `elfi.tools.external_operation`
    """
    # output_filename = kwinputs['output_filename']

    # # simulations = np.loadtxt(output_filename)
    # # simulations = pd.read_fwf(output_filename)
    # with open(output_filename, 'r') as f:
    #     # for line in f:
    #     #     print('line', line)
    #     #     for val in line.split(sep=";")
    #     simulations = [float(val) for line in f for val in line.strip().split(sep=',')]
    #     # for line in f:
    #     #     for word in line.split()

    # # Clean up the files after reading the data in
    # # os.remove(kwinputs['filename'])
    # os.remove(output_filename)

    return completed_process



# Create an external operation callable
# TODO: add seed, etc.

def invoke_simulation(*inputs, **kwinputs):
    lib = ctypes.cdll.LoadLibrary('./runsim.so')
    lib.run_sim_wrapper.restype = ctypes.c_int

    k, lmda_r, rho, tau = inputs
    # Retrieve inputs and convert to ctypes as needed
    env_res = 20
    environment_c, s_time, s_day, sol_lat, sol_long, observation_error = \
        kwinputs['environment_c'], kwinputs['s_time'], kwinputs['s_day'], \
        kwinputs['sol_lat'], kwinputs['sol_long'], kwinputs['observation_error']
    upper_left_left, upper_left_top, seed, times = kwinputs['upper_left_left'], \
                                                     kwinputs['upper_left_top'], \
                                                     kwinputs['seed'], \
                                                     kwinputs['times']
    times_c = c_ndarray(times, dtype=np.uintc, ndim=1)
    # Perform the call to the shared library function
    batch_size = 1  # TODO
    random_state = np.random  # TODO

    # Prepare the result array
    result_size = 2048  # Must match the expected size
    ResultArrayType = ctypes.c_double * result_size  # Define the array type
    result_array = ResultArrayType()  # Create an instance of the array
    res_all = []

    for i in range(batch_size):
        if hasattr(random_state, 'integers'):
            seed = random_state.integers(low=0, high=1e+9)
        else:
            seed = random_state.randint(0, 1e+9)

        result_status = lib.run_sim_wrapper(ctypes.c_double(rho), ctypes.c_double(k), ctypes.c_double(tau),
                        ctypes.c_double(lmda_r), env_res, ctypes.c_double(s_time), s_day,
                        ctypes.c_double(sol_lat), ctypes.c_double(sol_long),
                        ctypes.c_double(observation_error),
                        upper_left_left, upper_left_top,
                        ctypes.c_ulonglong(seed), times_c, environment_c,
                        ctypes.byref(result_array), result_size)

        result_np = np.ctypeslib.as_array(result_array)
        # np_res = np.array([el for el in res.contents])
        # np_res = np_res[:(7 * len(times))]  # trim down from 2048 to actual data
        # np_res = np_res.reshape((-1, 7))
        np_res = result_np[:(7 * len(times))]  # trim down from 2048 to actual data
        np_res = np_res.reshape((-1, 7))

        res_all.append(np_res)

    # Assume result processing is needed to convert ctypes to numpy array, etc.
    # processed_result = process_result(result, *inputs, **kwinputs)
    
    return res_all

# def owl(k, lmda_r, rho, tau, batch_size=1, random_state=None, **kwargs):
#     ind_data = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_data.csv",
#                 header=None,
#                 names=["timestamp", "xObserved", "yObserved",
#                         "stepDistanceObserved", "tuningAngleObserved",
#                         "habitatSuitabilityObserved", "rsc"])
#     ind_data_start_time = 0
#     with open("individual_data/individual_data/calibration_2011VA0533_start_time.txt") as f:
#         lines = f.readlines()
#         ind_data_start_time = float(lines[0].strip())

#     ind_data_start_day = 0
#     with open("individual_data/individual_data/calibration_2011VA0533_start_day.txt") as f:
#         lines = f.readlines()
#         ind_data_start_day = int(lines[0].strip())

#     # ind_data_start_time = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_start_time.txt")
#     # ind_data_start_day = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_start_day.txt")
#     ind_data_centroid = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_centroid.csv", header=None)
#     ind_data = np.array(ind_data)
#     times = np.array(ind_data[:, 0], dtype='uintc')
#     # str_timestamp = [str(int(time_val)) for time_val in ind_data[:, 0]]
#     # str_timestamp = ' '.join(str_timestamp)

#     s_time = ind_data_start_time
#     s_day = int(ind_data_start_day)
#     sol_lat = ind_data_centroid.values[0][0]
#     sol_long = ind_data_centroid.values[1][0]

#     upper_left_left = 3436195  # TODO
#     upper_left_top = 5474136  # TODO

#     random_state = random_state or np.random

#     # if 'seed' not in kwinputs:
#     #     seed =  rng.integers(low=0, high=1e+9)  # TODO? want bigger...
#     #     kwinputs['seed'] = seed

#     # times = str_timestamp
#     times_c = c_ndarray(times, dtype=np.uintc, ndim=1)
#     env_res = 20  # TODO: FLEXIBLE
#     tic = time.time()

#     # with open(filename, 'w') as f, redirect_stdout(f):
#     # func = lib.run_sim_wrapper
#     # environment_dummy = np.zeros((100, 100))
#     # environment_dummy_c = c_ndarray(environment_dummy, dtype=np.double, ndim=2)
#     func.restype = ctypes.POINTER(ctypes.c_double * 2048)
#     if hasattr(random_state, 'integers'):
#         seed = random_state.integers(low=0, high=1e+9)
#     else:
#         seed = random_state.randint(0, 1e+9)

#     # print('seed', seed)
#     # TODO! HARDCODED OBSERVATION ERROR FROM AccMetric R - LOSIM/code/calibrationScript.R
#     # observation_error = np.random.normal(35.03423, 33.47957)
#     observation_error = 2.4

#     # TODO! TRANSLATE SPATIAL CO-ORDS / ANYTHING ELSE RELEVANT

#     res = lib.run_sim_wrapper_wrapper(ctypes.c_double(rho), ctypes.c_double(k), ctypes.c_double(tau),
#                     ctypes.c_double(lmda_r), env_res, ctypes.c_double(s_time), s_day,
#                     ctypes.c_double(sol_lat), ctypes.c_double(sol_long),
#                     ctypes.c_double(observation_error),
#                     upper_left_left, upper_left_top,
#                     ctypes.c_ulonglong(seed), times_c,
#                     environment_c)
#     np_res = np.array([el for el in res.contents])
#     np_res = np_res[:(7 * len(times))]  # trim down from 2048 to actual data
#     np_res = np_res.reshape((-1, 7))
#     # print("time: ", toc - tic)
#     return np_res


# def owl_batch(k, lmda_r, rho, tau, batch_size=1, meta=None, random_state=None, **kwargs):
#     if hasattr(rho, '__len__') and len(rho) > 1:
#         rho = rho[0]
#         k = k[0]
#         tau = tau[0]
#         lmda_r = lmda_r[0]
#     else:  # assumes something array like passed
#         pass
#         # rho = rho[0]
#         # k = k[0]
#         # tau = tau[0]
#         # lmda_r = lmda_r[0]

#     data = load_and_process_data("habitatSuitability_scaled_20.tif")
#     environment_c = c_ndarray(data, dtype=np.double, ndim=2)
#     lib = load_shared_library('./runsim.so')

#     # TODO! HARDCODED ONE FOR NOW
#     ind_data = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_data.csv",
#                     header=None,
#                     names=["timestamp", "xObserved", "yObserved",
#                             "stepDistanceObserved", "tuningAngleObserved",
#                             "habitatSuitabilityObserved", "rsc"])
#     ind_data_start_time = 0
#     with open("individual_data/individual_data/calibration_2011VA0533_start_time.txt") as f:
#         lines = f.readlines()
#         ind_data_start_time = float(lines[0].strip())

#     ind_data_start_day = 0
#     with open("individual_data/individual_data/calibration_2011VA0533_start_day.txt") as f:
#         lines = f.readlines()
#         ind_data_start_day = int(lines[0].strip())

#     # ind_data_start_time = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_start_time.txt")
#     # ind_data_start_day = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_start_day.txt")
#     ind_data_centroid = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_centroid.csv", header=None)
#     ind_data = np.array(ind_data)
#     times = np.array(ind_data[:, 0], dtype='uintc')
#     # str_timestamp = [str(int(time_val)) for time_val in ind_data[:, 0]]
#     # str_timestamp = ' '.join(str_timestamp)

#     s_time = ind_data_start_time
#     s_day = int(ind_data_start_day)
#     sol_lat = ind_data_centroid.values[0][0]
#     sol_long = ind_data_centroid.values[1][0]

#     upper_left_left = 3436195  # TODO
#     upper_left_top = 5474136  # TODO

#     random_state = random_state or np.random

#     # if 'seed' not in kwinputs:
#     #     seed =  rng.integers(low=0, high=1e+9)  # TODO? want bigger...
#     #     kwinputs['seed'] = seed

#     # times = str_timestamp
#     times_c = c_ndarray(times, dtype=np.uintc, ndim=1)
#     env_res = 20  # TODO: FLEXIBLE
#     tic = time.time()

#     # with open(filename, 'w') as f, redirect_stdout(f):
#     func = lib.run_sim_wrapper
#     # environment_dummy = np.zeros((100, 100))
#     # environment_dummy_c = c_ndarray(environment_dummy, dtype=np.double, ndim=2)
#     func.restype = ctypes.POINTER(ctypes.c_double * 2048)
#     res_all = []
#     for i in range(batch_size):
#         if hasattr(random_state, 'integers'):
#             seed = random_state.integers(low=0, high=1e+9)
#         else:
#             seed = random_state.randint(0, 1e+9)
#         # TODO! HARDCODED OBSERVATION ERROR FROM AccMetric R - LOSIM/code/calibrationScript.R
#         observation_error = 2.4
#         res = lib.run_sim_wrapper(ctypes.c_double(rho), ctypes.c_double(k), ctypes.c_double(tau),
#                         ctypes.c_double(lmda_r), env_res, ctypes.c_double(s_time), s_day,
#                         ctypes.c_double(sol_lat), ctypes.c_double(sol_long),
#                         ctypes.c_double(observation_error),
#                         upper_left_left, upper_left_top,
#                         ctypes.c_ulonglong(seed), times_c,
#                         environment_c)
#         try:
#             np_res = np.array([el for el in res.contents])
#         except ValueError as e:
#             np_res = np.ones(2048)
#         np_res = np_res[:(7 * len(times))]  # trim down from 2048 to actual data
#         np_res = np_res.reshape((-1, 7))
#         res_all.append(np_res)
#     toc = time.time()
#     # print("time: ", toc - tic)

#     return res_all


# def sim_fn(rho, k, tau, lmda_r, batch_size=1, meta=None, random_state=None):
#     # sims = []
#     # # TODO! BAD TEMP ... PARALLELISE!
#     # for i in range(batch_size):
#     #     try:
#     res = OWL(rho[0], k[0], tau[0], lmda_r[0], meta=meta, random_state=random_state)
#         # except Exception as e:
#         #     print(e)
#         # sims.append(res)
#     return res


def summary_stats(*x):
    if np.allclose(x, 1):
        return -1.0 * np.ones(31)
    x = np.squeeze(np.array(x)).reshape((-1, 7))
    timestamp = x[:, 0]
    x_observed = x[:, 1]
    y_observed = x[:, 2]
    step_distance_observed = x[:, 3]
    turning_angle_observed = x[:, 4]
    habitat_suitability_observed = x[:, 5]
    rsc = x[:, 6]

    ssx = np.array([])  # TODO: magic

    # ss1 = np.quantile(habitat_suitability_observed, [.1, .9])
    # ssx[0] = ss1[0]  # TODO: moderate neg skew - not fixed
    # ssx[1] = ss1[1] # TODO! REMOVE? -> fairly arbitrary transform... good enough??
    # ss2 = np.log(np.std(habitat_suitability_observed))  # TODO: ADDED LOG TRANSFORM STILL LONG TAIL...
    ss3 = np.mean(habitat_suitability_observed)  # TODO? maybe too skewed shape?
    # ssx = np.append(ssx, ss2)
    ssx = np.append(ssx, ss3)
    # ssx[2] = ss2
    # ssx[3] = ss3
    num_samples = 1000  # TODO! Adjusting ... originally 5000(?)
    N = len(timestamp)
    from_samples = np.random.choice(N, num_samples, replace=True)
    to_samples = np.random.choice(N, num_samples, replace=True)
    delta_t = np.abs(timestamp[from_samples] - timestamp[to_samples])
    dist_cum = np.zeros(num_samples)
    for i in range(num_samples):
        if from_samples[i] < to_samples[i]:
            idx = np.arange(from_samples[i], to_samples[i]+1)
        else:
            idx = np.arange(from_samples[i], to_samples[i]-1, -1)
        dist_cum[i] = np.nansum(step_distance_observed[idx])
    dist_cum_sq = dist_cum ** 2
    dist_dir = np.sqrt((x_observed[to_samples] - x_observed[from_samples])**2 +
                    (y_observed[to_samples] - y_observed[from_samples])**2)
    X = np.column_stack((dist_cum, dist_cum_sq, dist_dir))
    lm = LinearRegression().fit(X, delta_t)
    # ss4_intercept = lm.intercept_
    # ss4_coeffs = lm.coef_
    # ss4 = np.array([lm.intercept_, lm.coef_])
    # ssx[4] = lm.intercept_
    # ssx[5] = lm.coef_[0]
    # ssx[6] = lm.coef_[1]
    # ssx[7] = lm.coef_[2]

    ssx = np.append(ssx, lm.intercept_)
    ssx = np.append(ssx, lm.coef_[0])
    ssx = np.append(ssx, lm.coef_[1])
    ssx = np.append(ssx, lm.coef_[2])  # TODO: TRANSFORM IF KEEP

    # points less than 30 min apart
    time_diff = np.diff(timestamp)
    short_steps = np.where(time_diff < 30)[0] + 1

    # distances travelled in short time intervals
    # TODO: INVESTIGATE DIFFERENCES - ss9-11
    short_dist = step_distance_observed[short_steps]
    # ss5 = np.nanstd(short_dist)  # TODO? slight diff - pretty rough summary distribution...
    ss6 = np.nanmean(short_dist)
    ss7 = np.nanquantile(short_dist, [.1, .9])

    # TODO: WHY DIFFERENCES IN SS?
    # ssx[8] = ss5
    ssx = np.append(ssx, ss6)
    ssx = np.append(ssx, ss7[0])
    ssx = np.append(ssx, ss7[1])

    # angles during short time intervals
    short_angle = turning_angle_observed[short_steps]
    ss8 = np.nanstd(short_angle)  # TODO? slight diff
    ss9 = np.nanmean(short_angle)
    ss10 = np.nanquantile(short_angle, [.1, .2, .8, .9])

    ssx = np.append(ssx, ss8)
    ssx = np.append(ssx, ss9)
    ssx = np.append(ssx, ss10[0])
    ssx = np.append(ssx, ss10[1])
    ssx = np.append(ssx, ss10[2])  # TODO?: summary distribution
    ssx = np.append(ssx, ss10[3])  # TODO?: summary distribution


    # start-end distance
    ss11 = np.sqrt((x_observed[-1] - x_observed[0]) ** 2 +
                   (y_observed[-1] - y_observed[0]) ** 2)

    # NOTE: S_15
    ssx = np.append(ssx, ss11)  # TODO? summary distribution

    # max effort
    quantiles = [.1, .9]
    ss12 = np.array([])  # TODO? slight different results... maybe solver?
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile, alpha=0, solver='highs')

        qrm = qr.fit(delta_t.reshape(-1, 1), dist_cum)
        ss12 = np.concatenate((ss12, np.array([qrm.intercept_]), qrm.coef_))
        # TODO! CHECK AGAIN ON S17 ... ie. .9 the intercept... changed alpha
    
    ss12[2]  = np.log(ss12[2])
    # TODO?: DIFFERENT SCALE AGAIN
    # TODO? S_18 log...
    ssx = np.append(ssx, ss12)

    # TODO: VERY DIFFERENT SCALES AGAIN...
    # distance cumulative moments
    ss13 = np.nanquantile(dist_cum, [.1, .9])
    ss14 = np.nanquantile(dist_dir, [.1, .9])

    ssx = np.append(ssx, ss13)
    ssx = np.append(ssx, ss14)

    # NOTE: added log transform
    ss15 = np.log(skew(dist_cum, nan_policy='omit'))  # TODO: summary shape
    ss16 = skew(dist_dir, nan_policy='omit')  # TODO: summary shape

    ssx = np.append(ssx, ss15)
    ssx = np.append(ssx, ss16)

    # extent of movement for max effort
    # NOTE: added log transform
    # ss17 = np.log(max(x_observed) - min(x_observed)) * (max(y_observed) - min(y_observed))

    # NOTE: S_26
    # ssx = np.append(ssx, ss17)  # TODO: SUMMARY DISTRIBUTION

    # cumulative step distance
    ss18 = np.nansum(step_distance_observed)
    ss19 = np.nansum(dist_cum)
    ss20 = np.log(np.nansum(dist_dir))  # TODO: ADDED LOG TRANSFORM

    # TODO: different scales...
    ssx = np.append(ssx, ss18)
    ssx = np.append(ssx, ss19)
    ssx = np.append(ssx, ss20)  # TODO?: maybe bad distribution

    # convex hull area
    X = np.column_stack((x_observed, y_observed))
    hull = ConvexHull(points=X)
    ss21 = hull.volume

    ssx = np.append(ssx, ss21)  # TODO? maybe bad distribution

    # cluster in distance matrix
    dd = np.zeros((N, N))
    # TODO? for loops probably slow...
    for i in range(N):
        for j in range(N):
            # x1 = x_observed[i]
            # y1 = y_observed[i]
            x = np.array([x_observed[i], y_observed[i]])
            y = np.array([x_observed[j], y_observed[j]])
            dd[i,j] = np.sqrt(np.sum((x - y) ** 2))
            # dd[i, j] = np.linalg.norm(x, y)

    # dd = pdist(X)
    hist = np.histogram(dd, bins=int(N/2))
    ss22 = np.std(hist[0])
    # number of peaks in the histogram
    # ss23 = np.sum(np.diff(np.sign(np.diff(hist[0]))) == -2)

    ssx = np.append(ssx, ss22)

    # NOTE: S_32 here
    # ssx = np.append(ssx, ss23)  # TODO: QUESTIONABLE SUMMARY DISTRIBUTION

    # quant regression...
    # TODO! REMOVED ! TIME CONSUMING AND CAUSED ISSUES... CONSIDER BACK IN LATER
    # quantiles = [.1, .8, .9]
    # ss24 = np.array([])  # TODO: some different results... maybe solver?
    # for quantile in quantiles:
    #     # TODO? check alpha = 0 vs 1
    #     qr = QuantileRegressor(quantile=quantile, alpha=0, solver='highs')
    #     qrm = qr.fit(dist_dir.reshape(-1, 1), delta_t)
    #     ss24 = np.concatenate((ss24, np.array([qrm.intercept_]), np.log(qrm.coef_)))  # NOTE: add log trans
    #     # TODO! - MISTAKENLY USED ss12 for same summary...

    # # TODO: VERY DIFFERENT AGAIN
    # ssx = np.append(ssx, ss24)  # TODO: re-added .9 quantile for testing
    # ssx[40] = ss24[4]  # TODO: REMOVED CAUSED VERY BAD SHAPE
    # ssx[41] = ss24[5]

    # skewness of habitat suitability
    # ss25 = skew(habitat_suitability_observed, nan_policy='omit')
    # ssx[42] = ss25  # TODO: REMOVED BAD DISTRIBUTION

    # resource selection function
    ss26 = np.nanmean(rsc)
    # ss27 = np.nanstd(rsc)  # TODO: BAD DISTRIBUTION - LOG?
    # ss28 = np.nanquantile(rsc, [.1, .2, .8, .9])

    ssx = np.append(ssx, ss26)
    # ssx = np.append(ssx, ss27)
    # ssx[45] = ss28[0]
    # ssx[46] = ss28[1]
    # ssx[47] = ss28[2]
    # ssx[48] = ss28[3]

    return ssx


def summary_stats_batch(sims):
    # columns 0 - timestamp, 1 - xObserved, 2 - yObserved,
    # 3 - stepDistanceObserved, 4 - turningAngleObserved,
    # 5 - habitatSuitabilityObserved, 6 - rsc
    batch_size = len(sims)
    ssx_all = np.zeros((batch_size, 31))  # TODO: BAD MAGIC NUMBER

    for ii, x in enumerate(sims):

        ssx_all[ii, :] = summary_stats(x)

    return ssx_all


def simulation_function(*inputs, **kwinputs):
    inputs, kwinputs = prepare_inputs(*inputs, **kwinputs)
    res = invoke_simulation(*inputs, **kwinputs)
    res = process_result(res, *inputs, **kwinputs)
    return res


def get_model(true_params=None, seed_obs=None, upper_left=None, observed=False):
    """Return the model in ..."""
    # NOTE: default params arbitrarily chosen
    rho = 2
    k = 1.5
    tau = 1
    lmda_r = 5
    if true_params is None:
        true_params = [k, lmda_r, rho, tau]

    # sim_fn = owl
    # summary_fn = summary_stats_parallel

    data = load_and_process_data("habitatSuitability_scaled_20.tif")

    sim_fn = simulation_function
    sim_fn = partial(sim_fn, environment_c=data)
    summary_fn = summary_stats_batch

    if observed:
        y = pd.read_csv("individual_data/individual_data/calibration_2011VA0533_data.csv",
                        header=None,
                        names=["timestamp", "xObserved", "yObserved",
                               "stepDistanceObserved", "tuningAngleObserved",
                               "habitatSuitabilityObserved", "rsc"])
        y = [np.array(y)]

    else:
        y = sim_fn(*true_params, random_state=np.random.RandomState(seed_obs))
        y = [y]

    m = elfi.ElfiModel(name='owl')
    elfi.Prior('uniform', 0.1, 3.9, model=m, name='k')
    elfi.Prior('uniform', 0.01, 29.99, model=m, name='lmda_r')
    elfi.Prior('uniform', 0.1, 4.9, model=m, name='rho')
    elfi.Prior('uniform', 0.1, 3.4, model=m, name='tau')

    # Simulator
    elfi.Simulator(sim_fn, m['k'], m['lmda_r'], m['rho'], m['tau'],
                   observed=y,
                   name='OWL')
    # m['OWL'].uses_meta = True

    # Summary
    elfi.Summary(summary_fn, m['OWL'], name='S')

    # Synthetic Likelihood
    # elfi.SyntheticLikelihood('semibsl', m['S'], name='SL')
    elfi.Distance('euclidean', m['S'], name='d')
    elfi.AdaptiveDistance('euclidean', m['S'], name='d_adapt')
    return m
