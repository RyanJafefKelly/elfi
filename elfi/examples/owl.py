"""Little owls..."""

import ctypes
import uuid
from functools import partial

import numpy as np
import pandas as pd
import rasterio
import copy
import pickle as pkl
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.stats import skew
# from numpyctypes import c_ndarray
from sklearn.linear_model import LinearRegression, QuantileRegressor
import time

import elfi
from numpyctypes import c_ndarray


def load_and_process_data(filepath):
    """read habitat infos"""
    data = np.zeros((15092, 16730))
    i = 0
    with rasterio.open(filepath) as src:
        # print(src.profile)
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
    """Prepare the inputs for the simulator."""
    rho, k, tau, lmda_r = inputs
    if 'meta' in kwinputs:
        meta = kwinputs['meta']
        filename = '{model_name}_{batch_index}_{submission_index}.txt'.format(**meta)
    else:
        filename = str(uuid.uuid4()) + '.txt'

    try:
        random_state = kwinputs['random_state']
        seed = random_state.integers(low=0, high=1e+9)
    except AttributeError:
        random_state = np.random
        seed = random_state.randint(1e+9)

    kwinputs['seed'] = seed
    individual_info = kwinputs['individual_info']

    # TODO HARDCODED INDIVIDUAL FOR NOW ... COULD DO AMORTISED / MULTIPLE RUNS
    ind_data = pd.read_csv(f"individual_data/individual_data/calibration_{individual_info}_data.csv",
                    header=None,
                    names=["timestamp", "xObserved", "yObserved",
                            "stepDistanceObserved", "tuningAngleObserved",
                            "habitatSuitabilityObserved", "rsc"])
    ind_data_start_time = 0
    with open(f"individual_data/individual_data/calibration_{individual_info}_start_time.txt") as f:
        lines = f.readlines()
        ind_data_start_time = float(lines[0].strip())

    ind_data_start_day = 0
    with open(f"individual_data/individual_data/calibration_{individual_info}_start_day.txt") as f:
        lines = f.readlines()
        ind_data_start_day = int(lines[0].strip())

    ind_data_centroid = pd.read_csv(f"individual_data/individual_data/calibration_{individual_info}_centroid.csv", header=None)
    ind_data = np.array(ind_data)

    times = np.array(ind_data[:, 0], dtype='uintc')

    kwinputs['s_time'] = ind_data_start_time
    kwinputs['s_day'] = ind_data_start_day
    kwinputs['sol_lat'] = ind_data_centroid.values[0][0]
    kwinputs['sol_long'] = ind_data_centroid.values[1][0]

    kwinputs['upper_left_left'] = 3436195  # TODO?
    kwinputs['upper_left_top'] = 5474136  # TODO?

    kwinputs['observation_error'] = 2.4  # TODO HARDCODED OBSERVATION ERROR FROM AccMetric R - LOSIM/code/calibrationScript.R
    kwinputs['environment_c'] = c_ndarray(kwinputs['environment_c'], dtype=np.double, ndim=2)

    kwinputs['times'] = times

    kwinputs['filename'] = filename
    kwinputs['output_filename'] = filename[:-4] + '_out.txt'

    kwinputs['batch_size'] = 1 if 'batch_size' not in kwinputs else kwinputs['batch_size']

    return inputs, kwinputs

def invoke_simulation(*inputs, **kwinputs):
    """Invoke the owl simulation."""
    k, lmda_r, rho, tau = inputs
    if hasattr(rho, '__len__') and len(rho) > 1:
        rho = rho[0]
        k = k[0]
        tau = tau[0]
        lmda_r = lmda_r[0]
    else:  # assumes something array like passed
        pass

    lib = ctypes.cdll.LoadLibrary('./librunsim.dylib')  # TODO: set .so or .dylib depending on mac vs HPC
    lib.run_sim_wrapper.restype = ctypes.c_int

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
    batch_size = kwinputs['batch_size']
    random_state = np.random

    # print(', '.join(['{}={!r}'.format(k, v) for k, v in kwinputs.items()]))

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

        _ = lib.run_sim_wrapper(ctypes.c_double(rho), ctypes.c_double(k), ctypes.c_double(tau),
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
        np_res = copy.copy(np_res.reshape((-1, 7)))
        # print("np_res ", np_res)
        res_all.append(np_res)

    return res_all


def _prep_x(x):
    if np.allclose(x, 1):  # Something gone wrong...
        return -1.0 * np.ones(31)
    x = np.squeeze(np.array(x)).reshape((-1, 7))
    return x


def summary_stats_batch(sims):
    batch_size = len(sims)
    ssx_all = np.zeros((batch_size, 42))
    for ii, sim in enumerate(sims):
        ssx = []
        ssx.append(habitat_suitability_summaries(sim))
        ssx.append(linear_reg_summaries(sim))
        ssx.append(short_step_summaries(sim))
        ssx.append(start_end_distance(sim))
        ssx.append(cumulative_distance_summaries(sim))
        ssx.append(direct_distance_summaries(sim))
        ssx.append(convex_hull_summary(sim))
        ssx.append(histogram_summaries(sim))
        ssx.append(rsc_summaries(sim))
        ssx_all[ii, :] = np.concatenate(ssx)

    # set any not finite values to -1e+6
    ssx_all[~np.isfinite(ssx_all)] = -1e+6
    return ssx_all

def habitat_suitability_summaries_batch(sims):
    batch_size = len(sims)
    ssx = np.zeros((batch_size, 4))
    for ii, sim in enumerate(sims):
        ssx[ii, :] = habitat_suitability_summaries(sim)

    return ssx


def linear_reg_summaries_batch(sims):
    batch_size = len(sims)
    ssx = np.zeros((batch_size, 4))
    for ii, sim in enumerate(sims):
        ssx[ii, :] = linear_reg_summaries(sim)

    return ssx

def short_step_summaries_batch(sims):
    batch_size = len(sims)
    ssx = np.zeros((batch_size, 10))
    for ii, sim in enumerate(sims):
        ssx[ii, :] = short_step_summaries(sim)

    return ssx

def cumulative_distance_summaries_batch(sims):
    batch_size = len(sims)
    ssx = np.zeros((batch_size, 9))
    for ii, sim in enumerate(sims):
        ssx[ii, :] = cumulative_distance_summaries(sim)

    return ssx

def direct_distance_summaries_batch(sims):
    batch_size = len(sims)
    ssx = np.zeros((batch_size, 4))
    for ii, sim in enumerate(sims):
        ssx[ii, :] = direct_distance_summaries(sim)

    return ssx

def histogram_summaries_batch(sims):
    batch_size = len(sims)
    ssx = np.zeros((batch_size, 2))
    for ii, sim in enumerate(sims):
        ssx[ii, :] = histogram_summaries(sim)

    return ssx

def rsc_summaries_batch(sims):
    batch_size = len(sims)
    ssx = np.zeros((batch_size, 7))
    for ii, sim in enumerate(sims):
        ssx[ii, :] = rsc_summaries(sim)

    return ssx

def convex_hull_summary_batch(sims):
    batch_size = len(sims)
    ssx = np.zeros((batch_size, 1))
    for ii, sim in enumerate(sims):
        ssx[ii, :] = convex_hull_summary(sim)

    return ssx

def start_end_distance_batch(sims):
    batch_size = len(sims)
    ssx = np.zeros((batch_size, 1))
    for ii, sim in enumerate(sims):
        ssx[ii, :] = start_end_distance(sim)

    return ssx


def habitat_suitability_summaries(*x):
    """Corresponds to S1-4."""
    x = _prep_x(x)
    habitat_suitability_observed = x[:, 5]
    hs_quantiles = np.quantile(habitat_suitability_observed, [.1, .9])
    hs_std = np.log(np.std(habitat_suitability_observed))
    hs_mean = np.mean(habitat_suitability_observed)

    return np.array([hs_quantiles[0], hs_quantiles[1], hs_std, hs_mean])


def linear_reg_summaries(*x):
    """Corresponds to S5-8."""
    x = _prep_x(x)
    timestamp = x[:, 0]
    x_observed = x[:, 1]
    y_observed = x[:, 2]
    step_distance_observed = x[:, 3]

    num_samples = 1000  # TODO: 5000 originally ... get away with less?
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

    return np.array([lm.intercept_, lm.coef_[0], lm.coef_[1], lm.coef_[2]])


def short_step_summaries(*x):
    """Corresponds to S9-18."""
    x = _prep_x(x)
    timestamp = x[:, 0]
    step_distance_observed = x[:, 3]
    turning_angle_observed = x[:, 4]

    time_diff = np.diff(timestamp)
    short_steps = np.where(time_diff < 30)[0] + 1

    short_dist = step_distance_observed[short_steps]
    short_dist_std = np.nanstd(short_dist)
    short_dist_mean = np.nanmean(short_dist)
    short_dist_quantiles = np.nanquantile(short_dist, [.1, .9])

    # angles during short time intervals
    short_angle = turning_angle_observed[short_steps]
    short_angle_std = np.nanstd(short_angle)  # TODO? slight diff
    short_angle_mean = np.nanmean(short_angle)
    short_angle_quantiles = np.nanquantile(short_angle, [.1, .2, .8, .9])

    return np.array([short_dist_std, short_dist_mean,
                     short_dist_quantiles[0], short_dist_quantiles[1],
                     short_angle_std, short_angle_mean,
                     short_angle_quantiles[0], short_angle_quantiles[1],
                     short_angle_quantiles[2], short_angle_quantiles[3]])


def start_end_distance(*x):
    """S19"""
    x = _prep_x(x)
    x_observed = x[:, 1]
    y_observed = x[:, 2]

    start_end = np.sqrt((x_observed[-1] - x_observed[0]) ** 2 +
                        (y_observed[-1] - y_observed[0]) ** 2)
    return np.array([np.log(start_end)])


def cumulative_distance_summaries(*x):
    """S20-25, S28, S32"""
    x = _prep_x(x)
    timestamp = x[:, 0]
    step_distance_observed = x[:, 3]

    num_samples = 1000  # TODO! 5000 as original ... or less?
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

    quantiles = [.1, .9]
    ss12 = np.array([])  # TODO? slight different results... maybe solver?
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile, alpha=0, solver='highs')

        qrm = qr.fit(delta_t.reshape(-1, 1), dist_cum)
        ss12 = np.concatenate((ss12, np.array([qrm.intercept_]), qrm.coef_))
        # TODO! CHECK AGAIN ON S17 ... ie. .9 the intercept... changed alpha
    ss12[2]  = np.log(ss12[2])

    cum_dist_quantiles = np.nanquantile(dist_cum, [.1, .9])

    cum_dist_skewness = np.log(skew(dist_cum, nan_policy='omit'))  # TODO: summary shape
    if not np.isfinite(cum_dist_skewness):
        cum_dist_skewness = -1e+6

    cum_dist_sum = np.nansum(dist_cum)

    return np.array([*ss12, cum_dist_sum,
                     cum_dist_quantiles[0], cum_dist_quantiles[1],
                     cum_dist_skewness, cum_dist_sum])


def direct_distance_summaries(*x):
    """SS26, 27, 29, 33"""
    x = _prep_x(x)
    timestamp = x[:, 0]
    x_observed = x[:, 1]
    y_observed = x[:, 2]

    num_samples = 1000  # TODO! 5000 as original ... or less?
    N = len(timestamp)
    from_samples = np.random.choice(N, num_samples, replace=True)
    to_samples = np.random.choice(N, num_samples, replace=True)

    dist_dir = np.sqrt((x_observed[to_samples] - x_observed[from_samples])**2 +
                       (y_observed[to_samples] - y_observed[from_samples])**2)

    ss14 = np.log(np.nanquantile(dist_dir, [.1, .9]) + 1e-6)  # add small value to avoid log(0)

    ss16 = skew(dist_dir, nan_policy='omit')  # TODO: summary shape

    ss20 = np.log(np.nansum(dist_dir))  # TODO: ADDED LOG TRANSFORM

    # quantiles = [.1, .8, .9]
    # ss24 = np.array([])  # TODO: some different results... maybe solver?
    # for quantile in quantiles:
    #     # TODO? check alpha = 0 vs 1
    #     qr = QuantileRegressor(quantile=quantile, alpha=0, solver='highs')
    #     qrm = qr.fit(dist_dir.reshape(-1, 1), delta_t)
    #     ss24 = np.concatenate((ss24, np.array([qrm.intercept_]), np.log(qrm.coef_)))  # NOTE: add log trans
    #     # TODO! - MISTAKENLY USED ss12 for same summary...


    return np.array([ss14[0], ss14[1], ss16, ss20])


def histogram_summaries(*x):
    """S35, 36"""
    x = _prep_x(x)
    timestamp = x[:, 0]
    x_observed = x[:, 1]
    y_observed = x[:, 2]
    N = len(timestamp)
    dd = np.zeros((N, N))
    # TODO? for loops probably slow...
    for i in range(N):
        for j in range(N):
            # x1 = x_observed[i]
            # y1 = y_observed[i]
            x = np.array([x_observed[i], y_observed[i]])
            y = np.array([x_observed[j], y_observed[j]])
            dd[i, j] = np.sqrt(np.sum((x - y) ** 2))

    hist = np.histogram(dd, bins=int(N/2))
    bin_counts_stdev = np.std(hist[0])
    # number of peaks in the histogram
    num_peaks = np.sum(np.diff(np.sign(np.diff(hist[0]))) == -2)

    return np.array([bin_counts_stdev, num_peaks])


def rsc_summaries(*x):
    "S43-49"  # TODO! RECHECK ADDITIONAL SUMMARIES
    x = _prep_x(x)
    habitat_suitability_observed = x[:, 5]
    rsc = x[:, 6]

    # skewness of habitat suitability
    hs_skew = skew(habitat_suitability_observed, nan_policy='omit')

    # resource selection function
    rsc_mean = np.log(np.nanmean(rsc))
    rsc_std = np.log(np.nanstd(rsc))
    rsc_quantiles = np.log(np.nanquantile(rsc, [.1, .2, .8, .9]))
    if not np.isfinite(rsc_quantiles[0]):
        rsc_quantiles[0] = -1e+6

    return np.array([hs_skew, rsc_mean, rsc_std,
                     rsc_quantiles[0], rsc_quantiles[1],
                     rsc_quantiles[2], rsc_quantiles[3]])


def convex_hull_summary(*x):
    """S34"""
    x = _prep_x(x)
    x_observed = x[:, 1]
    y_observed = x[:, 2]
    # convex hull area
    X = np.column_stack((x_observed, y_observed))
    try:
        hull = ConvexHull(points=X)
        ss21 = hull.volume
    except Exception as e:
        print("Hull error: ", e)
        ss21 = -1e+6

    return np.array([np.log(ss21)])


def simulation_function(*inputs, **kwinputs):
    inputs, kwinputs = prepare_inputs(*inputs, **kwinputs)
    res = invoke_simulation(*inputs, **kwinputs)
    return res


def get_model(true_params=None, seed_obs=None, upper_left=None,
              observed=False, individual_info=None):
    """Return the model in ..."""
    # NOTE: default params arbitrarily chosen
    rho = 2.0
    # rho = 1.0
    k = 1.5
    tau = 1.0
    lmda_r = 5.0
    if true_params is None:
        true_params = [k, lmda_r, rho, tau]

    # sim_fn = owl
    # summary_fn = summary_stats_parallel

    data = load_and_process_data("habitatSuitability_scaled_20.tif")


    if individual_info is None:
        individual_info = "2011VA0533"

    sim_fn = simulation_function
    sim_fn = partial(sim_fn, environment_c=data, individual_info=individual_info)

    if observed:
        ind_filename = "individual_data/individual_data/calibration_" + \
            individual_info + "_data.csv"
        y = pd.read_csv(ind_filename,
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
    summ = elfi.Summary(summary_stats_batch, m['OWL'], name='S')
    # sumstats = []

    # sumstats.append(elfi.Summary(habitat_suitability_summaries_batch, m['OWL'], name='hs_S'))
    # sumstats.append(elfi.Summary(linear_reg_summaries_batch, m['OWL'], name='lr_S'))
    # sumstats.append(elfi.Summary(short_step_summaries_batch, m['OWL'], name='short_step_S'))
    # sumstats.append(elfi.Summary(start_end_distance_batch, m['OWL'], name='start_end_S'))
    # sumstats.append(elfi.Summary(cumulative_distance_summaries_batch, m['OWL'], name='cum_dist_S'))
    # sumstats.append(elfi.Summary(direct_distance_summaries_batch, m['OWL'], name='dir_dist_S'))
    # sumstats.append(elfi.Summary(convex_hull_summary_batch, m['OWL'], name='convex_hull_S'))
    # sumstats.append(elfi.Summary(histogram_summaries_batch, m['OWL'], name='hist_S'))
    # sumstats.append(elfi.Summary(rsc_summaries_batch, m['OWL'], name='rsc_S'))

    # with open("owl_inv_cov_mat.pkl", "rb") as f:
    #     VI = pkl.load(f)

    # d = elfi.Distance('euclidean', m['S'], name='d')
    # elfi.Distance('mahalanobis', m['S'], name='d', VI=VI)
    elfi.AdaptiveDistance(summ, name='d_adapt')
    return m
