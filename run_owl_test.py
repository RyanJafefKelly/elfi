import numpy as np
from elfi.examples import owl
from elfi.examples.owl import summary_stats


def run_test():
    np.random.seed(123)
    rho = 2.0  #  Habitat-specialization level: weight defining the importance of habitat suitability in the selection of a new location
    k = 3.5  #  Flight-distance shape: shape parameter in a gamma distribution describing the flight-distance preference	
    tau = 1.0  # Directional persistence: inverse standard deviation in a Gaussian distribution (with μ
    lmda_r = 5.0  # Average roosting duration in day

    test_params = [k, lmda_r, rho, tau]

    m = owl.get_model(true_params=test_params, seed_obs=123, observed=False)

    observed_summary = summary_stats(m.observed['OWL'])
    for ii, obs_ii in enumerate(observed_summary):
        print(f"Summary {ii} Value: {obs_ii}")


if __name__ == '__main__':
    run_test()
