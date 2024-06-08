import numpy as np
from elfi.examples import owl
from elfi.examples.owl import summary_stats

def run_test():
    np.random.seed(123)
    m = owl.get_model(seed_obs=123, observed=False)    

    observed_summary = summary_stats(m.observed['OWL'])
    for ii, obs_ii in enumerate(observed_summary):
        print(f"Summary {ii} Value: {obs_ii}")

if __name__ == '__main__':
    run_test()
