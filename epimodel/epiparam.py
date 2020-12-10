"""
:code:`epi_params.py`

Calculate delay distributions and generate delay parameter dictionaries for region building.

Mostly copied from https://github.com/epidemics/COVIDNPIs/blob/manuscript/epimodel/pymc3_models/epi_params.py
"""

import numpy as np
import pprint


class EpidemiologicalParameters():
    """
    Epidemiological Parameters Class
    Wrapper Class, contains information about the epidemiological parameters used in this project.
    """

    def __init__(self, generation_interval=None, infection_to_fatality_delay=None,
                 infection_to_reporting_delay=None):
        """
        Constructor
        Input dictionaries corresponding to the relevant delay with the following fields:
            - mean_mean: mean of the mean value
            - mean_sd: sd of the mean value
            - sd_mean: mean of the sd value
            - sd_sd: sd of the sd value
            - source: str describing source information
            - distribution type: only 'gamma' and 'lognorm' are currently supported
            - notes: any other notes
        :param numpy seed used for randomisation
        :param generation_interval: dictionary containing relevant distribution information
        :param incubation_period : dictionary containing relevant distribution information
        :param infection_to_fatality_delay: dictionaries containing relevant distribution information
        :param infection_to_reporting_delay: dictionaries containing relevant distribution information
        """
        if generation_interval is not None:
            self.generation_interval = generation_interval
        else:
            self.generation_interval = {
                'mean': 5.06,
                'sd': 2.11,
                'dist': 'gamma',
            }

        if infection_to_fatality_delay is not None:
            self.infection_to_fatality_delay = infection_to_fatality_delay
        else:
            self.infection_to_fatality_delay = {
                'mean': 21.82,
                'disp': 14.26,
                'dist': 'negbinom',
            }

        if infection_to_reporting_delay is not None:
            self.infection_to_reporting_delay = infection_to_reporting_delay
        else:
            self.infection_to_reporting_delay = {
                'mean': 10.93,
                'disp': 5.41,
                'dist': 'negbinom',
            }

        self.DPCv, self.DPDv, self.GIv = self.generate_all_delay_vectors()

        # note: the
        self.GI_projmat = np.zeros((self.GIv.size-1, self.GIv.size-1))
        for i in range(self.GIv.size-2):
            self.GI_projmat[i+1, i] = 1
        self.GI_projmat[:, -1] = self.GIv[:, ::-1][:, :-1]

    def generate_dist_samples(self, dist, nRVs):
        """
        Generate samples from given distribution.
        :param dist: Distribution dictionary to use.
        :param nRVs: number of random variables to sample
        :param with_noise: if true, add noise to distributions, else do not.
        :return: samples
        """
        # specify seed because everything here is random!!
        mean = dist['mean']
        if dist['dist'] == 'gamma':
            sd = dist['sd']
            k = mean ** 2 / sd ** 2
            theta = sd ** 2 / mean
            samples = np.random.gamma(k, theta, size=nRVs)
        elif dist['dist'] == 'negbinom':
            disp = dist['mean']
            p = disp / (disp + mean)
            samples = np.random.negative_binomial(disp, p, size=nRVs)

        return samples

    def discretise_samples(self, samples, max_int):
        """
        Discretise a set of samples to form a pmf, truncating to max.
        :param samples: Samples to discretize.
        :param max: Truncation.
        :return: pmf - discretised distribution.
        """
        bins = np.arange(-1.0, float(max_int))
        bins[2:] += 0.5

        counts = np.histogram(samples, bins)[0]
        # normalise
        pmf = counts / np.sum(counts)
        pmf = pmf.reshape((1, pmf.size))
        return pmf

    def generate_pmf_statistics_str(self, delay_prob):
        """
        Make mean and variance of delay string.
        :param delay_prob: delay to compute statistics of.
        :return: Information string.
        """
        n_max = delay_prob.size
        mean = np.sum([(i) * delay_prob[i] for i in range(n_max)])
        var = np.sum([(i ** 2) * delay_prob[i] for i in range(n_max)]) - mean ** 2
        return f'mean: {mean:.3f}, sd: {var ** 0.5:.3f}, max: {n_max}'

    def generate_dist_vector(self, dist, nRVs=int(1e7), truncation=28):
        """
        Generate discretised vector describing dist. We use Monte Carlo sampling to generate this delay vector.

        :param nRVs: nRVs: int - number of random variables used for integration
        :param max_gi: int - reporting delay truncation
        :return: discretised generation interval
        """
        samples = self.generate_dist_samples(dist, nRVs)
        return self.discretise_samples(samples, truncation)

    def generate_all_delay_vectors(self, nRVs=int(1e7), max_reporting=32, max_fatality=48, max_gi=28):
        """
        Generate reporting and fatality discretised delays using Monte Carlo sampling.

        :param nRVs: int - number of random variables used for integration
        :param max_reporting: int - reporting delay truncation
        :param max_fatality: int - death delay truncation
        :param max_gi: int - generation interval truncation
        :return: reporting_delay, fatality_delay, generation interval tuple of numpy arrays
        """

        delays = [self.infection_to_reporting_delay, self.infection_to_fatality_delay, self.generation_interval]
        truncations = [max_reporting, max_fatality, max_gi]
        delays_list = [self.generate_dist_vector(delay, nRVs, truncation) for delay, truncation in
                       zip(delays, truncations)]

        return tuple(delays_list)

    def R_to_daily_growth(self, R):
        gi_beta = self.generation_interval['mean'] / self.generation_interval['sd'] ** 2
        gi_alpha = self.generation_interval['mean'] ** 2 / self.generation_interval['sd'] ** 2

        g = np.exp(gi_beta * (R ** (1/ gi_alpha) - 1))
        return g

    def summarise_parameters(self):
        """
        Print summary of parameters.
        """
        print('Epidemiological Parameters Summary\n'
              '----------------------------------\n')
        print('Generation Interval')
        pprint.pprint(self.generation_interval)
        print('Infection to Reporting Delay')
        pprint.pprint(self.infection_to_reporting_delay)
        print('Infection to Fatality Delay')
        pprint.pprint(self.infection_to_fatality_delay)
        print('----------------------------------\n')
