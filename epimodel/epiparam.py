"""
:code:`epi_params.py`

Calculate delay distributions and generate delay parameter dictionaries for region building.

Mostly copied from https://github.com/epidemics/COVIDNPIs/blob/manuscript/epimodel/pymc3_models/epi_params.py
"""

import pprint

import numpy as np


class EpidemiologicalParameters:
    """
    Epidemiological Parameters Class
    Wrapper Class, contains information about the epidemiological parameters used in this project.
    """

    def __init__(
        self,
        generation_interval=None,
        incubation_period=None,
        onset_to_death_delay=None,
        onset_to_case_delay=None,
        gi_truncation=28,
        cd_truncation=32,
        dd_truncation=64,
    ):
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
        :param onset_to_case_delay dictionary containing relevant distribution information
        :param onset_to_death_delay: dictionary containing relevant distribution information
        """
        if generation_interval is not None:
            self.generation_interval = generation_interval
        else:
            self.generation_interval = {
                "mean": 4.83,  # 4.31 to 5.4
                "sd": 1.73,
                "dist": "gamma",
            }

        if incubation_period is not None:
            self.incubation_period = incubation_period
        else:
            self.incubation_period = {"mean": 5.53, "sd": 4.73, "dist": "gamma"}

        if onset_to_case_delay is not None:
            self.onset_to_case_delay = onset_to_case_delay
        else:
            self.onset_to_case_delay = {"mean": 5.2775, "sd": 3.7466, "dist": "gamma"}

        if onset_to_death_delay is not None:
            self.onset_to_death_delay = onset_to_death_delay
        else:
            self.onset_to_death_delay = {"mean": 18.6063, "sd": 13.618, "dist": "gamma"}

        self.gi_truncation = gi_truncation
        self.cd_truncation = cd_truncation
        self.dd_truncation = dd_truncation

        self.generate_delays()

    def generate_delays(self, nRv=int(1e7)):
        self.GIv = self.generate_dist_vector(
            self.generation_interval, nRv, self.gi_truncation
        )
        self.GI_projmat = np.zeros((self.GIv.size - 1, self.GIv.size - 1))
        for i in range(self.GIv.size - 2):
            self.GI_projmat[i + 1, i] = 1
        self.GI_projmat[:, -1] = self.GIv[:, ::-1][:, :-1]
        self.GI_flat_rev = self.GIv[:, 1:][:, ::-1].flatten()

        self.DPC = self.generate_dist_vector(
            [self.incubation_period, self.onset_to_case_delay],
            int(1e7),
            self.cd_truncation,
        )
        self.DPD = self.generate_dist_vector(
            [self.incubation_period, self.onset_to_death_delay],
            int(1e7),
            self.dd_truncation,
        )

    def generate_dist_samples(self, dist, nRVs):
        """
        Generate samples from given distribution.
        :param dist: Distribution dictionary to use.
        :param nRVs: number of random variables to sample
        :param with_noise: if true, add noise to distributions, else do not.
        :return: samples
        """
        # specify seed because everything here is random!!
        mean = dist["mean"]
        if dist["dist"] == "gamma":
            sd = dist["sd"]
            k = mean ** 2 / sd ** 2
            theta = sd ** 2 / mean
            samples = np.random.gamma(k, theta, size=nRVs)
        elif dist["dist"] == "negbinom":
            disp = dist["mean"]
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

        # print(f"Sample mean: {np.mean(samples)} Sample std: {np.std(samples)}")
        bins = np.arange(-1.0, float(max_int))
        bins[2:] += 0.5

        counts = np.histogram(samples, bins)[0]
        # normalise
        pmf = counts / np.sum(counts)
        pmf = pmf.reshape((1, pmf.size))

        # print(self.generate_pmf_statistics_str(pmf))
        return pmf

    def generate_pmf_statistics_str(self, delay_prob_full):
        """
        Make mean and variance of delay string.
        :param delay_prob: delay to compute statistics of.
        :return: Information string.
        """
        delay_prob = delay_prob_full.flatten()
        n_max = delay_prob.size
        mean = np.sum([(i) * delay_prob[i] for i in range(n_max)])
        var = np.sum([(i ** 2) * delay_prob[i] for i in range(n_max)]) - mean ** 2
        return f"mean: {mean:.3f}, sd: {var ** 0.5:.3f}, max: {n_max}"

    def generate_dist_vector(self, dist, nRVs=int(1e7), truncation=28):
        """
        Generate discretised vector describing dist. We use Monte Carlo sampling to generate this delay vector.

        :param nRVs: nRVs: int - number of random variables used for integration
        :param max_gi: int - reporting delay truncation
        :return: discretised generation interval
        """
        if isinstance(dist, dict):
            samples = self.generate_dist_samples(dist, nRVs)
        elif isinstance(dist, list):
            samples = np.zeros(nRVs)
            for d in dist:
                samples = samples + self.generate_dist_samples(d, nRVs)

        return self.discretise_samples(samples, truncation)

    def R_to_daily_growth(self, R):
        gi_beta = self.generation_interval["mean"] / self.generation_interval["sd"] ** 2
        gi_alpha = (
            self.generation_interval["mean"] ** 2 / self.generation_interval["sd"] ** 2
        )

        g = np.exp(gi_beta * (R ** (1 / gi_alpha) - 1))
        return g

    def summarise_parameters(self):
        """
        Print summary of parameters.
        """
        print(
            "Epidemiological Parameters Summary\n"
            "----------------------------------\n"
        )
        print("Generation Interval")
        pprint.pprint(self.generation_interval)
        print("Infection to Reporting Delay")
        pprint.pprint(self.infection_to_reporting_delay)
        print("Infection to Fatality Delay")
        pprint.pprint(self.infection_to_fatality_delay)
        print("----------------------------------\n")
