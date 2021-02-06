"""
:code:`preprocessed_data.py`

PreprocessedData Class definition.
"""
import numpy as np
import pandas as pd


def set_household_limits(active_CMs, household_NPI_index, gathering_NPI_index):
    nRs, _, nDs = active_CMs.shape
    new_acms = np.copy(active_CMs)
    for r in range(nRs):
        for day in range(nDs):
            if (
                active_CMs[r, household_NPI_index, day] == 0
                or active_CMs[r, gathering_NPI_index, day]
                < active_CMs[r, household_NPI_index, day]
            ):
                new_acms[r, household_NPI_index, day] = active_CMs[
                    r, gathering_NPI_index, day
                ]
    return new_acms


def set_all_household_limits(active_CMs):
    new_acms = np.copy(active_CMs)
    new_acms = set_household_limits(new_acms, 4, 0)
    new_acms = set_household_limits(new_acms, 5, 1)
    new_acms = set_household_limits(new_acms, 6, 2)
    new_acms = set_household_limits(new_acms, 7, 3)
    return new_acms


def preprocess_data(
    data_path, last_day="2021-01-09", npi_start_col=3, skipcases=14, skipdeaths=30
):
    """
    Process data, return PreprocessedData() object

    :param data_path: path to data
    :param last_day: last day of window of analysis
    :param npi_start_col: column index (-2) of first npi
    :return: PreprocessedData() object with loaded data.
    """

    df = pd.read_csv(
        data_path, parse_dates=["Date"], infer_datetime_format=True
    ).set_index(["Area", "Date"])

    if last_day is None:
        Ds = list(df.index.levels[1])
    else:
        Ds = list(df.index.levels[1])
        last_ts = pd.to_datetime(last_day)
        Ds = Ds[: (1 + Ds.index(last_ts))]

    Rs = list(df.index.levels[0])
    CMs = list(df.columns[npi_start_col:])

    nRs = len(Rs)
    nDs = len(Ds)
    nCMs = len(CMs)

    Cs = []

    for r in Rs:
        c = df.loc[r]["Country"][0]
        Cs.append(c)

    # sort the countries by name. we need to preserve the ordering
    unique_Cs = sorted(list(set(Cs)))
    nCs = len(unique_Cs)

    C_indices = []
    for uc in unique_Cs:
        a_indices = np.nonzero([uc == c for c in Cs])[0]
        C_indices.append(a_indices)

    active_cms = np.zeros((nRs, nCMs, nDs))
    new_cases = np.ma.zeros((nRs, nDs), dtype="int")
    new_deaths = np.ma.zeros((nRs, nDs), dtype="int")

    for r_i, r in enumerate(Rs):
        r_df = df.loc[r].loc[Ds]
        new_cases.data[r_i, :] = r_df["New Cases"]
        new_deaths.data[r_i, :] = r_df["New Deaths"]

        for cm_i, cm in enumerate(CMs):
            active_cms[r_i, cm_i, :] = r_df[cm]
    # set household limits to at least as strong as gathering limits
    active_cms = set_all_household_limits(active_cms)

    # mask days where there are negative cases or deaths - because this
    # is clearly wrong
    new_cases[new_cases < 0] = np.ma.masked
    new_deaths[new_deaths < 0] = np.ma.masked
    # do this to make sure.
    new_cases.data[new_cases.data < 0] = 0
    new_deaths.data[new_deaths.data < 0] = 0

    new_cases[:, :skipcases] = np.ma.masked
    new_deaths[:, :skipdeaths] = np.ma.masked

    return PreprocessedData(
        Rs, Ds, CMs, new_cases, new_deaths, active_cms, Cs, unique_Cs, C_indices
    )


class PreprocessedData(object):
    """
    PreprocessedData Class

    Class to hold data which is subsequently passed onto a PyMC3 model. Mostly a data wrapper, with some utility
    functions.
    """

    def __init__(
        self, Rs, Ds, CMs, new_cases, new_deaths, active_cms, Cs, unique_Cs, C_indices
    ):
        """

        :param Rs:
        :param Ds:
        :param CMs:
        :param new_cases:
        :param new_deaths:
        :param active_cms:
        """
        super().__init__()
        self.Rs = Rs
        self.Ds = Ds
        self.CMs = CMs
        self.new_cases = new_cases
        self.new_deaths = new_deaths
        self.active_cms = active_cms
        self.Cs = Cs
        self.unique_Cs = unique_Cs
        self.C_indices = C_indices

        # the RC mat is used for partial pooling to grab the right country specific noise
        # for that particular region
        self.RC_mat = np.zeros((self.nRs, self.nCs))
        for r_i, c in enumerate(self.Cs):
            C_ind = self.unique_Cs.index(c)
            self.RC_mat[r_i, C_ind] = 1

    @property
    def nCMs(self):
        return len(self.CMs)

    @property
    def nRs(self):
        return len(self.Rs)

    @property
    def nCs(self):
        return len(self.unique_Cs)

    @property
    def nDs(self):
        return len(self.Ds)

    def featurize(self):
        # everything is hardcoded for now
        gathering_thresholds = [6, 30]
        household_thresholds = [2, 5]
        mask_thresholds = [3, 4]

        bin_cms = self.active_cms[:, 9:, :]
        bin_cm_names = self.CMs[9:]

        mask_cms = np.zeros((self.nRs, len(mask_thresholds), self.nDs))
        mask_cm_names = []
        gathering_cms = np.zeros((self.nRs, 4 * len(gathering_thresholds), self.nDs))
        gathering_cm_names = []
        household_cms = np.zeros((self.nRs, 4 * len(household_thresholds), self.nDs))
        household_cm_names = []

        for i in range(4):
            s_i = i * len(gathering_thresholds)
            for t_i, t in enumerate(gathering_thresholds):
                gathering_cms[:, s_i + t_i, :] = np.logical_and(
                    self.active_cms[:, i, :] > 0, self.active_cms[:, i, :] < t
                )
                gathering_cm_names.append(f"{self.CMs[i]}ed to {t}")

        for h_npi_i, i in enumerate(range(4, 8)):
            s_i = h_npi_i * len(household_thresholds)
            for t_i, t in enumerate(household_thresholds):
                household_cms[:, s_i + t_i, :] = np.logical_and(
                    self.active_cms[:, i, :] > 0, self.active_cms[:, i, :] < t
                )
                household_cm_names.append(f"{self.CMs[i]}ed to {t}")

        for mask_npi_i, t in enumerate(mask_thresholds):
            mask_cms[:, mask_npi_i, :] = self.active_cms[:, 8, :] > t
            mask_cm_names.append(f"Masks Level {t}")

        all_cm_names = [
            *bin_cm_names,
            *gathering_cm_names,
            *household_cm_names,
            *mask_cm_names,
        ]
        all_activecms = np.concatenate(
            [bin_cms, gathering_cms, household_cms, mask_cms], axis=1
        )
        self.CMs = all_cm_names
        self.active_cms = all_activecms

    def mask_region_by_index(self, region_index, nz_case_days_shown=60):
        mask_start = np.nonzero(
            np.cumsum(self.new_cases.data[region_index, :] > 0)
            == nz_case_days_shown + 1
        )[0][0]

        self.new_cases[region_index, mask_start:] = True
        self.new_deaths.mask[region_index, mask_start:] = True

        return mask_start

    def featurize(self):
        # everything is hardcoded for now
        gathering_thresholds = [6, 30]
        household_thresholds = [2, 5]
        mask_thresholds = [3, 4]

        bin_cms = self.active_cms[:, 9:, :]
        bin_cm_names = self.CMs[9:]

        mask_cms = np.zeros((self.nRs, len(mask_thresholds), self.nDs))
        mask_cm_names = []
        gathering_cms = np.zeros((self.nRs, 4 * len(gathering_thresholds), self.nDs))
        gathering_cm_names = []
        household_cms = np.zeros((self.nRs, 4 * len(household_thresholds), self.nDs))
        household_cm_names = []

        for i in range(4):
            s_i = i * len(gathering_thresholds)
            for t_i, t in enumerate(gathering_thresholds):
                gathering_cms[:, s_i + t_i, :] = np.logical_and(
                    self.active_cms[:, i, :] > 0, self.active_cms[:, i, :] < t
                )
                gathering_cm_names.append(f"{self.CMs[i]}ed to {t}")

        for h_npi_i, i in enumerate(range(4, 8)):
            s_i = h_npi_i * len(household_thresholds)
            for t_i, t in enumerate(household_thresholds):
                household_cms[:, s_i + t_i, :] = np.logical_and(
                    self.active_cms[:, i, :] > 0, self.active_cms[:, i, :] < t
                )
                household_cm_names.append(f"{self.CMs[i]}ed to {t}")

        for mask_npi_i, t in enumerate(mask_thresholds):
            mask_cms[:, mask_npi_i, :] = self.active_cms[:, 8, :] > t
            mask_cm_names.append(f"Masks Level {t}")

        all_cm_names = [
            *bin_cm_names,
            *gathering_cm_names,
            *household_cm_names,
            *mask_cm_names,
        ]
        all_activecms = np.concatenate(
            [bin_cms, gathering_cms, household_cms, mask_cms], axis=1
        )
        self.CMs = all_cm_names
        self.active_cms = all_activecms
