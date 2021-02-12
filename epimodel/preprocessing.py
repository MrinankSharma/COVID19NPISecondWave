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
    data_path, last_day="2021-01-09", npi_start_col=3, skipcases=10, skipdeaths=30
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
        mask_thresholds = [3, 4]

        gathering_npis = [
            "Public Indoor Gathering Person Limit",
            "Private Indoor Gathering Person Limit",
        ]
        binary_npis = [
            "Some Face-to-Face Businesses Closed",
            "Gastronomy Closed",
            "Leisure Venues Closed",
            "All Face-to-Face Businesses Closed",
            "Curfew",
            "Childcare Closed",
            "Primary Schools Closed",
            "Secondary Schools Closed",
            "Universities Away",
        ]
        mask_npi = ["Mandatory Mask Wearing"]

        nCMs = (
            len(binary_npis)
            + len(mask_thresholds) * mask_npi
            + len(gathering_npis) * len(gathering_thresholds)
        )

        nRs, _, nDs = self.active_cms.shape

        new_active_cms = np.zeros((nRs, nCMs, nDs))

        cm_index = 0
        cm_names = []
        for bin_npi in binary_npis:
            old_index = self.CMs.index(bin_npi)
            new_active_cms[:, cm_index, :] = self.active_cms[:, old_index, :]
            cm_index += 1
            cm_names.append(bin_npi)

        for gat_npi in gathering_npis:
            for t in gathering_thresholds:
                old_index = self.CMs.index(gat_npi)
                new_active_cms[:, cm_index, :] = np.logical_and(
                    self.active_cms[:, old_index, :] > 0,
                    self.active_cms[:, old_index, :] < t,
                )
                cm_names.append(f"{gat_npi} < {t}")
                cm_index += 1

        for t in mask_thresholds:
            old_index = self.CMs.index(mask_npi)
            new_active_cms[:, cm_index, :] = np.logical_and(
                self.active_cms[:, old_index, :] > 0,
                self.active_cms[:, old_index, :] < t,
            )
            cm_names.append(f"{mask_npi} < {t}")
            cm_index += 1

        self.CMs = cm_names
        self.active_cms = new_active_cms

    def mask_region_by_index(self, region_index, nz_case_days_shown=60):
        mask_start = np.nonzero(
            np.cumsum(self.new_cases.data[region_index, :] > 0)
            == nz_case_days_shown + 1
        )[0][0]

        self.new_cases[region_index, mask_start:] = True
        self.new_deaths.mask[region_index, mask_start:] = True

        return mask_start
