"""
:code:`preprocessed_data.py`

PreprocessedData Class definition.
"""
import numpy as np
import pandas as pd


def preprocess_data(data_path, last_day='2020-12-05', npi_start_col=6):
    """
    Process data, return PreprocessedData() object

    :param data_path: path to data
    :param last_day: last day of window of analysis
    :param npi_start_col: column index (-2) of first npi
    :return: PreprocessedData() object with loaded data.
    """
    df = pd.read_csv(data_path, parse_dates=["date"], infer_datetime_format=True).set_index(
        ["area", "date"])

    if last_day is None:
        Ds = list(df.index.levels[1])
    else:
        Ds = list(df.index.levels[1])
        last_ts = pd.to_datetime(last_day, utc=True)
        Ds = Ds[:(1 + Ds.index(last_ts))]

    Rs = list(df.index.levels[0])
    CMs = list(df.columns[npi_start_col:])

    nRs = len(Rs)
    nDs = len(Ds)
    nCMs = len(CMs)

    active_cms = np.zeros((nRs, nCMs, nDs))
    new_cases = np.ma.zeros((nRs, nDs))
    new_deaths = np.ma.zeros((nRs, nDs))

    for r_i, r in enumerate(Rs):
        r_df = df.loc[r].loc[Ds]
        new_cases.data[r_i, :] = r_df['new_cases']
        new_deaths.data[r_i, :] = r_df['new_deaths']

        for cm_i, cm in enumerate(CMs):
            active_cms[r_i, cm_i, :] = r_df[cm]

    # mask days where there are negative cases or deaths - because this
    # is clearly wrong
    new_cases[new_cases < 0] = np.ma.masked
    new_deaths[new_deaths < 0] = np.ma.masked

    return PreprocessedData(Rs, Ds, CMs, new_cases, new_deaths, active_cms)


class PreprocessedData(object):
    """
    PreprocessedData Class

    Class to hold data which is subsequently passed onto a PyMC3 model. Mostly a data wrapper, with some utility
    functions.
    """

    def __init__(self, Rs, Ds, CMs, new_cases, new_deaths, active_cms):
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

    @property
    def nCMs(self):
        return len(self.CMs)

    @property
    def nRs(self):
        return len(self.Rs)

    @property
    def nDs(self):
        return len(self.Ds)

    def reduce_regions_from_index(self, reduced_regions_indx):
        """
        Reduce data to only pertain to region indices given. Occurs in place.

        e.g., if reduced_regions_indx = [0], the resulting data object will contain data about only the first region.

        :param reduced_regions_indx: region indices to retain.
        """
        self.Active = self.Active[reduced_regions_indx, :]
        self.Confirmed = self.Confirmed[reduced_regions_indx, :]
        self.Deaths = self.Deaths[reduced_regions_indx, :]
        self.NewDeaths = self.NewDeaths[reduced_regions_indx, :]
        self.NewCases = self.NewCases[reduced_regions_indx, :]
        self.ActiveCMs = self.ActiveCMs[reduced_regions_indx, :, :]

    def remove_regions(self, regions_to_remove):
        """
        Remove region codes corresponding to regions in regions_to_remove. Occurs in place.

        :param regions_to_remove: Region codes, corresponding to regions to remove.
        """
        reduced_regions = []
        reduced_regions_indx = []
        for indx, r in enumerate(self.Rs):
            if r in regions_to_remove:
                pass
            else:
                reduced_regions_indx.append(indx)
                reduced_regions.append(r)

        self.Rs = reduced_regions
        _, nCMs, nDs = self.ActiveCMs.shape
        self.reduce_regions_from_index(reduced_regions_indx)

    def mask_reopenings(self, d_min=90, n_extra=0, print_out=True):
        """
        Mask reopenings.

        This finds dates NPIs reactivate, then mask forwards, giving 3 days for cases and 12 days for deaths.

        :param d_min: day after which to mask reopening.
        :param n_extra: int, number of extra days to mask
        """
        total_cms = self.active_cms
        diff_cms = np.zeros_like(total_cms)
        diff_cms[:, :, 1:] = total_cms[:, :, 1:] - total_cms[:, :, :-1]
        rs, ds = np.nonzero(np.any(diff_cms < 0, axis=1))
        nnz = rs.size

        for nz_i in range(nnz):
            if (ds[nz_i] + 3) > d_min and ds[nz_i] + 3 < len(self.Ds):
                if print_out:
                    print(f"Masking {self.Rs[rs[nz_i]]} from {self.Ds[ds[nz_i] + 3]}")
                self.new_cases[rs[nz_i], ds[nz_i] + 3 - n_extra:].mask = True
                self.new_deaths[rs[nz_i], ds[nz_i] + 11 - n_extra:].mask = True

    def mask_region_ends(self, n_days=20):
        """
        Mask the final n_days days across all countries.

        :param n_days: number of days to mask.
        """
        for rg in self.Rs:
            i = self.Rs.index(rg)
            self.new_cases.mask[i, -n_days:] = True
            self.new_deaths.mask[i, -n_days:] = True

    def mask_region(self, region, days_shown=14):
        """
        Mask all but the first 14 days of cases and deaths for a specific region

        :param region: region code (2 digit EpidemicForecasting.org) code to mask
        :param days: Number of days to provide to the model
        """
        i = self.Rs.index(region)
        c_s = np.nonzero(np.cumsum(self.new_cases.data[i, :] > 0) == days_shown + 1)[0][0]
        d_s = np.nonzero(np.cumsum(self.new_deaths.data[i, :] > 0) == days_shown + 1)[0]
        if len(d_s) > 0:
            d_s = d_s[0]
        else:
            d_s = len(self.Ds)

        self.new_cases.mask[i, c_s:] = True
        self.new_deaths.mask[i, d_s:] = True

        return c_s, d_s

    def unmask_all(self):
        """
        Unmask all cases, deaths.
        """
        self.Active.mask = False
        self.Confirmed.mask = False
        self.Deaths.mask = False
        self.NewDeaths.mask = False
        self.NewCases.mask = False
