import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def plot_area_transmission_curve(area_Rt_samples, Ds, title=None):
    li, lq, m, uq, ui = np.percentile(area_Rt_samples, [2.5, 25, 50, 75, 97.5], axis=0)

    plt.plot(Ds, m, color="k")
    plt.fill_between(Ds, li, ui, color="k", alpha=0.1, linewidth=0)
    plt.fill_between(Ds, lq, uq, color="k", alpha=0.3, linewidth=0)
    plt.plot([Ds[0], Ds[-1]], [1, 1], color="tab:red", linewidth=1, zorder=-3)

    if title is not None:
        plt.title(title, fontsize=10)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(fontsize=8)
    plt.xlim([Ds[0], Ds[-1]])

    plt.yticks(fontsize=8)
    plt.ylim([0.25, 2.5])
    plt.ylabel("$Rt$")


def plot_area_infections_curve(area_infections, Ds, title=None):
    li, lq, m, uq, ui = np.percentile(area_infections, [2.5, 25, 50, 75, 97.5], axis=0)

    plt.plot(Ds, m, color="k")
    plt.fill_between(Ds, li, ui, color="k", alpha=0.1, linewidth=0)
    plt.fill_between(Ds, lq, uq, color="k", alpha=0.3, linewidth=0)

    if title is not None:
        plt.title(title, fontsize=10)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(fontsize=8)
    plt.xlim([Ds[0], Ds[-1]])

    plt.yticks(fontsize=8)
    plt.ylabel("$N_t$")


def plot_area_cases_curve(expected_cases, psi_cases, new_cases, Ds, title=None):
    nS, nDs = expected_cases.shape
    output_cases = np.random.negative_binomial(
        psi_cases.reshape((nS, 1)).repeat(nDs, axis=1),
        psi_cases.reshape((nS, 1)) / (expected_cases + psi_cases.reshape((nS, 1))),
    )
    li, lq, m, uq, ui = np.percentile(output_cases, [2.5, 25, 50, 75, 97.5], axis=0)

    plt.plot(Ds, m, color="k")
    plt.fill_between(Ds, li, ui, color="k", alpha=0.1, linewidth=0)
    plt.fill_between(Ds, lq, uq, color="k", alpha=0.3, linewidth=0)
    plt.scatter(Ds, new_cases, s=8, color="tab:blue")

    if title is not None:
        plt.title(title, fontsize=10)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(fontsize=8)
    plt.xlim([Ds[0], Ds[-1]])

    plt.yticks(fontsize=8)
    plt.ylabel("$N_t$")


def plot_area_deaths_curve(expected_deaths, psi_deaths, new_deaths, Ds, title=None):
    nS, nDs = expected_deaths.shape
    output_deaths = np.random.negative_binomial(
        psi_deaths.reshape((nS, 1)).repeat(nDs, axis=1),
        psi_deaths.reshape((nS, 1)) / (expected_deaths + psi_deaths.reshape((nS, 1))),
    )
    li, lq, m, uq, ui = np.percentile(output_deaths, [2.5, 25, 50, 75, 97.5], axis=0)

    plt.plot(Ds, m, color="k")
    plt.fill_between(Ds, li, ui, color="k", alpha=0.1, linewidth=0)
    plt.fill_between(Ds, lq, uq, color="k", alpha=0.3, linewidth=0)
    plt.scatter(Ds, new_deaths, color="tab:red", s=8)

    if title is not None:
        plt.title(title, fontsize=10)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(fontsize=8)
    plt.xlim([Ds[0], Ds[-1]])

    plt.yticks(fontsize=8)
    plt.ylabel("$N_t$")


def add_cms_to_plot(cm_names, active_cms, Ds):
    plt.twinx()
    cm_diff = np.zeros_like(active_cms)
    cm_diff[:, 1:] = np.diff(active_cms)

    diff_marker = np.sum(np.abs(cm_diff), axis=0) > 0
    diff_dates = np.nonzero(diff_marker)[0]

    if len(diff_dates) > 0:
        for diff_date_i, diff_date in enumerate(diff_dates):
            plt.axvline(Ds[diff_date], color="tab:grey", linewidth=1, alpha=0.5)
            changes = np.nonzero(cm_diff[:, diff_date])[0]
            if len(changes > 0):
                for change_i, cm_i in enumerate(changes):
                    plt.text(
                        Ds[diff_date],
                        20 - change_i,
                        cm_names[cm_i],
                        fontsize=4,
                        ha="left" if diff_date_i % 2 == 1 else "right",
                        rotation=45 if diff_date_i % 2 == 1 else -45,
                    )

    plt.ylim([0, 20])
    plt.yticks([])
    plt.xlim([Ds[0], Ds[-1]])


def plot_area_ifr_curve(ifr, Ds, title=None):
    li, lq, m, uq, ui = np.percentile(ifr, [2.5, 25, 50, 75, 97.5], axis=0)

    plt.plot(Ds, m, color="tab:red")
    plt.fill_between(Ds, li, ui, color="tab:red", alpha=0.1, linewidth=0)
    plt.fill_between(Ds, lq, uq, color="tab:red", alpha=0.3, linewidth=0)

    if title is not None:
        plt.title(title, fontsize=10)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(fontsize=8)
    plt.xlim([Ds[0], Ds[-1]])

    plt.yticks(fontsize=8)
    plt.ylabel("IFR$_t$")


def plot_area_iar_curve(ifr, Ds, title=None):
    li, lq, m, uq, ui = np.percentile(ifr, [2.5, 25, 50, 75, 97.5], axis=0)

    plt.plot(Ds, m, color="tab:blue")
    plt.fill_between(Ds, li, ui, color="tab:blue", alpha=0.1, linewidth=0)
    plt.fill_between(Ds, lq, uq, color="tab:blue", alpha=0.3, linewidth=0)

    if title is not None:
        plt.title(title, fontsize=10)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(fontsize=8)
    plt.xlim([Ds[0], Ds[-1]])

    plt.yticks(fontsize=8)
    plt.ylabel("IAR$_t$")


def plot_area_summary(posterior_samples, region_index, data, cm_names=None):
    plt.figure(figsize=(8, 8), dpi=300)

    plt.subplot(611)
    area_Rt_samples = posterior_samples["Rt"][:, region_index, :]
    area_active_cms = data.active_cms[region_index, :, :]
    plot_area_transmission_curve(area_Rt_samples, data.Ds)
    if cm_names is None:
        add_cms_to_plot(data.CMs, area_active_cms, data.Ds)
    else:
        add_cms_to_plot(cm_names, area_active_cms, data.Ds)

    plt.subplot(612)
    area_infections = posterior_samples["total_infections"][:, region_index, 7:]
    plot_area_infections_curve(area_infections, data.Ds)

    plt.subplot(613)
    expected_cases = posterior_samples["expected_cases"][:, region_index, :]
    psi_cases = posterior_samples["psi_cases"]
    new_cases = data.new_cases[region_index, :]
    plot_area_cases_curve(expected_cases, psi_cases, new_cases, data.Ds)

    plt.subplot(614)
    expected_deaths = posterior_samples["expected_deaths"][:, region_index, :]
    psi_deaths = posterior_samples["psi_deaths"]
    new_deaths = data.new_deaths[region_index, :]
    plot_area_deaths_curve(expected_deaths, psi_deaths, new_deaths, data.Ds)

    plt.subplot(615)
    if "iar_r" in posterior_samples.keys():
        area_iar = posterior_samples["iar_t"][:, region_index, 7:]
        plot_area_iar_curve(area_iar, data.Ds, None)

    plt.subplot(616)
    if "ifr_r" in posterior_samples.keys():
        area_ifr = posterior_samples["ifr_t"][:, region_index, 7:]
        plot_area_iar_curve(area_ifr, data.Ds, None)

    plt.suptitle(data.Rs[region_index], fontsize=10)
    plt.tight_layout()
