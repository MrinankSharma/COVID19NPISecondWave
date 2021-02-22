import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_gi(posterior_samples, mean_varname="gi_mean", sd_varname="gi_sd", newfig=True):
    if newfig:
        plt.figure(figsize=(6, 3), dpi=300)
    plt.subplot(121)
    az.plot_kde(posterior_samples[mean_varname], ax=plt.gca())
    plt.ylabel("density")
    plt.xlabel("$\mu_{GI}$")
    plt.gca().set_ylim(bottom=0)

    plt.subplot(122)
    az.plot_kde(posterior_samples[sd_varname], ax=plt.gca())
    plt.ylabel("density")
    plt.xlabel("$\sigma_{GI}$")
    plt.tight_layout()
    plt.gca().set_ylim(bottom=0)


def plot_cases_death_delays(
    data, posterior_samples, cd_prefix="cd_", dd_prefix="dd_", newfig=True
):
    if newfig:
        plt.figure(figsize=(6, 6), dpi=300)

    Cs = list(data.unique_Cs)
    nCs = len(Cs)
    cols = sns.color_palette("colorblind")
    plt.subplot(221)
    for c_i, c in enumerate(Cs):
        sns.kdeplot(
            posterior_samples[f"{cd_prefix}mean_{c}"],
            ax=plt.gca(),
            color=cols[c_i],
            label=c,
        )

    plt.ylabel("density")
    plt.xlabel("$\mu_{CD}$")
    plt.gca().set_ylim(bottom=0)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(loc="upper right", fancybox=True, shadow=True, fontsize=8)

    plt.subplot(222)
    for c_i, c in enumerate(Cs):
        az.plot_kde(
            posterior_samples[f"{cd_prefix}disp_{c}"],
            ax=plt.gca(),
            plot_kwargs={"color": cols[c_i]},
        )
    plt.ylabel("density")
    plt.xlabel("$\Psi_{CD}$")
    plt.tight_layout()
    plt.gca().set_ylim(bottom=0)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.subplot(223)
    for c_i, c in enumerate(Cs):
        az.plot_kde(
            posterior_samples[f"{dd_prefix}mean_{c}"],
            ax=plt.gca(),
            plot_kwargs={"color": cols[c_i]},
        )

    plt.ylabel("density")
    plt.xlabel("$\mu_{DD}$")
    plt.gca().set_ylim(bottom=0)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.subplot(224)
    for c_i, c in enumerate(Cs):
        az.plot_kde(
            posterior_samples[f"{cd_prefix}disp_{c}"],
            ax=plt.gca(),
            plot_kwargs={"color": cols[c_i]},
        )
    plt.ylabel("density")
    plt.xlabel("$\Psi_{DD}$")
    plt.tight_layout()
    plt.gca().set_ylim(bottom=0)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)


def plot_output_noise_scales(
    posterior_samples,
    cases_varname="psi_cases",
    deaths_varname="psi_deaths",
    newfig=True,
):
    if newfig:
        plt.figure(figsize=(6, 3), dpi=300)

    cols = sns.color_palette("colorblind")

    plt.subplot(221)
    sns.kdeplot(posterior_samples[cases_varname], ax=plt.gca())

    plt.ylabel("density")
    plt.xlabel("$\Psi_{C}$")
    plt.gca().set_ylim(bottom=0)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.subplot(222)
    sns.kdeplot(posterior_samples[deaths_varname], ax=plt.gca())

    plt.ylabel("density")
    plt.xlabel("$\Psi_{D}$")
    plt.tight_layout()
    plt.gca().set_ylim(bottom=0)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)


def plot_rw_noise_scales(
    posterior_samples,
    r_varname="r_walk_noise_scale",
    cases_varname="iar_walk_noise_scale",
    deaths_varname="ifr_walk_noise_scale",
    inf_varname="infection_noise_scale",
    newfig=True,
):
    if newfig:
        plt.figure(figsize=(6, 6), dpi=300)

    cols = sns.color_palette("colorblind")

    plt.subplot(221)
    if r_varname in posterior_samples.keys():
        sns.kdeplot(posterior_samples[r_varname], ax=plt.gca())

    plt.ylabel("density")
    plt.xlabel("$\sigma_R$")
    plt.gca().set_ylim(bottom=0)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.subplot(222)

    if cases_varname in posterior_samples.keys():
        sns.kdeplot(posterior_samples[cases_varname], ax=plt.gca())

    plt.ylabel("density")
    plt.xlabel("$\sigma_C$")
    plt.tight_layout()
    plt.gca().set_ylim(bottom=0)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.subplot(223)
    if deaths_varname in posterior_samples.keys():
        sns.kdeplot(posterior_samples[deaths_varname], ax=plt.gca())

    plt.ylabel("density")
    plt.xlabel("$\sigma_D$")
    plt.tight_layout()
    plt.gca().set_ylim(bottom=0)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.subplot(224)
    if inf_varname in posterior_samples.keys():
        sns.kdeplot(posterior_samples[inf_varname], ax=plt.gca())

    plt.ylabel("density")
    plt.xlabel("$\sigma_N$")
    plt.tight_layout()
    plt.gca().set_ylim(bottom=0)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)


def plot_kdes(
    posterior_samples,
    varnames,
    newfig=True,
):
    if newfig:
        plt.figure(figsize=(6, 6), dpi=300)

    nVs = len(varnames)
    width = int(np.sqrt(nVs)) + 1

    for i, v in enumerate(varnames):
        plt.subplot(width, width, i + 1)
        sns.kdeplot(posterior_samples[v], ax=plt.gca())
        plt.ylabel("density")
        plt.xlabel(v)
        plt.gca().set_ylim(bottom=0)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

    plt.tight_layout()
