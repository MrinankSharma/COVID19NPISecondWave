import matplotlib.pyplot as plt
import numpy as np


def plot_intervention_effectiveness(
    posterior_samples,
    cm_names=None,
    intervention_varname="alpha_i",
    xlim="auto",
    newfig=True,
):
    if newfig:
        plt.figure(figsize=(4, 3), dpi=300)

    if isinstance(posterior_samples, dict):
        per_red = 100 * (1 - np.exp(-posterior_samples[intervention_varname]))
    else:
        per_red = 100 * (1 - np.exp(-posterior_samples))

    median_alpha = np.median(per_red, axis=0)

    nS, nCMs = per_red.shape

    for i in range(0, nCMs, 2):
        plt.fill_between(
            [-100, 100],
            [-i + 0.5, -i + 0.5],
            [-i - 0.5, -i - 0.5],
            color="k",
            alpha=0.1,
            linewidth=0,
        )

    li, lq, uq, ui = np.percentile(per_red, [2.5, 25, 75, 97.5], axis=0)

    for n in range(nCMs):
        plt.plot([li[n], ui[n]], [-n, -n], color="k", alpha=0.1)
        plt.plot([lq[n], uq[n]], [-n, -n], color="k", alpha=0.5)

    plt.plot([0, 0], [0.5, -nCMs - 2], "k--", linewidth=0.5)

    if cm_names is None:
        cm_names = [f"NPI {i + 1}" for i in range(nCMs)]

    assert len(cm_names) == nCMs

    plt.yticks(-np.arange(len(median_alpha)), cm_names)
    plt.scatter(median_alpha, -np.arange(len(median_alpha)), marker="|", color="k")
    plt.xlabel("Percentage reduction in $R_t$")
    plt.ylim([-nCMs + 0.5, 0.5])

    if xlim == "auto":
        lower_xlim, upper_xlim = plt.xlim()
        lower_xlim = np.floor(lower_xlim / 25.0) * 25
        upper_xlim = np.ceil(upper_xlim / 25.0) * 25
        plt.xlim([lower_xlim, upper_xlim])
    else:
        plt.xlim(xlim)

    plt.title("NPI Effectiveness")


def plot_intervention_correlation(
    posterior_samples,
    cm_names=None,
    intervention_varname="alpha_i",
    newfig=True,
):
    if newfig:
        plt.figure(figsize=(3, 4.5), dpi=300)

    if isinstance(posterior_samples, dict):
        cormat = np.corrcoef(posterior_samples[intervention_varname].T)
    else:
        cormat = np.corrcoef(posterior_samples.T)

    nCMs, _ = cormat.shape
    if cm_names is None:
        cm_names = [f"NPI {i + 1}" for i in range(nCMs)]

    assert len(cm_names) == nCMs

    plt.imshow(cormat, vmin=-1, vmax=1, cmap="PuOr")
    plt.colorbar()
    plt.xticks(np.arange(nCMs), cm_names, rotation=90, fontsize=4)
    plt.yticks(np.arange(nCMs), cm_names, fontsize=4)
    plt.title(f"most correlation\n: {np.min(cormat):-.2f}")


def plot_intervention_sd(
    posterior_samples,
    cm_names=None,
    sd_varname="sigma_i",
    xlim=None,
):
    if isinstance(posterior_samples, dict):
        sigma = posterior_samples[sd_varname]
    else:
        sigma = posterior_samples

    if xlim is None:
        xlim = [0, 1]

    nS, _, nCMs = sigma.shape
    sigma = sigma.reshape((nS, nCMs))

    median_alpha = np.median(sigma, axis=0)

    for i in range(0, nCMs, 2):
        plt.fill_between(
            [-100, 100],
            [-i + 0.5, -i + 0.5],
            [-i - 0.5, -i - 0.5],
            color="k",
            alpha=0.1,
            linewidth=0,
        )

    li, lq, uq, ui = np.percentile(sigma, [2.5, 25, 75, 97.5], axis=0)

    for n in range(nCMs):
        plt.plot([li[n], ui[n]], [-n, -n], color="k", alpha=0.1)
        plt.plot([lq[n], uq[n]], [-n, -n], color="k", alpha=0.5)

    plt.plot([0, 0], [0.5, -nCMs - 2], "k--", linewidth=0.5)

    if cm_names is None:
        cm_names = [f"NPI {i + 1}" for i in range(nCMs)]

    assert len(cm_names) == nCMs

    plt.yticks(-np.arange(len(median_alpha)), cm_names)
    plt.scatter(median_alpha, -np.arange(len(median_alpha)), marker="|", color="k")
    plt.xlabel("Percentage reduction in $R_t$")
    plt.ylim([-nCMs + 0.5, 0.5])

    if xlim == "auto":
        lower_xlim, upper_xlim = plt.xlim()
        lower_xlim = np.floor(lower_xlim / 25.0) * 1
        upper_xlim = np.ceil(upper_xlim / 25.0) * 1
        plt.xlim([lower_xlim, upper_xlim])
    else:
        plt.xlim(xlim)

    plt.title("NPI Effectiveness Variability")


"""
 e.g., grouped_npis can be something like this:
 grouped_npi_base = {
    "Some Face-to-Face Businesses Closed": {
        "npis": ["Some Face-to-Face Businesses Closed"],
        "type": "include"},
    "Gastronomy Closed": {
        "npis": ["Gastronomy Closed"],
        "type": "include"},
    "Leisure Venues Closed": {
        "npis": ["Leisure Venues Closed"],
        "type": "include"},
    "All Face-to-Face Businesses Closed + Stay at Home Order": {
        "npis": ["All Face-to-Face Businesses Closed", "Leisure Venues Closed", "Gastronomy Closed", "Some Face-to-Face Businesses Closed", "Stay at Home Order"],
        "type": "include"},
    "Curfew": {
        "npis": ["Curfew"],
        "type": "include"},
    "Childcare Closed": {
        "npis": ["Childcare Closed"],
        "type": "include"},
    "Schools Closed": {
        "npis": ["Primary Schools Closed", "Secondary Schools Closed"],
        "type": "include"},
    "All Education Closed": {
        "npis": ["Primary Schools Closed", "Secondary Schools Closed", "Universities Away", "Childcare Closed"],
        "type": "include"},
    "Universities Away": {
        "npis": ["Universities Away"],
        "type": "include"},
    "Mandatory Mask Wearing >= 3": {
        "npis": ['Mandatory Mask Wearing >= 3'],
        "type": "include"}
}


 :param grouped_npis: 
 :param alpha_i_samples: 
 :param npi_names: 
 :return: 
 """


def combine_npi_samples(grouped_npis, alpha_i_samples, npi_names):
    nS, nCMs_orig = alpha_i_samples.shape
    CMs_new = list(grouped_npis.keys())
    CMs_new_include = np.ones(len(CMs_new), dtype=np.bool)
    nCMs_new = len(CMs_new)

    new_samples = np.zeros((nS, nCMs_new))
    for cm_i_new, (gnpi, sub_npilist_dict) in enumerate(grouped_npis.items()):
        sub_npilist = sub_npilist_dict["npis"]
        add_type = sub_npilist_dict["type"]

        for cm in sub_npilist:
            if cm in npi_names:
                new_samples[:, cm_i_new] += alpha_i_samples[:, npi_names.index(cm)]
            elif add_type == "exclude":
                CMs_new_include[cm_i_new] = False

    CMs_new = np.array(CMs_new)[CMs_new_include].tolist()
    new_samples = new_samples[:, CMs_new_include]

    return new_samples, CMs_new
