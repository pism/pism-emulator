#!/env/bin python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import numpy as np
import pandas as pd
import pylab as plt


def set_size(w, h, ax=None):
    """ w, h: width, height in inches """

    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


if __name__ == "__main__":

    __spec__ = None

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Gaussian Process Emulator."

    options = parser.parse_args()

    idir = "gp-stepaic"

    rcps = ["26", "45", "85"]
    percentiles = [5, 16, 50, 84, 95]

    rcp_col_dict = {"CTRL": "k", "85": "#990002", "45": "#5492CD", "26": "#003466"}
    rcp_shade_col_dict = {"CTRL": "k", "85": "#F4A582", "45": "#92C5DE", "26": "#4393C3"}

    year = 2100
    years = range(2010, 2300)
    nt = len(years)
    m_predictions = {}
    for rcp in rcps:
        predictions = {
            "5": np.ma.array(data=np.zeros((nt)), mask=np.zeros((nt))),
            "16": np.ma.array(data=np.zeros((nt)), mask=np.zeros((nt))),
            "50": np.ma.array(data=np.zeros((nt)), mask=np.zeros((nt))),
            "84": np.ma.array(data=np.zeros((nt)), mask=np.zeros((nt))),
            "95": np.ma.array(data=np.zeros((nt)), mask=np.zeros((nt))),
        }
        for idx, year in enumerate(years):
            filename = os.path.join(idir, "gp_rcp_{}_{}.csv".format(rcp, year))
            missing = False
            try:
                m_df = pd.read_csv(filename)
                predictions["5"][idx], predictions["16"][idx], predictions["50"][idx], predictions["84"][
                    idx
                ], predictions["95"][idx] = np.percentile(m_df["prediction"].values, percentiles)
            except:
                predictions["5"].mask[idx], predictions["16"].mask[idx], predictions["50"].mask[idx], predictions[
                    "84"
                ].mask[idx], predictions["95"].mask[idx] = (True, True, True, True, True)

        m_predictions[rcp] = predictions

    for rcp in rcps:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.fill_between(
            years, m_predictions[rcp]["16"], m_predictions[rcp]["84"], color=rcp_shade_col_dict[rcp], linewidth=0
        )
        for pctl in percentiles:
            ax.plot(
                years, m_predictions[rcp][str(pctl)], color=rcp_col_dict[rcp], linewidth=0.4, linestyle="-", label="GP"
            )
        ax.plot(years, m_predictions[rcp]["50"], color=rcp_col_dict[rcp], linewidth=0.6, linestyle="-", label="GP")
        set_size(3.2, 1.2)
        fig.savefig("gp_rcp_{}.pdf".format(rcp), bbox_inches="tight")
