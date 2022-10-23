# Post-processing functions and scripts for, e.g.,
# visualizing DANSE outputs.
#
# ~created on 20.10.2022 by Paul Didier

import matplotlib.pyplot as plt
from danse.danse_toolbox.d_classes import DANSEoutputs

def plot_des_sig_est(data: DANSEoutputs):
    """
    Visualize DANSE output: desired signal estimates.
    """

    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    axes.plot(data.TDdesiredSignals)
    axes.grid()
    plt.tight_layout()
    plt.show()


def plot_sros(data: DANSEoutputs):
    """
    Visualize DANSE output: desired signal estimates.
    """

    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    axes.plot(data.SROsEstimates[0] * 1e6)
    axes.plot(data.SROsResiduals[0] * 1e6, '--')
    axes.grid()
    plt.tight_layout()
    plt.show()

    stop = 1