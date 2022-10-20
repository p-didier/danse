# Post-processing functions and scripts for, e.g.,
# visualizing DANSE outputs.
#
# ~created on 20.10.2022 by Paul Didier

import matplotlib.pyplot as plt
from danse.danse_toolbox.d_classes import DANSEoutputs

def visualization(data: DANSEoutputs):
    """
    Visualize DANSE outputs.
    """

    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    axes.plot(data.TDdesiredSignals)
    axes.grid()
    plt.tight_layout()
    plt.show()

    stop = 1