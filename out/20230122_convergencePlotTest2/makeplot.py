import sys
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append('danse')


# PATH = 'danse/out/20230122_convergencePlotTest2/smallSROs'  # path to data
PATH = 'danse/out/20230122_convergencePlotTest2/largeSROs'  # path to data
REFERENCE_SENSOR_IDX = 0
#
EXPORT_FILES = True

def main():

    # Load data
    convDataComp, convDataNoComp, convDataCentr = load_data(PATH)

    # Make plot
    plotit(convDataComp, convDataNoComp, convDataCentr)


def plotit(convDataComp, convDataNoComp, convDataCentr):

    nNodes = convDataComp.DANSEfilters.shape[-1]
    
    for k in range(nNodes):

        # Compute Deltas
        diffFiltersReal_comp = 20 * np.log10(np.mean(np.abs(
            np.real(convDataComp.DANSEfilters[:, :, k]) - \
            np.real(convDataCentr.DANSEfiltersCentr[:, :, k])
        ), axis=1))
        diffFiltersImag_comp = 20 * np.log10(np.mean(np.abs(
            np.imag(convDataComp.DANSEfilters[:, :, k]) - \
            np.imag(convDataCentr.DANSEfiltersCentr[:, :, k])
        ), axis=1))
        diffFiltersReal_nocomp = 20 * np.log10(np.mean(np.abs(
            np.real(convDataNoComp.DANSEfilters[:, :, k]) - \
            np.real(convDataCentr.DANSEfiltersCentr[:, :, k])
        ), axis=1))
        diffFiltersImag_nocomp = 20 * np.log10(np.mean(np.abs(
            np.imag(convDataNoComp.DANSEfilters[:, :, k]) - \
            np.imag(convDataCentr.DANSEfiltersCentr[:, :, k])
        ), axis=1))


        fig, axes = plt.subplots(1,1)
        fig.set_size_inches(5.5, 3.5)
        axes.plot(diffFiltersReal_comp, 'k', label=f'$\Delta_\\mathrm{{r}}[i]$ with SRO comp.')
        axes.plot(diffFiltersImag_comp, 'r', label=f'$\Delta_\\mathrm{{i}}[i]$ with SRO comp.')
        axes.plot(diffFiltersReal_nocomp, 'k--', label=f'$\Delta_\\mathrm{{r}}[i]$ without SRO comp.')
        axes.plot(diffFiltersImag_nocomp, 'r--', label=f'$\Delta_\\mathrm{{i}}[i]$ without SRO comp.')
        #
        axes.set_title(f'DANSE convergence towards centr. MWF estimate: node {k+1}')
        nIter = convDataComp.DANSEfilters[:, :, k].shape[1]
        axes.set_xlim([0, nIter])
        axes.set_xlabel('Frame index $i$', loc='left')
        axes.legend()
        axes.grid()
        #
        plt.tight_layout()
        # Export
        if EXPORT_FILES:
            plt.show(block=False)
            fig.savefig(f'{PATH}/convPlot_node{k+1}.png')
            fig.savefig(f'{PATH}/convPlot_node{k+1}.pdf')
        else:
            plt.show()


def load_data(path):
    """Load the convergence data."""

    convDataComp = pickle.load(
        gzip.open(f'{path}/ConvergenceData_comp.pkl.gz', 'r')
    )
    convDataNoComp = pickle.load(
        gzip.open(f'{path}/ConvergenceData_noComp.pkl.gz', 'r')
    )
    convDataCentr = pickle.load(
        gzip.open(f'{path}/ConvergenceData_centr.pkl.gz', 'r')
    )

    return convDataComp, convDataNoComp, convDataCentr


if __name__ == '__main__':
    sys.exit(main())