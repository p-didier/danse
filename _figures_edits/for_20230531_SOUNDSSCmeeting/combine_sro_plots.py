# Purpose of script:
# Combines SRO line plots to show both online and batch mode results.

# TODO: TODO: TODO: TODO: TODO: TODO: TODO: 
# Not actually used.
# TODO: TODO: TODO: TODO: TODO: TODO: TODO: 
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tests.danse_robustness_to_sros_postproc import import_results

FOLDER_BATCH = f'{Path(__file__).parent.parent}/out/20230505_tests/sros_effect/run1'
FOLDER_ONLINE = f'{Path(__file__).parent.parent}/out/20230505_tests/sros_effect/run1'

def main(
        folderOnline=FOLDER_ONLINE,
        folderBatch=FOLDER_BATCH
    ):
    """Main function (called by default when running script)."""
    resOnline = import_results(folderOnline, onlyMetrics=True)
    resBatch = import_results(folderBatch, onlyMetrics=True)
    
    # Derive number of sensors per node
    nSensorsPerNode = np.unique(resOnline[1], return_counts=True)[1] 
    # Metrics plot
    figsMetrics = plot_metrics_as_fct_of_sros(
        resOnline[0],
        resBatch[0],
        nSensorsPerNode
    )
    stop = 1


def plot_metrics_as_fct_of_sros(
        resOnline: list[dict],
        resBatch: list[dict],
        nSensorPerNode: np.ndarray
    ):
    """
    Post-processes the results of a test batch.
    
    Parameters
    ----------
    resOnline : list[dict]
        Results for online mode.
    resBatch : list[dict]
        Results for batch mode.
    nSensorPerNode : np.ndarray[int]
        Number of sensors per node.
    
    Returns
    ----------
    figs : list[matplotlib.figure.Figure]
        List of figures, one for each node in the WASN.
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """

    # Get useful variables
    nNodes = len(resOnline[0]['snr'])
    # Extract local and raw results (same for all SROs)
    localResSNR = np.zeros(nNodes)
    localResSTOI = np.zeros(nNodes)
    rawResSNR = np.zeros(nNodes)
    rawResSTOI = np.zeros(nNodes)
    for k in range(nNodes):
        localResSNR[k] = resOnline[0]['snr'][f'Node{k+1}']['local']
        localResSTOI[k] = resOnline[0]['estoi'][f'Node{k+1}']['local']
        rawResSNR[k] = resOnline[0]['snr'][f'Node{k+1}']['raw']
        rawResSTOI[k] = resOnline[0]['estoi'][f'Node{k+1}']['raw']

    # Build arrays for DANSE and centralized results
    danseResSNR = np.zeros((len(resOnline), nNodes))
    danseResSTOI = np.zeros((len(resOnline), nNodes))
    centralResSNR = np.zeros((len(resOnline), nNodes))
    centralResSTOI = np.zeros((len(resOnline), nNodes))
    for ii in range(len(resOnline)):
        for k in range(nNodes):
            danseResSNR[ii, k] = resOnline[ii]['snr'][f'Node{k+1}']['danse']
            danseResSTOI[ii, k] = resOnline[ii]['estoi'][f'Node{k+1}']['danse']
            centralResSNR[ii, k] = resOnline[ii]['snr'][f'Node{k+1}']['centr']
            centralResSTOI[ii, k] = resOnline[ii]['estoi'][f'Node{k+1}']['centr']
    
    # Plot
    figs = []
    # Derive y-axis limits for the SNR plot
    yLimSNR = [
        np.amin(np.array([np.amin(danseResSNR), np.amin(centralResSNR), np.amin(localResSNR), np.amin(rawResSNR)])),
        np.amax(np.array([np.amax(danseResSNR), np.amax(centralResSNR), np.amax(localResSNR), np.amax(rawResSNR)]))
    ]
    yLimSNR = [
        yLimSNR[0] - 0.1 * (yLimSNR[1] - yLimSNR[0]),
        yLimSNR[1] + 0.1 * (yLimSNR[1] - yLimSNR[0])
    ]
    
    for k in range(nNodes):
        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)
        axes[0].plot(danseResSNR[:, k], color='C1', marker='o', label='DANSE')
        axes[0].plot(centralResSNR[:, k], color='C2', marker='s', label='Centralized')
        axes[0].hlines(localResSNR[k], 0, len(resOnline) - 1, color='C3', linestyles='dashed', label='Local')
        axes[0].hlines(rawResSNR[k], 0, len(resOnline) - 1, color='C0', linestyles='dashdot', label='Raw')
        axes[0].set_xlabel('SROs [PPM]')
        axes[0].set_title('SNR')
        axes[0].set_ylabel('[dB]')
        axes[0].set_xticks(np.arange(len(resOnline)))
        axes[0].set_xticklabels(
            [str(resOnline[ii]['sros'][1:]) for ii in range(len(resOnline))],
            rotation=90
        )
        axes[0].legend(loc='upper right')
        axes[0].grid()
        axes[0].set_ylim(yLimSNR)  # SNR limits
        # plt.show()
        axes[1].plot(danseResSTOI[:, k], color='C1', marker='o', label='DANSE')
        axes[1].plot(centralResSTOI[:, k], color='C2', marker='s', label='Centralized')
        axes[1].hlines(localResSTOI[k], 0, len(resOnline) - 1, color='C3', linestyles='dashed', label='Local')
        axes[1].hlines(rawResSTOI[k], 0, len(resOnline) - 1, color='C0', linestyles='dashdot', label='Raw')
        axes[1].set_xlabel('SROs [PPM]')
        axes[1].set_title('eSTOI')
        axes[1].set_xticks(np.arange(len(resOnline)))
        axes[1].set_xticklabels(
            [str(resOnline[ii]['sros'][1:]) for ii in range(len(resOnline))],
            rotation=90
        )
        axes[1].legend(loc='lower left')
        axes[1].grid()
        axes[1].set_ylim([0, 1])  # eSTOI limits
        fig.suptitle(f'Node {k + 1} ({nSensorPerNode[k]} sensors)')
        fig.tight_layout()

        # Save figure
        figLab = f'metricsSROs_n{k + 1}'
        fig.set_label(figLab)
        figs.append(fig)
        plt.close(fig)

    return figs


if __name__ == '__main__':
    sys.exit(main())