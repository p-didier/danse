# Purpose of script:
# Post-process results from the DANSE "inherent robustness to SROs" tests
# conducted with the `tests.danse_robustness_to_sros` script.
# -- Shows a visualization of the filter coefficients norms as a function
# of the amount of SRO in the WASN. 
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Date: 14.04.2023.

import re
import sys
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

FOLDER = f'{Path(__file__).parent.parent}/out/20230503_tests/sros_effect/run2_beta10s_N4096'
RELATIVE_PATH_TO_RESULTS = 'filtNorms/filtNorms.pkl'  # relative to subfolder
SROS_REF_FILENAME = 'srosConsidered.pkl'  # file containing the SROs used for each test in `FOLDER`
LOOK_AT_THE_LAST_N_ITERATIONS = 100  # number of iterations to consider for computing the average filter norms

def main():
    """Main function (called by default when running script)."""
    figs = plot_results(*import_results(folder=FOLDER))

    for fig in figs:
        fig.savefig(f'{FOLDER}/{fig.get_label()}.png', dpi=300)
        fig.savefig(f'{FOLDER}/{fig.get_label()}.pdf')


def import_results(folder: str):
    """
    Imports the `tests.danse_robustness_to_sros` results from the sub-folders
    contained in the specified folder `folder`.

    Parameters:
    -----------
    folder: str
        Path to the folder containing the sub-folders with the results.

    Returns:
    --------
    resFiltNorm: list
        List of filter-norm results (one per sub-folder).
    resMetrics: list
        List of metrics results (one per sub-folder).
    srosConsidered: list
        List of SROs considered (one per sub-folder).
    sensorToNodeIdx: list[int]
        Sensor-to-node index mappings.
    """

    # List subfolders
    subfolders = [f for f in Path(folder).iterdir() if f.is_dir()]
    # Discard "backupvals" folder
    subfolders = [f for f in subfolders if f.name != 'backupvals']

    # Read SROs considered
    with open(f'{folder}/{SROS_REF_FILENAME}', 'rb') as f:
        srosConsidered = pickle.load(f)

    # Import results
    resFiltNorm = []
    for ii, subfolder in enumerate(subfolders):
        # Import results
        print(f'Importing results from {subfolder}...')
        pathToFile = f'{subfolder}/{RELATIVE_PATH_TO_RESULTS}'
        with open(pathToFile, 'rb') as f:
            results = pickle.load(f)
        # Append results
        resFiltNorm.append(results)
        
        if ii == 0:
            # Find YAML file containing numbers of sensor per node
            yamlFile = [f for f in Path(subfolder).iterdir() if f.suffix == '.yaml'][0]
            # Read numbers of sensor per node
            with open(yamlFile, 'r') as f:
                yamlContent = f.readlines()
                # Find line containing numbers of sensor per node
                for line in yamlContent:
                    if 'nSensorPerNode' in line:
                        break
                # Extract numbers of sensor per node
                nSensorPerNode = [int(n) for n in re.findall(r'\d+', line)]
            # Translate to sensor-to-node index mapping
            sensorToNodeIdx = []
            for idxNode, nSensors in enumerate(nSensorPerNode):
                sensorToNodeIdx += [idxNode] * nSensors

    # Load metrics res object
    with open(f'{folder}/metricsAsFctOfSROs.pkl', 'rb') as f:
        resMetrics = pickle.load(f)

    return resFiltNorm, resMetrics, srosConsidered, sensorToNodeIdx


def plot_results(
        resFiltNorm: list,
        resMetrics: list,
        srosConsidered: list,
        sensorToNodeIdx: list
    ) -> list:
    """
    Plots the results from the `tests.danse_robustness_to_sros` script.

    Parameters:
    -----------
    resFiltNorm: list
        List of filter-norm results (one per sub-folder).
    resMetrics: list
        List of metrics results (one per sub-folder).
    srosConsidered: list
        List of SROs considered (one per sub-folder).
    sensorToNodeIdx: list[int]
        Sensor-to-node index mappings.

    Returns:
    --------
    figs: list
        List of all figures (metrics, then filter-norm).
    """

    # Derive number of sensors per node
    nSensorsPerNode = np.unique(sensorToNodeIdx, return_counts=True)[1] 
    # Metrics plot
    figsMetrics = plot_metrics_as_fct_of_sros(
        resMetrics,
        nSensorPerNode=nSensorsPerNode
    )

    # Filter norms plot
    figsFiltNorm = plot_filtnorm_as_fct_of_sros(
        resFiltNorm,
        sensorToNodeIdx,
        srosConsidered
    )

    return figsMetrics + figsFiltNorm


def plot_filtnorm_as_fct_of_sros(resFiltNorm, sensorToNodeIdx, srosConsidered):
    """
    Plots the filter-norm results from the `tests.danse_robustness_to_sros`
    script.

    Parameters:
    -----------
    resFiltNorm: list
        List of filter-norm results (one per sub-folder).
    srosConsidered: list
        List of SROs considered (one per sub-folder).
    sensorToNodeIdx: list[int]
        Sensor-to-node index mappings.

    Returns:
    --------
    figs: list
        List of figures.
    """
    # Infer titles
    nNodes = int(len(resFiltNorm[0]) / 2)
    titles = [f'DANSE filter norms, node {k + 1}'\
        for k in range(nNodes)]
    titles += [f'Centralized filter norms, node {k + 1}'\
        for k in range(nNodes)]

    # Arrange results
    filtNorms = [np.zeros((len(resFiltNorm), resFiltNorm[0][k].shape[1]))\
        for k in range(len(resFiltNorm[0]))]
    # ^^^ one list per filtNorm plot (i.e., one per node + one per node centralized)
    #     In each list, the first dimension is the number of SROs considered,
    #     the second dimension is the number of filter coefficients.

    for idxSROs, resultsCurrSROs in enumerate(resFiltNorm):
        for k in range(len(resultsCurrSROs)):
            # Compute mean filter norms starting from iteration
            nTotalIters = resultsCurrSROs[k].shape[0]
            # startIterIdx = int(nTotalIters / 2) # start from the middle of the iterations
            startIterIdx = nTotalIters - LOOK_AT_THE_LAST_N_ITERATIONS
            # Get filter coefficients norms
            filtNorms[k][idxSROs, :] = np.mean(
                resultsCurrSROs[k][startIterIdx:, :],
                axis=0
            )

    def _build_legend_danse(k, sensorToNodeIdx) -> list:
        """Builds the legend for the filter coefficients norms plot, 
        when the filters considered are DANSE filters.
        
        Parameters:
        -----------
        k: int
            Node index.
        sensorToNodeIdx: np.ndarray[int]
            Sensor-to-node index mappings.
        """
        # Derive number of sensors per node
        nSensorsPerNode = np.unique(sensorToNodeIdx, return_counts=True)[1]

        # Initialize legend
        legend = [f'Local ($k={k+1}$), $m={m + 1}/{nSensorsPerNode[k]}$'\
            for m in range(nSensorsPerNode[k])]
        
        # Derive indices of other nodes
        otherNodesIdx = np.arange(len(nSensorsPerNode))
        otherNodesIdx = otherNodesIdx[otherNodesIdx != k]
        # Add other nodes
        for idxNode in otherNodesIdx:
            legend += [f'Node {idxNode + 1} (fused)']
        return legend
    
    def _build_legend_centralized(k, sensorToNodeIdx):
        """Builds the legend for the filter coefficients norms plot, 
        when the filters considered are centralized filters.
        
        Parameters:
        -----------
        k: int
            Node index.
        sensorToNodeIdx: np.ndarray[int]
            Sensor-to-node index mappings.
        """
        # Derive number of sensors per node
        nSensorsPerNode = np.unique(sensorToNodeIdx, return_counts=True)[1]

        # Base: node index
        legend = [f'Node {ii+1}' for ii in sensorToNodeIdx]
        # Add mic index
        currNode = 0
        counterCurrNode = 0
        for ii in range(len(sensorToNodeIdx)):
            if sensorToNodeIdx[ii] == currNode:
                counterCurrNode += 1
            else:
                currNode = sensorToNodeIdx[ii]
                counterCurrNode = 1
            legend[ii] += f', $m={counterCurrNode}/{nSensorsPerNode[k]}$'
        # Add reference marker
        legend[sum(nSensorsPerNode[:k] + 1)] += ' (ref)'
        return legend
    
    def plot_results_single(fNorm, sros, ti, leg, yLim=None):
        """Plots the results for a single filter coefficients norms plot."""
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 6)
        # Plot
        ax.plot(fNorm, '.-')
        ax.grid(True)
        # Add labels
        ax.set_xticks(range(len(sros)))
        ax.set_xticklabels(sros, rotation=45)
        ax.set_xlabel(f'SROs considered [PPM] (nodes 2 to {len(sros[0]) + 1})')
        ax.set_ylabel('Average filter coefficients norms')
        # Set legend on the right outside of the plot
        ax.legend(leg, bbox_to_anchor=(1.05, 1), loc='upper left')
        # Add text to the right of the plot to indicate the number of iterations
        ax.text(
            1.05, 0.25, f'Iterations used\nfor average:\nlast {LOOK_AT_THE_LAST_N_ITERATIONS}',
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes
        )
        ax.text(
            1.05, 0.5, f'Total number of\niterations: {nTotalIters}',
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes
        )
        #
        ax.set_title(ti)
        if yLim is not None:
            ax.set_ylim(yLim)
        fig.tight_layout()
        return fig
    
    # Derive y axis limits
    yLim = [
        np.amin(np.array([np.amin(f) for f in filtNorms])),
        np.amax(np.array([np.amax(f) for f in filtNorms]))
    ]
    yLim = [
        yLim[0] - 0.1 * (yLim[1] - yLim[0]),
        yLim[1] + 0.1 * (yLim[1] - yLim[0])
    ]
    # Plot results
    figs = []
    for k in range(len(filtNorms)):
        if k < nNodes:
            leg = _build_legend_danse(k, np.array(sensorToNodeIdx))
            figLab = f'filtNormsComp_n{k + 1}'
        else:
            leg = _build_legend_centralized(k - nNodes, np.array(sensorToNodeIdx))
            figLab = f'filtNormsComp_c{k - nNodes + 1}'
        fig = plot_results_single(
            filtNorms[k],
            srosConsidered,
            titles[k],
            leg,
            yLim
        )
        fig.set_label(figLab)
        figs.append(fig)

    return figs


def plot_metrics_as_fct_of_sros(res: list[dict], nSensorPerNode: np.ndarray):
    """
    Post-processes the results of a test batch.
    
    Parameters
    ----------
    res : list[dict]
        Results.
    nSensorPerNode : np.ndarray[int]
        Number of sensors per node.
    
    Returns
    ----------
    figs : list[matplotlib.figure.Figure]
        List of figures, one for each node in the WASN.
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """

    # Get useful variables
    nNodes = len(res[0]['snr'])
    # Extract local and raw results (same for all SROs)
    localResSNR = np.zeros(nNodes)
    localResSTOI = np.zeros(nNodes)
    rawResSNR = np.zeros(nNodes)
    rawResSTOI = np.zeros(nNodes)
    for k in range(nNodes):
        localResSNR[k] = res[0]['snr'][f'Node{k+1}']['local']
        localResSTOI[k] = res[0]['estoi'][f'Node{k+1}']['local']
        rawResSNR[k] = res[0]['snr'][f'Node{k+1}']['raw']
        rawResSTOI[k] = res[0]['estoi'][f'Node{k+1}']['raw']

    # Build arrays for DANSE and centralized results
    danseResSNR = np.zeros((len(res), nNodes))
    danseResSTOI = np.zeros((len(res), nNodes))
    centralResSNR = np.zeros((len(res), nNodes))
    centralResSTOI = np.zeros((len(res), nNodes))
    for ii in range(len(res)):
        for k in range(nNodes):
            danseResSNR[ii, k] = res[ii]['snr'][f'Node{k+1}']['danse']
            danseResSTOI[ii, k] = res[ii]['estoi'][f'Node{k+1}']['danse']
            centralResSNR[ii, k] = res[ii]['snr'][f'Node{k+1}']['centr']
            centralResSTOI[ii, k] = res[ii]['estoi'][f'Node{k+1}']['centr']
    
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
        axes[0].hlines(localResSNR[k], 0, len(res) - 1, color='C3', linestyles='dashed', label='Local')
        axes[0].hlines(rawResSNR[k], 0, len(res) - 1, color='C0', linestyles='dashdot', label='Raw')
        axes[0].set_xlabel('SROs [PPM]')
        axes[0].set_title('SNR')
        axes[0].set_ylabel('[dB]')
        axes[0].set_xticks(np.arange(len(res)))
        axes[0].set_xticklabels(
            [str(res[ii]['sros'][1:]) for ii in range(len(res))],
            rotation=90
        )
        axes[0].legend(loc='upper right')
        axes[0].grid()
        axes[0].set_ylim(yLimSNR)  # SNR limits
        # plt.show()
        axes[1].plot(danseResSTOI[:, k], color='C1', marker='o', label='DANSE')
        axes[1].plot(centralResSTOI[:, k], color='C2', marker='s', label='Centralized')
        axes[1].hlines(localResSTOI[k], 0, len(res) - 1, color='C3', linestyles='dashed', label='Local')
        axes[1].hlines(rawResSTOI[k], 0, len(res) - 1, color='C0', linestyles='dashdot', label='Raw')
        axes[1].set_xlabel('SROs [PPM]')
        axes[1].set_title('eSTOI')
        axes[1].set_xticks(np.arange(len(res)))
        axes[1].set_xticklabels(
            [str(res[ii]['sros'][1:]) for ii in range(len(res))],
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