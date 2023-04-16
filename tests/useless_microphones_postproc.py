# Purpose of script:
# Post-process results from the DANSE "rendering mics useless" tests
# conducted with the `tests.useless_microphones` script.
# -- Shows a visualization of the filter coefficients norms ... TODO
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Date: 15.04.2023.

import re
import sys
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

FOLDER = f'{Path(__file__).parent.parent}/out/20230415_tests/uselessMics/test'
RELATIVE_PATH_TO_RESULTS = 'filtNorms/filtNorms.pkl'  # relative to subfolder
N_USELESS_MICS_TO_PLOT = 3  # number of mics rendered useless to plot
EXPORT_FOLDER = f'postproc{N_USELESS_MICS_TO_PLOT}uselessMic'  # relative to FOLDER

def main():
    """Main function (called by default when running script)."""
    figs = plot_results_1uselessmic(
        *import_results(folder=FOLDER),
        N_USELESS_MICS_TO_PLOT
    )

    for fig in figs:
        fullExportPath = f'{FOLDER}/{EXPORT_FOLDER}'
        # Ensure export folder exists
        if not Path(fullExportPath).exists():
            Path(fullExportPath).mkdir()
        fig.savefig(f'{fullExportPath}/{fig.get_label()}.png', dpi=300)
        # fig.savefig(f'{fullExportPath}/{fig.get_label()}.pdf')


def import_results(folder: str):
    """
    Import results from the `tests.useless_microphones` script.
    
    Parameters
    ----------
    folder : str
        Path to the folder containing the results.
    
    Returns
    ----------
    results : dict
        Dictionary containing the results.
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """
    
    # List subfolders
    subfolders = [f for f in Path(folder).iterdir() if f.is_dir()]
    # Only select the subfolders corresponding to the test cases
    subfolders = [f for f in subfolders if f.name[:4] == 'comb']
    # Sort subfolders by the comb reference number
    subfolders = sorted(
        subfolders,
        key=lambda x: int(re.findall(r'\d+', x.name)[0])
    )

    # Import results
    results = []
    for ii, subfolder in enumerate(subfolders):
        data = pickle.load(open(f'{subfolder}/{RELATIVE_PATH_TO_RESULTS}', 'rb'))
        results.append((subfolder.name, data))
    
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

    # Convert to dictionary
    results = dict(results)

    return results, sensorToNodeIdx


def plot_results_1uselessmic(res: dict, sensorToNodeIdx: list, nUselessMicsToPlot: int = 1):
    """
    Plot results from the `tests.useless_microphones` script, for the case
    where only one microphone is rendered useless.
    
    Parameters
    ----------
    res : dict
        Dictionary containing the results.
    sensorToNodeIdx : list
        List containing the sensor-to-node index mapping.
    nUselessMicsToPlot : int, optional
        Number of microphones rendered useless in the figures to plot.
        The default is 1.
    
    Returns
    ----------
    figs : list
        List containing the figures.
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """
    # Derive number of microphones rendered useless in each entry of `res`
    # using the key of the entry.
    nUselessMics = np.array(
        [int(len(re.findall(r'\d+', k))) - 1 for k in list(res.keys())]
    )

    # Select results for the case where only one microphone is rendered useless
    resNmics = {k: v for ii, (k, v) in enumerate(res.items())\
        if nUselessMics[ii] == nUselessMicsToPlot}

    # Infer number of nodes
    nNodes = np.amax(sensorToNodeIdx) + 1

    # Plot results
    figs = []
    for idx, (key, resCurr) in enumerate(resNmics.items()):
        # Inform user
        print(f'Plotting results for {key} ({idx+1}/{len(resNmics.keys())})...')
        # Get y-axis limits
        yLim = np.array([
            np.amin([np.amin(resCurr[ii]) for ii in range(len(resCurr))]),
            np.amax([np.amax(resCurr[ii]) for ii in range(len(resCurr))])
        ])
        # Get index of the microphone rendered useless
        uselessMicIdx = np.array(
            [int(x) for x in re.findall(r'\d+', key)[-nUselessMicsToPlot:]]
        )
        for ii in range(len(resCurr)):
            fig = plt.figure()
            if ii < nNodes:
                legType = 'danse'
                currNodeIndex = ii
                ti = f'DANSE, node $k={currNodeIndex + 1}$'
                figLab = f'uselessMicIdx{uselessMicIdx + 1}_filtNorms_danse_k{currNodeIndex + 1}'
            else:
                legType = 'centr'
                currNodeIndex = ii - nNodes
                ti = f'Centr., node $k={currNodeIndex + 1}$'
                figLab = f'uselessMicIdx{uselessMicIdx + 1}_filtNorms_centr_k{currNodeIndex + 1}'
            fig.set_label(figLab)
            leg, colors = get_legend_and_colors(
                legType,
                sensorToNodeIdx,
                currNodeIndex,
                uselessMicIdx
            )
            # Plot
            for jj in range(resCurr[ii].shape[-1]):
                plt.plot(resCurr[ii][:, jj], color=colors[jj])
            fig.legend(leg, loc='upper right')
            plt.xlabel('Iteration index')
            plt.ylabel('Filter coefficient norm')
            plt.title(ti, loc='left')
            plt.grid()
            plt.ylim(yLim)
            fig.tight_layout()
            figs.append(fig)
            plt.close()

    return figs


def get_legend_and_colors(
        legType: str,
        sensorToNodeIdx: np.ndarray,
        nodeIndex: int,
        uselessMics: np.ndarray
    ):
    """Build legend for the results of one subfolder."""
    # Ensure sensor-to-node index mapping is a numpy array
    sensorToNodeIdx = np.array(sensorToNodeIdx)
    # Build legend
    if legType == 'danse':
        leg = get_danse_legend(sensorToNodeIdx, nodeIndex, uselessMics)
    elif legType == 'centr':
        leg = get_centr_legend(sensorToNodeIdx, nodeIndex, uselessMics)

    # Build colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [colors[i] for i in range(len(leg))]
    # Remove color of useless microphone
    idxUselessMicLegEntry = [True if 'useless' in entry else False for entry in leg]
    legIndicesUselessMic = np.where(idxUselessMicLegEntry)[0]
    for ii, legIdx in enumerate(legIndicesUselessMic):
        colors[legIdx] = str(
            np.round(1 - (ii + 1) / len(legIndicesUselessMic), 1)
        )

    return leg, colors


def get_danse_legend(sensorToNodeIdx, nodeIndex, uselessMics=None):
    """
    Build legend for the DANSE results.

    Parameters
    ----------
    sensorToNodeIdx : np.ndarray
        Array containing the sensor-to-node index mapping.
    nodeIndex : int
        Index of the node for which the legend is built.
    uselessMics : np.ndarray[int], optional
        Indices of the microphone(s) rendered useless. The default is None.

    Returns
    -------
    leg : list[str]
        Legend.

    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """
    leg = []

    # Add local sensors legend entries
    nLocalSensors = len(np.where(sensorToNodeIdx == nodeIndex)[0])
    for idxSensor in range(len(np.where(sensorToNodeIdx == nodeIndex)[0])):
        leg.append(f'Local $m={idxSensor + 1}/{nLocalSensors}$')
        
    # Add one entry per remote node
    for idxNode in np.unique(sensorToNodeIdx):
        if idxNode != nodeIndex:
            leg.append(f'Remote node (fused) $k={idxNode + 1}$')

    if uselessMics is not None:
        # Mark useless microphone
        allNodesIndices = np.arange(np.amax(sensorToNodeIdx) + 1)
        remoteNodesIndices = np.delete(allNodesIndices, nodeIndex)
        nodeOfUselessMic = sensorToNodeIdx[uselessMics]
        for ii in range(len(nodeOfUselessMic)):
            # Get local index of useless mic
            idxUselessMic = np.where(np.where(sensorToNodeIdx == nodeOfUselessMic[ii])[0] == uselessMics[ii])[0][0]
            if nodeOfUselessMic[ii] == nodeIndex:
                leg[idxUselessMic] += ', useless'
            else:
                # Find index of node with useless mic in remote nodes list
                idxNode = np.where(remoteNodesIndices == nodeOfUselessMic[ii])[0][0]
                leg[nLocalSensors + idxNode] += f', mic {idxUselessMic + 1}/{len(np.where(sensorToNodeIdx == nodeOfUselessMic[ii])[0])} useless'
    
    return leg


def get_centr_legend(sensorToNodeIdx, nodeIndex, uselessMics=None):
    """
    Build legend for the centralised results.

    Parameters
    ----------
    sensorToNodeIdx : np.ndarray
        Array containing the sensor-to-node index mapping.
    nodeIndex : int
        Index of the node for which the legend is built.
    uselessMics : np.ndarray[int], optional
        Indices of the microphone(s) rendered useless. The default is None.

    Returns
    -------
    leg : list[str]
        Legend.

    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """
    leg = []
    # Add one entry per sensor
    for idxSensor in range(len(sensorToNodeIdx)):
        # Get node index of current sensor
        idxNode = sensorToNodeIdx[idxSensor]
        if idxNode == nodeIndex:
            leg.append(f'Local $m={idxSensor + 1}/{sum(sensorToNodeIdx == nodeIndex)}$')
        else:
            # Get local index of current sensor
            idxLocalSensor = np.where(np.where(sensorToNodeIdx == idxNode)[0] == idxSensor)[0][0]
            leg.append(f'Remote $m={idxLocalSensor + 1}/{len(np.where(sensorToNodeIdx == idxNode)[0])}$ of node $k={idxNode + 1}$')

    # Get index of reference sensor
    idxRefSensor = np.where(sensorToNodeIdx == nodeIndex)[0][0]
    # Mark reference sensor (first sensor of current node)
    leg[idxRefSensor] += ', ref.'

    if uselessMics is not None:
        for ii in range(len(uselessMics)):
            leg[uselessMics[ii]] += f', useless'
    
    return leg


if __name__ == '__main__':
    sys.exit(main())