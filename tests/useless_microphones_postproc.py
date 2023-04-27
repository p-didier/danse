# Purpose of script:
# Post-process results from the DANSE "rendering mics useless" tests
# conducted with the `tests.useless_microphones` script.
# -- Shows a visualization of the filter coefficients norms ... TODO
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Date: 15.04.2023.

import re
import sys
import gzip
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

FOLDER = f'{Path(__file__).parent.parent}/out/20230426_tests/test1'  # GEVD tests
RELPATH_TO_FN_RESULTS = 'filtNorms/filtNorms.pkl'  # relative to subfolder (filter norm results)
RELPATH_TO_M_RESULTS = 'DANSEoutputs.pkl.gz'  # relative to subfolder (metrics results)
N_USELESS_MICS_TO_PLOT = 1  # number of mics rendered useless to plot (only used if TEST_TYPE == 'render_mics_useless')

# TEST_TYPE = ['render_mics_useless']  # type of test to post-process
# TEST_TYPE = ['add_useless_mics']
TEST_TYPE = ['add_useless_mics', 'render_mics_useless']
    # ^^^ 'render_mics_useless': render some mics useless.
    # ^^^ 'add_useless_mics': add some useless mics.


def main():
    """Main function (called by default when running script)."""

    for testType in TEST_TYPE:
    
        if testType == 'render_mics_useless':  
            EXPORT_FOLDER = f'postproc_r{N_USELESS_MICS_TO_PLOT}mu'  # relative to `FOLDER`
        elif testType == 'add_useless_mics':
            EXPORT_FOLDER = 'postproc_aum'  # relative to `FOLDER`

        # Plot filter coefficients norms
        figsFiltNorm = plot_filtnorm_results(
            *import_results_fn(testType),
            testType=testType
        )
        # Plot metrics
        if testType == 'render_mics_useless':
            print('Metrics plot not implemented (yet) for RMU tests.')
            figsMetrics = []
        elif testType == 'add_useless_mics':
            figsMetrics = plot_metrics_results(
                *import_results_m(testType),
                testType=testType
            )

        # Export figures
        figs = figsMetrics + figsFiltNorm
        for fig in figs:
            fullExportPath = f'{FOLDER}/{EXPORT_FOLDER}'
            # Ensure export folder exists
            if not Path(fullExportPath).exists():
                Path(fullExportPath).mkdir()
            fig.savefig(f'{fullExportPath}/{fig.get_label()}.png', dpi=300)
            # fig.savefig(f'{fullExportPath}/{fig.get_label()}.pdf')


def import_results_fn(testType):
    """
    Import filter norm results from the `tests.useless_microphones` script.
    
    Returns
    ----------
    results : dict
        Dictionary containing the results.
    sensorToNodeIdx : list
        List of sensor-to-node indices for each test case.
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """
    # List subfolders
    subfolders = get_subfolders(testType)

    # Import results
    results = []
    sensorToNodeIdx = []
    for _, subfolder in enumerate(subfolders):
        data = pickle.load(open(f'{subfolder}/{RELPATH_TO_FN_RESULTS}', 'rb'))
        results.append((subfolder.name, data))
        sensorToNodeIdx.append(
            get_sensor_to_node_idx(subfolder, testType=testType)
        )

    # Convert to dictionary
    results = dict(results)

    return results, sensorToNodeIdx


def import_results_m(testType):
    """
    Import speech enhancement metrics results from the
    `tests.useless_microphones` script.
    
    Returns
    ----------
    results : dict
        Dictionary containing the results.
    sensorToNodeIdx : list
        List of sensor-to-node indices for each test case.
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """
    # List subfolders
    subfolders = get_subfolders(testType)

    # Import results
    results = []
    sensorToNodeIdx = []
    for _, subfolder in enumerate(subfolders):
        data = pickle.load(gzip.open(f'{subfolder}/{RELPATH_TO_M_RESULTS}', 'rb'))
        results.append((subfolder.name, data.metrics))
        sensorToNodeIdx.append(
            get_sensor_to_node_idx(subfolder, testType=testType)
        )

    # Convert to dictionary
    results = dict(results)

    return results, sensorToNodeIdx


def get_subfolders(testType):
    """
    Lists the desired tests subfolders.
    """
    # Get test ref from test type
    testRef = ''.join([s[0] for s in re.split('_', testType)]) 
    
    # List subfolders
    subfolders = [f for f in Path(FOLDER).iterdir() if f.is_dir()]
    # Only select the subfolders corresponding to the test cases
    subfolders = [f for f in subfolders if f.name[:len(testRef)] == testRef]
    # Sort subfolders by the comb reference number
    subfolders = sorted(
        subfolders,
        key=lambda x: int(re.findall(r'\d+', x.name)[0])
    )

    return subfolders


def get_sensor_to_node_idx(
        subfolder: str,
        txtFileName: str = 'TestParameters_text.txt',
        testType: str = 'render_mics_useless'
    ):
    """
    Get the sensor-to-node index mapping from YAML or TXT file.

    Parameters
    ----------
    subfolder : str
        Path to the folder containing the results.
    txtFileName : str, optional
        Name of the TXT file containing the sensor-to-node index mapping.
        Only used if `TEST_TYPE` is 'add_useless_mics'. Default is
        'TestParameters_text.txt'.
    testType : str, optional
        Type of test to post-process. Default is 'render_mics_useless'.
    
    Returns
    ----------
    sensorToNodeIdx : list
        List containing the sensor-to-node index mapping.  
    """

    if testType == 'render_mics_useless':
        # Find YAML file containing numbers of sensor per node
        file = [f for f in Path(subfolder).iterdir() if\
            f.suffix == '.yaml'][0]
    elif testType == 'add_useless_mics':
        # Find TXT file containing sensor-to-node index mapping
        file = [f for f in Path(subfolder).iterdir() if\
            f.name == txtFileName][0]
    else:
        raise ValueError(f'Unknown test type "{testType}".')
    
    # Read sensor-to-node index mapping
    with open(file, 'r') as f:
        txtContent = f.readlines()
        # Find line containing sensor-to-node index mapping
        for line in txtContent:
            if 'nSensorPerNode' in line:
                break
    # Extract sensor-to-node index mapping
    nSensorPerNode = [int(n) for n in re.findall(r'\d+', line)]
    # Translate to sensor-to-node index mapping
    sensorToNodeIdx = []
    for idxNode, nSensors in enumerate(nSensorPerNode):
        sensorToNodeIdx += [idxNode] * nSensors

    return sensorToNodeIdx


def plot_metrics_results(
        metrics: dict,
        sensorToNodeIdx: list,
        metricsToPlot: list = ['snr', 'stoi'],
        testType: str = 'add_useless_mics',
    ):
    """
    Plot speech enhancement metrics results from the
    `tests.useless_microphones` script.
    
    Parameters
    ----------
    metrics : dict
        Dictionary containing the results.
    sensorToNodeIdx : list
        List containing the sensor-to-node index mapping for each test case.
    metricsToPlot : list, optional
        List of metrics to plot. Default is ['snr', 'stoi'].
    testType : str, optional
        Type of test. Default is 'add_useless_mics'.
    
    Returns
    ----------
    figs : list
        List containing the figures.
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """

    # Infer number of nodes
    nNodes = np.amax(sensorToNodeIdx[0]) + 1

    # Plot metrics
    figs = []
    if testType == 'render_mics_useless':
        raise ValueError('Not implemented yet.')
    
    elif testType == 'add_useless_mics':
        for k in range(nNodes):
            print(f'Plotting metrics for node {k + 1}...')
            # Order metrics
            metricsOut = dict([
                (metricKey, dict([
                    (testKey, None) for testKey in metrics.keys()
                ])) for metricKey in metricsToPlot
            ])
            # Get metrics for node `k`
            for testKey, value in metrics.items():
                for metricStr in metricsToPlot:
                    metricsOut[metricStr][testKey] =\
                        getattr(value, metricStr)[f'Node{k + 1}']

            # Plot metrics
            fig = plot_metrics(metricsOut, testType)
            fig.set_label(f'aum_metrics_k{k + 1}')
            fig.suptitle(f'Node {k + 1}')
            figs.append(fig)
            plt.close(fig=fig)
    
    return figs


def plot_metrics(metrics: dict, testType: str):

    nTests = len(metrics[list(metrics.keys())[0]])
    nMetrics = len(metrics.keys())

    # Build x-ticks labels
    if testType == 'render_mics_useless':
        pass  # TODO
    elif testType == 'add_useless_mics':
        xTicksLabels = [f'{ii} AUMs' for ii in range(nTests)]

    fig, axes = plt.subplots(2, nMetrics)
    fig.set_size_inches(8.5, 3.5)
    for idxMetric, (mKey, mVal) in enumerate(metrics.items()):
        # Regular DANSE filters (with fusion)
        for ii in range(nTests):
            currMetrics = mVal[list(mVal.keys())[ii]]
            axes[0, idxMetric].bar(ii, currMetrics.after)
        axes[0, idxMetric].set_xticks(range(nTests))
        axes[0, idxMetric].set_xticklabels(xTicksLabels)
        axes[0, idxMetric].grid()
        axes[0, idxMetric].set_title(f'{mKey.upper()} - Regular DANSE')
        # No-fusion DANSE filters ("centralized"-ish)
        for ii in range(nTests):
            currMetrics = mVal[list(mVal.keys())[ii]]
            axes[1, idxMetric].bar(ii, currMetrics.afterCentr)
        axes[1, idxMetric].set_xticks(range(nTests))
        axes[1, idxMetric].set_xticklabels(xTicksLabels)
        axes[1, idxMetric].grid()
        axes[1, idxMetric].set_title(f'{mKey.upper()} - No-fusion DANSE')
    plt.tight_layout()

    return fig


def plot_filtnorm_results(
        res: dict,
        sensorToNodeIdx: list,
        testType: str = 'add_useless_mics',
    ):
    """
    Plot filter norm results from the `tests.useless_microphones` script.
    
    Parameters
    ----------
    res : dict
        Dictionary containing the results.
    sensorToNodeIdx : list
        List containing the sensor-to-node index mapping for each test case.
    testType : str, optional
        Type of test. Default is 'add_useless_mics'.
    
    Returns
    ----------
    figs : list
        List containing the figures.
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """
    # Infer number of nodes
    nNodes = np.amax(sensorToNodeIdx[0]) + 1

    if testType == 'render_mics_useless':
        # Derive number of microphones rendered useless in each entry of `res`
        # using the key of the entry.
        nUselessMics = np.array(
            [int(len(re.findall(r'\d+', k))) - 1 for k in list(res.keys())]
        )
        # Select results for the case where only `nUselessMicsToPlot`
        # microphone(s) is (are) rendered useless.
        res = {k: v for ii, (k, v) in enumerate(res.items())\
            if nUselessMics[ii] == N_USELESS_MICS_TO_PLOT}

    # Plot results
    figs = []
    for idx, (key, resCurr) in enumerate(res.items()):
        # Inform user
        print(f'Plotting filter norm results for "{key}" ({idx+1}/{len(res.keys())})...')
        # Get y-axis limits
        yLim = np.array([
            np.amin([np.amin(resCurr[ii]) for ii in range(len(resCurr))]),
            np.amax([np.amax(resCurr[ii]) for ii in range(len(resCurr))])
        ])
        yLim = yLim + np.array([-1, 1]) * 0.1 * np.diff(yLim)
        
        if testType == 'render_mics_useless':
            # Get index of the microphone rendered useless
            uselessMicIdx = np.array(
                [int(x) for x in re.findall(r'\d+', key)[-N_USELESS_MICS_TO_PLOT:]]
            )
            # Get label snippet for the figure
            labSnippet = 'uselessMicIdx' + str(uselessMicIdx)
            nAddedMics = None  # Not used
        
        elif testType == 'add_useless_mics':
            # Current number of added mics
            nAddedMics = int(re.findall(r'\d+', key)[-1])
            # Get label snippet for the figure
            labSnippet = 'nAddedMics' + str(nAddedMics + 1)
            uselessMicIdx = None  # Not used
        
        for ii in range(len(resCurr)):
            fig = plt.figure()
            if ii < nNodes:
                legType = 'danse'
                currNodeIndex = ii
                ti = f'DANSE, node $k={currNodeIndex + 1}$'
                figLab = f'{labSnippet}_filtNorms_danse_k{currNodeIndex + 1}'
            else:
                legType = 'centr'
                currNodeIndex = ii - nNodes
                ti = f'Centr., node $k={currNodeIndex + 1}$'
                figLab = f'{labSnippet}_filtNorms_centr_k{currNodeIndex + 1}'
            fig.set_label(figLab)
            leg, colors = get_legend_and_colors(
                legType,
                sensorToNodeIdx[idx],
                currNodeIndex,
                indices=uselessMicIdx,
                nAddedMics=nAddedMics,
                testType=testType,
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
        indices: list = None,
        nAddedMics: int = None,
        testType: str = 'add_useless_mics',
    ):
    """
    Build legend for the results of one subfolder for the test type
    "add useless mics".
    """
    # Ensure sensor-to-node index mapping is a numpy array
    sensorToNodeIdx = np.array(sensorToNodeIdx)

    if testType == 'add_useless_mics':
        # Infer indices of useless microphones from the number of useless
        # microphones added.
        indicesNewNode = list(np.diff(sensorToNodeIdx).nonzero()[0])
        indices = [list(np.linspace(
            start=(i - nAddedMics + 1),
            stop=i,
            num=nAddedMics,
            dtype=int
        )) for i in indicesNewNode]
        indices = [item for sublist in indices for item in sublist]  # flatten
        # if nAddedMics > 1:
        indices += list(np.linspace(
            start=len(sensorToNodeIdx) - nAddedMics,
            stop=len(sensorToNodeIdx) - 1,
            num=nAddedMics,
            dtype=int
        ))  # add last index (indices)
    elif testType == 'render_mics_useless':
        pass  # do nothing -- use indices passed as argument

    # Build legend
    if legType == 'danse':
        # Infer indices of useless microphones from the number of useless
        # microphones added.
        leg = get_danse_legend(sensorToNodeIdx, nodeIndex, indices)
    elif legType == 'centr':
        leg = get_centr_legend(sensorToNodeIdx, nodeIndex, indices)

    # Build colors
    colors = build_line_colors(leg)

    return leg, colors


def build_line_colors(leg):
    """Build colors for the lines in the plots."""

    def _get_node_index(entry):
        """Extract node index from legend entry."""
        return int(re.findall(r'\d+', entry)[0]) - 1
    
    def _hex_to_rgb(hex):
        """Convert hex color to rgb."""
        hex = hex.lstrip('#')
        return tuple(int(hex[i:i+2], 16) / 255 for i in (0, 2, 4))
    
    uselessMicCount = 0
    legIndicesUselessMic = np.where(['useless' in entry for entry in leg])[0]
    # Infer nodes indices from legend entries
    nodeIndexPerEntry = [_get_node_index(entry) for entry in leg]
    nNodes = len(np.unique(nodeIndexPerEntry))
    # Get number of entries per node
    nEntriesPerNode = np.array([
        len(np.where(np.array(nodeIndexPerEntry) == ii)[0]) for ii in range(nNodes)
    ])
    # Define base colors (one per node)
    baseColors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:nNodes]
    
    colors = []
    counterPerNode = np.zeros(nNodes)
    for ii, entry in enumerate(leg):
        if 'useless' in entry:
            colors.append(str(
                np.round(
                    1 - (uselessMicCount + 1) / len(legIndicesUselessMic),
                    1
                )
            ))
            uselessMicCount += 1
        else:
            # Current node index
            currNodeIndex = _get_node_index(entry)
            # Get base color for current node
            currBaseColor = _hex_to_rgb(baseColors[currNodeIndex])
            # Add transparency
            alpha = 1 - counterPerNode[currNodeIndex] / nEntriesPerNode[currNodeIndex]
            currBaseColor = tuple(np.append(currBaseColor, alpha))
            colors.append(currBaseColor)
            counterPerNode[currNodeIndex] += 1

    return colors


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
        leg.append(f'Local ($k={nodeIndex + 1}$) $m={idxSensor + 1}/{nLocalSensors}$')
        
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
            leg.append(f'Local ($k={nodeIndex + 1}$) $m={idxSensor + 1}/{sum(sensorToNodeIdx == nodeIndex)}$')
        else:
            # Get local index of current sensor
            idxLocalSensor = np.where(np.where(sensorToNodeIdx == idxNode)[0] == idxSensor)[0][0]
            leg.append(f'Remote ($k={idxNode + 1}$) $m={idxLocalSensor + 1}/{len(np.where(sensorToNodeIdx == idxNode)[0])}$')

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