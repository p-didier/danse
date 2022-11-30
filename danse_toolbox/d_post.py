# Post-processing functions and scripts for, e.g.,
# visualizing DANSE outputs.
#
# ~created on 20.10.2022 by Paul Didier
import sys
import time
import copy
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pathlib import Path, PurePath
import pyroomacoustics as pra
from dataclasses import dataclass, fields
from siggen.classes import Node, WASNparameters
from danse_toolbox.d_classes import DANSEparameters, DANSEvariables
from danse_toolbox.d_eval import *
import danse_toolbox.dataclass_methods as met

@dataclass
class DANSEoutputs(DANSEparameters):
    """
    Dataclass to assemble all useful outputs
    of the DANSE algorithm.
    """
    initialised : bool = False
    # ^^^ turns to True when `self.from_variables()` is called
    
    def import_params(self, p: DANSEparameters):
        self.__dict__.update(p.__dict__)
    
    def from_variables(self, dv: DANSEvariables):
        """
        Selects useful output values from `DANSEvariables` object
        after DANSE processing.
        """

        # Inits
        self.TDdesiredSignals_c = None
        self.STFTDdesiredSignals_c = None
        self.TDdesiredSignals_l = None
        self.STFTDdesiredSignals_l = None

        # Original microphone signals
        self.micSignals = dv.yin
        # DANSE desired signal estimates
        self.TDdesiredSignals = dv.d
        self.STFTDdesiredSignals = dv.dhat
        if self.computeCentralised:
            # Centralised desired signal estimates
            self.TDdesiredSignals_c = dv.dCentr
            self.STFTDdesiredSignals_c = dv.dHatCentr
        if self.computeLocal:
            # Local desired signal estimates
            self.TDdesiredSignals_l = dv.dLocal
            self.STFTDdesiredSignals_l = dv.dHatLocal
        # SROs
        self.SROsEstimates = dv.SROsEstimates
        self.SROsResiduals = dv.SROsResiduals
        # Filters
        self.filters = dv.wTilde
        # Other useful things
        self.tStartForMetrics = dv.tStartForMetrics
        self.tStartForMetricsCentr = dv.tStartForMetricsCentr
        self.tStartForMetricsLocal = dv.tStartForMetricsLocal

        # Show initialised status
        self.initialised = True

        return self

    def check_init(self):
        """Check if object is correctly initialised."""
        if not self.initialised:
            return ValueError('The DANSEoutputs object is empty.')

    def save(self, foldername, light=False, exportType='pkl'):
        """Saves dataclass to file."""
        if light:
            mycls = copy.copy(self)
            for f in fields(mycls):
                if 'signal' in f.name.lower():
                    delattr(mycls, f)
            met.save(self, foldername, exportType=exportType)
        else:
            met.save(self, foldername, exportType=exportType)

    def load(self, foldername, dataType='pkl'):
        """Loads dataclass to Pickle archive in folder `foldername`."""
        return met.load(self, foldername, silent=True, dataType=dataType)

    def export_sounds(self, wasn, exportFolder):
        self.check_init()  # check if object is correctly initialised
        export_sounds(self, wasn, exportFolder)

    def plot_perf(self, wasn, exportFolder):
        """Plots DANSE performance."""
        self.check_init()  # check if object is correctly initialised
        self.metrics = compute_metrics(self, wasn)
        figStatic, figDynamic = plot_metrics(self)
        figStatic.savefig(f'{exportFolder}/metrics.png')
        figStatic.savefig(f'{exportFolder}/metrics.pdf')
        if figDynamic is not None:
            figDynamic.savefig(f'{exportFolder}/metrics_dyn.png')
            figDynamic.savefig(f'{exportFolder}/metrics_dyn.pdf')

    def plot_sigs(self, wasn, exportFolder):
        """Plots signals before/after DANSE."""
        figs = plot_signals_all_nodes(self, wasn)
        for k in range(len(figs)):
            figs[k].savefig(f'{exportFolder}/sigs_node{k+1}.png')


def compute_metrics(
        out: DANSEoutputs,
        wasn: list[Node]) -> EnhancementMeasures:
    """
    Compute and store evaluation metrics after signal enhancement.

    Parameters
    ----------
    out : `DANSEoutputs` object
        DANSE run outputs.
    wasn : list of `Node` objects
        WASN under consideration.
    folder : str
        Folder where to create the "wav" folder where to export files.
    
    Returns
    -------
    measures : `EnhancementMeasures` object
        The enhancement metrics.
    """

    def _ndict(N):
        """Initialize a 'node' dictionary."""
        return dict([(key, []) for key in [f'Node{n+1}' for n in range(N)]])

    # Initialisations
    startIdx = np.zeros(out.nNodes, dtype=int)
    startIdxCentr = np.zeros(out.nNodes, dtype=int)
    startIdxLocal = np.zeros(out.nNodes, dtype=int)
    snr = _ndict(out.nNodes)  # Unweighted SNR
    fwSNRseg = _ndict(out.nNodes)  # Frequency-weighted segmental SNR
    stoi = _ndict(out.nNodes)  # (extended) Short-Time Objective Intelligibility
    pesq = _ndict(out.nNodes)  # Perceptual Evaluation of Speech Quality
    tStart = time.perf_counter()
    for k in range(out.nNodes):
        # Derive starting sample for metrics computations
        startIdx[k] = int(np.floor(out.tStartForMetrics[k] * wasn[k].fs))
        print(f"Node {k+1}: computing speech enhancement metrics from the {startIdx[k] + 1}'th sample on (t_start = {out.tStartForMetrics[k]} s --> avoid bias due to initial filters guesses in first iterations)...")
        print(f'Computing signal enhancement evaluation metrics for node {k + 1}/{out.nNodes} (sensor {out.referenceSensor + 1}/{wasn[k].nSensors})...')

        enhan_c, enhan_l = None, None
        if out.computeCentralised:
            startIdxCentr[k] = int(np.floor(
                out.tStartForMetricsCentr[k] * wasn[k].fs
            ))
            print(f"Node {k+1}: computing speech enhancement metrics for CENTRALISED PROCESSING from the {startIdxCentr[k] + 1}'th sample on (t_start = {out.tStartForMetricsCentr[k]} s).")
            enhan_c = out.TDdesiredSignals_c[startIdxCentr[k]:, k]
        if out.computeLocal:
            startIdxLocal[k] = int(np.floor(
                out.tStartForMetricsLocal[k] * wasn[k].fs
            ))
            print(f"Node {k+1}: computing speech enhancement metrics for LOCAL PROCESSING from the {startIdxLocal[k] + 1}'th sample on (t_start = {out.tStartForMetricsLocal[k]} s).")
            enhan_l = out.TDdesiredSignals_l[startIdxLocal[k]:, k]

    # TOFIX: TODO: the `clean` should also start at `startIdxCentr` and `startIdxLocal`
    # --> give `startIdx`, `startIdxCentr`, and `startIdxLocal` as inputs to `get_metrics()`

        out0, out1, out2, out3 = get_metrics(
            # Clean speech mixture (desired signal)
            clean=wasn[k].cleanspeechCombined[startIdx[k]:],
            # Microphone signals
            noisy=wasn[k].data[startIdx[k]:, out.referenceSensor],
            # DANSE outputs (desired signal estimates)
            enhan=out.TDdesiredSignals[startIdx[k]:, k],
            # Centralised desired signal estimates
            enhan_c=enhan_c,
            # Local desired signal estimates
            enhan_l=enhan_l,
            #
            fs=wasn[k].fs,
            VAD=wasn[k].vadCombined[startIdx[k]:],
            dynamic=out.dynMetrics,
            gamma=out.gammafwSNRseg,
            fLen=out.frameLenfwSNRseg
        )

        snr[f'Node{k + 1}'] = out0
        fwSNRseg[f'Node{k + 1}'] = out1
        stoi[f'Node{k + 1}'] = out2
        pesq[f'Node{k + 1}'] = out3

    print(f'All signal enhancement evaluation metrics computed in {np.round(time.perf_counter() - tStart, 3)} s.')

    # Group measures into EnhancementMeasures object
    metrics = EnhancementMeasures(
        fwSNRseg=fwSNRseg, stoi=stoi, pesq=pesq, snr=snr, startIdx=startIdx
    )

    return metrics

def plot_metrics(out: DANSEoutputs):
    """
    Visualize evaluation metrics.

    Parameters
    ----------
    out : `DANSEoutputs` object
        DANSE outputs.
    """

    # Useful variables
    barWidth = 1
    
    fig1 = plt.figure(figsize=(12,3))
    ax = fig1.add_subplot(1, 4, 1)   # Unweighted SNR
    metrics_subplot(out.nNodes, ax, barWidth, out.metrics.snr)
    ax.set(title='SNR', ylabel='[dB]')
    #
    ax = fig1.add_subplot(1, 4, 2)   # fwSNRseg
    metrics_subplot(out.nNodes, ax, barWidth, out.metrics.fwSNRseg)
    ax.set(title='fwSNRseg', ylabel='[dB]')
    #
    ax = fig1.add_subplot(1, 4, 3)   # STOI
    metrics_subplot(out.nNodes, ax, barWidth, out.metrics.stoi)
    ax.set(title='eSTOI')
    ax.set_ylim([0, 1])
    #
    ax = fig1.add_subplot(1, 4, 4)   # PESQ
    metrics_subplot(out.nNodes, ax, barWidth, out.metrics.pesq)
    ax.set(title='PESQ')
    ax.legend(bbox_to_anchor=(1, 0), loc="lower left")

    fig1.suptitle("Speech enhancement metrics")
    plt.tight_layout()

    # Check where dynamic metrics were computed
    flagsDynMetrics = np.zeros(len(fields(out.metrics)), dtype=bool)
    for ii, field in enumerate(fields(out.metrics)):
        if type(field) == dict:
            if getattr(out.metrics, field.name)['Node1'].dynamicFlag:
                flagsDynMetrics[ii] = True

    nDynMetrics = np.sum(flagsDynMetrics)

    if nDynMetrics > 0:
        # Prepare subplots for dynamic metrics
        if nDynMetrics < 4:
            nRows, nCols = 1, nDynMetrics
        else:
            nRows, nCols = 2, int(np.ceil(nDynMetrics / 2))
        fig2, axes = plt.subplots(nRows, nCols)
        fig2.set_figheight(2.5 * nRows)
        fig2.set_figwidth(5 * nCols)
        axes = axes.flatten()   # flatten axes array for easy indexing
        
        # Select dictionary elements
        dynMetricsNames = [fields(out.metrics)[ii].name\
            for ii in range(len(fields(out.metrics))) if flagsDynMetrics[ii]
        ]
        dynMetrics = [getattr(out.metrics, n) for n in dynMetricsNames]

        # Plot
        for ii, dynMetric in enumerate(dynMetrics):
            for nodeRef, value in dynMetric.items():        # loop over nodes
                metric = value.dynamicMetric
                idxColor = int(nodeRef[-1]) - 1
                axes[ii].plot(
                    metric.timeStamps,
                    metric.before,
                    color=f'C{idxColor}',
                    linestyle='--',
                    label=f'{nodeRef}: Before'
                )
                axes[ii].plot(
                    metric.timeStamps,
                    metric.after,
                    color=f'C{idxColor}',
                    linestyle='-',
                    label=f'{nodeRef}: After'
                )
            axes[ii].grid()
            axes[ii].set_title(dynMetricsNames[ii])
            if ii == 0:
                axes[ii].legend(loc='lower left', fontsize=8)
            axes[ii].set_xlabel('$t$ [s]')  
        plt.tight_layout()
        fig2.suptitle("Dynamic speech enhancement metrics")
    else:
        fig2 = None

    return fig1, fig2


def metrics_subplot(numNodes, ax, barWidth, data):
    """Helper function for <Results.plot_enhancement_metrics()>.
    
    Parameters
    ----------
    numNodes : int
        Number of nodes in network.
    ax : Axes handle
        Axes handle to plot on.
    barWidth : float
        Width of bars for bar plot.
    data : dict of np.ndarrays of floats /or/ dict 
            of np.ndarrays of [3 x 1] lists of floats
        Speech enhancement metric(s) per node.
    """

    flagZeroBar = False  # flag for plotting a horizontal line at `metric = 0`

    # Columns count
    baseCount = 2
    if data['Node1'].afterCentr != 0.:
        baseCount += 1
    if data['Node1'].afterLocal != 0.:
        baseCount += 1
    widthFact = baseCount + 1
    colShifts = np.arange(start=1-baseCount, stop=baseCount, step=2)

    delta = barWidth / (2 * widthFact)

    for idxNode in range(numNodes):
        if idxNode == 0:    # only add legend labels to first node
            ax.bar(
                idxNode + colShifts[0] * delta,
                data[f'Node{idxNode + 1}'].before,
                width=barWidth / widthFact,
                color='C0',
                edgecolor='k',
                label='Before'
            )
            ax.bar(
                idxNode + colShifts[1] * delta,
                data[f'Node{idxNode + 1}'].after,
                width=barWidth / widthFact,
                color='C1',
                edgecolor='k',
                label='After'
            )
            if data['Node1'].afterCentr != 0.:
                ax.bar(
                    idxNode + colShifts[2] * delta,
                    data[f'Node{idxNode + 1}'].afterCentr,
                    width=barWidth / widthFact,
                    color='C2',
                    edgecolor='k',
                    label='After (centralised)'
                )
            if data['Node1'].afterLocal != 0.:
                ax.bar(
                    idxNode + colShifts[3] *  delta,
                    data[f'Node{idxNode + 1}'].afterLocal,
                    width=barWidth / widthFact,
                    color='C3',
                    edgecolor='k',
                    label='After (local)'
                )
        else:
            ax.bar(
                idxNode + colShifts[0] * delta,
                data[f'Node{idxNode + 1}'].before,
                width=barWidth / widthFact,
                color='C0',
                edgecolor='k'
            )
            ax.bar(
                idxNode + colShifts[1] * delta,
                data[f'Node{idxNode + 1}'].after,
                width=barWidth / widthFact,
                color='C1',
                edgecolor='k'
            )
            if data['Node1'].afterCentr != 0.:
                ax.bar(
                    idxNode + colShifts[2] * delta,
                    data[f'Node{idxNode + 1}'].afterCentr,
                    width=barWidth / widthFact,
                    color='C2',
                    edgecolor='k',
                )
            if data['Node1'].afterLocal != 0.:
                ax.bar(
                    idxNode + colShifts[3] * delta,
                    data[f'Node{idxNode + 1}'].afterLocal,
                    width=barWidth / widthFact,
                    color='C3',
                    edgecolor='k',
                )

        if data[f'Node{idxNode + 1}'].after < 0 or\
            data[f'Node{idxNode + 1}'].before < 0:
            flagZeroBar = True
    plt.xticks(
        np.arange(numNodes),
        [f'N{ii + 1}' for ii in range(numNodes)],
        fontsize=8
    )
    ax.tick_params(axis='x', labelrotation=90)
    if flagZeroBar:
        ax.hlines(  # plot horizontal line at `metric = 0`
            0,
            - barWidth/2,
            numNodes - 1 + barWidth/2,
            colors='k',
            linestyles='dashed'
        )


def export_sounds(out: DANSEoutputs, wasn: list[Node], folder):
    """
    Exports the enhanced, noisy, and desired signals as WAV files.

    Parameters
    ----------
    out : `DANSEoutputs` object
        DANSE outputs (signals, etc.)
    wasn : list of `Node` objects
        WASN under consideration.
    folder : str
        Folder where to create the "wav" folder where to export files.
    """

    folderShort = met.shorten_path(folder)
    # Check path validity
    if not Path(f'{folder}/wav').is_dir():
        Path(f'{folder}/wav').mkdir()
        print(f'Created .wav export folder ".../{folderShort}/wav".')
    for k in range(len(wasn)):
        data = normalize_toint16(wasn[k].data)
        wavfile.write(
            f'{folder}/wav/noisy_N{k + 1}_Sref{out.referenceSensor + 1}.wav',
            int(wasn[k].fs), data
        )
        #
        data = normalize_toint16(wasn[k].cleanspeech)
        wavfile.write(
            f'{folder}/wav/desired_N{k + 1}_Sref{out.referenceSensor + 1}.wav',
            int(wasn[k].fs), data
        )
        # vvv if enhancement has been performed
        if len(out.TDdesiredSignals[:, k]) > 0:
            data = normalize_toint16(out.TDdesiredSignals[:, k])
            wavfile.write(
                f'{folder}/wav/enhanced_N{k + 1}.wav',
                int(wasn[k].fs), data
            )
        # vvv if enhancement has been performed and centralised estimate computed
        if out.computeCentralised:
            if len(out.TDdesiredSignals_c[:, k]) > 0:
                data = normalize_toint16(out.TDdesiredSignals_c[:, k])
                wavfile.write(
                    f'{folder}/wav/enhancedCentr_N{k + 1}.wav',
                    int(wasn[k].fs), data
                )
        # vvv if enhancement has been performed and local estimate computed
        if out.computeLocal:
            if len(out.TDdesiredSignals_l[:, k]) > 0:
                data = normalize_toint16(out.TDdesiredSignals_l[:, k])
                wavfile.write(
                    f'{folder}/wav/enhancedLocal_N{k + 1}.wav',
                    int(wasn[k].fs), data
                )
    print(f'Signals exported in folder ".../{folderShort}/wav".')


def plot_asc(
        asc: pra.room.ShoeBox,
        p: WASNparameters,
        folder=None
    ):
    """
    Plots an acoustic scenario nicely.

    Parameters
    ----------
    asc : `pyroomacoustics.room.ShoeBox` object
        Acoustic scenario under consideration.
    p : `WASNparameters` object
        WASN parameters.
    folder : str
        Folder where to export figure (if not `None`).
    """

    # Determine appropriate node radius for ASC subplots
    nodeRadius = 0
    for k in range(p.nNodes):
        allIndices = np.arange(sum(p.nSensorPerNode))
        sensorIndices = allIndices[p.sensorToNodeIndices == k]
        if len(sensorIndices) > 1:
            meanpos = np.mean(asc.mic_array.R[:, sensorIndices], axis=1)
            curr = np.amax(asc.mic_array.R[:, sensorIndices] - \
                meanpos[:, np.newaxis])
        else:
            curr = 0.1
        if curr > nodeRadius:
            nodeRadius = copy.copy(curr)

    fig = plt.figure()
    fig.set_size_inches(6.5, 3.5)

    ax = fig.add_subplot(1, 2, 1)
    plot_side_room(
        ax,
        p.rd[:2],
        np.array([ii.position[:2] for ii in asc.sources[:p.nDesiredSources]]), 
        np.array([ii.position[:2] for ii in asc.sources[-p.nNoiseSources:]]), 
        asc.mic_array.R[:2, :].T,
        p.sensorToNodeIndices,
        dotted=p.t60==0,
        showLegend=False,
        nodeRadius=nodeRadius
    )
    ax.set(xlabel='$x$ [m]', ylabel='$y$ [m]', title='Top view')
    #
    ax = fig.add_subplot(1, 2, 2)
    plot_side_room(
        ax,
        p.rd[-2:],
        np.array([ii.position[-2:] for ii in asc.sources[:p.nDesiredSources]]), 
        np.array([ii.position[-2:] for ii in asc.sources[-p.nNoiseSources:]]), 
        asc.mic_array.R[-2:, :].T,
        p.sensorToNodeIndices,
        dotted=p.t60==0,
        showLegend=False,
        nodeRadius=nodeRadius
    )
    ax.set(xlabel='$y$ [m]', ylabel='$z$ [m]', title='Side view')

    plt.tight_layout()

    # Export
    if folder is not None:
        fig.savefig(f'{folder}/asc.png')
        fig.savefig(f'{folder}/asc.pdf')
        
    return fig


def plot_side_room(
        ax, rd2D, rs, rn, r,
        micToNodeTags,
        scatsize=20,
        dotted=False,
        showLegend=True,
        nodeRadius=None
    ):
    """Plots a 2-D room side, showing the positions of
    sources and nodes inside of it.
    Parameters
    ----------
    ax : Axes handle
        Axes handle to plot on.
    rd2D : [2 x 1] list
        2-D room dimensions [m].
    rs : [Ns x 2] np.ndarray (real)
        Desired (speech) source(s) coordinates [m]. 
    rn : [Nn x 2] np.ndarray (real)
        Noise source(s) coordinates [m]. 
    r : [N x 2] np.ndarray (real)
        Sensor(s) coordinates [m].
    sensorToNodeTags : [N x 1] np.ndarray (int)
        Tags relating each sensor to a node number (>=1).
    scatsize : float
        Scatter plot marker size.
    dotted : bool
        If True, use dotted lines. Else, use solid lines (default).
    showLegend : bool
        If True, show legend.
    nodeRadius : float
        Pre-defined node radius. 
    """

    numNodes = len(np.unique(micToNodeTags))
    numSensors = len(micToNodeTags)
    
    plot_room2D(ax, rd2D, dotted)
    # Desired sources
    for m in range(rs.shape[0]):
        ax.scatter(
            rs[m,0],
            rs[m,1],
            s=2*scatsize,
            c='lime',
            marker='d',
            edgecolor='k'
        )
    # Noise sources
    for m in range(rn.shape[0]):
        ax.scatter(
            rn[m,0],
            rn[m,1],
            s=2*scatsize,
            c='red',
            marker='P',
            edgecolor='k'
        )
    # Nodes and sensors
    circHandles = []
    leg = []
    for k in range(numNodes):
        allIndices = np.arange(numSensors)
        micsIdx = allIndices[micToNodeTags == k]
        for m in micsIdx:
            ax.scatter(
                r[m,0],
                r[m,1],
                s=scatsize,
                c=f'C{k}',
                edgecolors='black',
                marker='o'
            )
        # Draw circle around node
        if nodeRadius is not None:
            radius = nodeRadius
        else:
            radius = np.amax(r[micsIdx, :] - np.mean(r[micsIdx, :], axis=0))
        circ = plt.Circle(
            (np.mean(r[micsIdx,0]), np.mean(r[micsIdx,1])),
            radius * 2,
            color=f'C{k}',
            fill=False
        )
        circHandles.append(circ)
        leg.append(f'Node {k + 1}')
        # Add label
        ax.text(np.mean(r[micsIdx,0]) + 1.5*radius,
                np.mean(r[micsIdx,1]) + 1.5*radius,
                f'$\\mathbf{{{k+1}}}$', c=f'C{k}')
        ax.add_patch(circ)
    ax.grid()
    ax.set_axisbelow(True)
    ax.axis('equal')
    if showLegend:
        nc = 1  # number of columbs in legend object
        if len(circHandles) >= 4:
            nc = 2
        ax.legend(circHandles, leg, loc='lower right', ncol=nc, mode='expand')


def plot_room2D(ax, rd, dotted=False):
    """Plots the edges of a rectangle in 2D on the axes <ax>
    
    Parameters
    ----------
    ax : matplotlib Axes object
        Axes object onto which the rectangle should be plotted.
    rd : [3 x 1] (or [1 x 3], or [2 x 1], or [1 x 2]) np.ndarray or list of float
        Room dimensions [m].
    dotted : bool
        If true, use dotted lines. Else, use solid lines (default).
    """
    fmt = 'k'
    if dotted:
        fmt += '--'
    ax.plot([rd[0],0], [0,0], fmt)
    ax.plot([0,0], [0,rd[1]], fmt)
    ax.plot([rd[0],rd[0]], [0,rd[1]], fmt)
    ax.plot([0,rd[0]], [rd[1],rd[1]], fmt)


def plot_signals_all_nodes(out: DANSEoutputs, wasn: list[Node]):
    """
    Plot DANSE output signals, comparing with inputs.

    Parameters
    ----------
    out : `DANSEoutputs` object
        DANSE run outputs.

    Returns
    -------
    figs : list of `matplotlib.figure.Figure` objects
        Figure handle for each node.
    """

    figs = []
    # Plot per node
    for k in range(out.nNodes):
        fig, axForTitle = plot_signals(
            node=wasn[k],
            win=out.winWOLAanalysis,
            ovlp=out.WOLAovlp
        )
        axForTitle.set_title(f'Node {k + 1}, ref. sensor (#{out.referenceSensor + 1})')
        plt.tight_layout()
        figs.append(fig)

    return figs


def plot_signals(node: Node, win, ovlp):
    """
    Creates a visual representation of the signals at a particular sensor.

    Parameters
    ----------
    node : `Node` object
        Node, containing all needed info (fs, signals, etc.).
    win : np.ndarry (float)
        STFT analysis window.
    ovlp : float
        Consecutive STFT window overlap [/100%].

    Returns
    -------
    fig : `matplotlib.figure.Figure` object
        Figure handle.
    """

    # Waveforms
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,2,1)
    delta = np.amax(np.abs(node.data))
    ax.plot(
        node.timeStamps,
        node.cleanspeechCombined,
        label='Desired'
    )
    ax.plot(
        node.timeStamps,
        node.vad * np.amax(node.cleanspeechCombined) * 1.1,
        'k-', label='VAD'
    )
    ax.plot(
        node.timeStamps,
        node.data[:, 0] - 2*delta,
        label='Noisy'
    )
    # Desired signal estimate waveform 
    ax.plot(
        node.timeStamps,
        node.enhancedData - 4*delta,
        label='Enhanced (global)'
    )
    currDelta = 4*delta
    if len(node.enhancedData_l) > 0:
        ax.plot(
            node.timeStamps,
            node.enhancedData_l - currDelta - 2*delta,
            label='Enhanced (local)'
        )
        currDelta += 2*delta
    if len(node.enhancedData_c) > 0:
        ax.plot(
            node.timeStamps,
            node.enhancedData_c - currDelta - 2*delta,
            label='Enhanced (centr.)'
        )
    ax.set_yticklabels([])
    ax.set(xlabel='$t$ [s]')
    ax.grid()
    plt.legend(loc=(0.01, 0.5), fontsize=8)

    # -------- STFTs --------
    # Number of subplot rows
    nRows = 3

    # Compute STFTs
    cleanSTFT, f, t = get_stft(node.cleanspeechCombined, node.fs, win, ovlp)
    noisySTFT, _, _ = get_stft(node.data[:, 0], node.fs, win, ovlp)
    enhanSTFT, _, _ = get_stft(node.enhancedData, node.fs, win, ovlp)
    if len(node.enhancedData_l) > 0:
        enhanSTFT_l, _, _ = get_stft(node.enhancedData_l, node.fs, win, ovlp)
        nRows += 1
    if len(node.enhancedData_c) > 0:
        enhanSTFT_c, _, _ = get_stft(node.enhancedData_c, node.fs, win, ovlp)
        nRows += 1
    
    # Get color plot limits
    limLow = 20 * np.log10(
        np.amin([np.amin(np.abs(cleanSTFT)), np.amin(np.abs(noisySTFT))])
    )
    # Ensures that pure silences do not bring the limit too low
    limLow = np.amax([-100, limLow])
    limHigh = 20 * np.log10(
        np.amax([np.amax(np.abs(cleanSTFT)), np.amax(np.abs(noisySTFT))])
    )

    # Plot spectrograms
    ax = fig.add_subplot(nRows,2,2)  # Wet desired signal
    axForTitle = copy.copy(ax)
    data = 20 * np.log10(np.abs(np.squeeze(cleanSTFT)))
    stft_subplot(ax, t, f, data, [limLow, limHigh], 'Desired')
    plt.xticks([])
    ax = fig.add_subplot(nRows,2,4)  # Sensor signals
    data = 20 * np.log10(np.abs(np.squeeze(noisySTFT)))
    stft_subplot(ax, t, f, data, [limLow, limHigh], 'Noisy')
    plt.xticks([])
    ax = fig.add_subplot(nRows,2,6)   # Enhanced signals (global)
    data = 20 * np.log10(np.abs(np.squeeze(enhanSTFT)))
    stft_subplot(ax, t, f, data, [limLow, limHigh], 'Global DANSE')
    currSubplotIdx = 6
    if len(node.enhancedData_l) > 0:    # Enhanced signals (local)
        ax = fig.add_subplot(nRows,2,currSubplotIdx + 2)
        data = 20 * np.log10(np.abs(np.squeeze(enhanSTFT_l)))
        stft_subplot(ax, t, f, data, [limLow, limHigh], 'Local est.')
        currSubplotIdx += 2
    if len(node.enhancedData_c) > 0:    # Enhanced signals (centralised)
        ax = fig.add_subplot(nRows,2,currSubplotIdx + 2)
        data = 20 * np.log10(np.abs(np.squeeze(enhanSTFT_c)))
        stft_subplot(ax, t, f, data, [limLow, limHigh], 'Centr. est.')
    ax.set(xlabel='$t$ [s]')

    return fig, axForTitle


def stft_subplot(ax, t, f, data, vlims, label=''):
    """Helper function for <Signals.plot_signals()>."""
    # Text boxes properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #
    mappable = ax.pcolormesh(t, f / 1e3, data, vmin=vlims[0], vmax=vlims[1], shading='auto')
    ax.set(ylabel='$f$ [kHz]')
    if label != '':
        ax.text(0.025, 0.9, label, fontsize=8, transform=ax.transAxes,
            verticalalignment='top', bbox=props)
    ax.yaxis.label.set_size(8)
    cb = plt.colorbar(mappable)
    return cb


def get_stft(x, fs, win, ovlp):
    """
    Derives time-domain signals' STFT representation
    given certain settings.

    Parameters
    ----------
    x : [N x C] np.ndarray (float)
        Time-domain signal(s).
    fs : int
        Sampling frequency [samples/s].
    settings : ProgramSettings object
        Settings (contains window, window length, overlap)

    Returns
    -------
    out : [Nf x Nt x C] np.ndarray (complex)
        STFT-domain signal(s).
    f : [Nf x C] np.ndarray (real)
        STFT frequency bins, per channel (because of different sampling rates).
    t : [Nt x 1] np.ndarray (real)
        STFT time frames.
    """

    if x.ndim == 1:
        x = x[:, np.newaxis]

    for channel in range(x.shape[-1]):

        fcurr, t, tmp = sig.stft(
            x[:, channel],
            fs=fs,
            window=win,
            nperseg=len(win),
            noverlap=int(ovlp * len(win)),
            return_onesided=True
        )
        if channel == 0:
            out = np.zeros(
                (tmp.shape[0], tmp.shape[1], x.shape[-1]), dtype=complex
            )
            f = np.zeros((tmp.shape[0], x.shape[-1]))
        out[:, :, channel] = tmp
        f[:, channel] = fcurr

    # Flatten array in case of single-channel data
    if x.shape[-1] == 1:
        f = np.array([i[0] for i in f])

    return out, f, t


def normalize_toint16(nparray):
    """Normalizes a NumPy array to integer 16.
    Parameters
    ----------
    nparray : np.ndarray
        Input array to be normalized.

    Returns
    ----------
    nparrayNormalized : np.ndarray
        Normalized array.
    """
    amplitude = np.iinfo(np.int16).max
    nparrayNormalized = (amplitude * nparray / \
        np.amax(nparray) * 0.5).astype(np.int16)  # 0.5 to avoid clipping
    return nparrayNormalized