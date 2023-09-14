# Basic functions necessary for the good functioning of DANSE.
# -- Mostly surrounding the most important functions in `d_core.py`.
#
# ~created on 19.10.2022 by Paul Didier

import copy
import warnings
import itertools
import collections
import numpy as np
import networkx as nx
from numba import njit
import scipy.signal as sig
import scipy.linalg as sla
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from siggen.classes import WASNparameters, Node, WASN
from danse_toolbox.d_eval import DynamicMetricsParameters

from itertools import cycle, islice


@dataclass
class DANSEeventInstant:
    t: float = 0.   # event time instant [s]
    nodes: np.ndarray = np.array([0])   # node(s) concerned
    type: list[str] = field(default_factory=list)   # event type
    bypassUpdate: list[bool] = field(default_factory=list)  
        # ^^^ if True, bypass filter update at that instant and for that node.
        # This value is adapted, e.g., depending on the node-updating strategy
        # chosen (may be set to True for certain updates if sequential DANSE
        # node-updating is used). 

    def __post_init__(self):
        self.nEvents = len(self.nodes)


@dataclass
class CohDriftParameters():
    """
    Dataclass containing the required parameters for the
    "Coherence drift" SRO estimation method.

    Attributes
    ----------
    alpha : float
        Exponential averaging constant.
    segLength : int 
        Number of DANSE filter updates per SRO estimation segment
    estEvery : int
        Estimate SRO every `estEvery` signal frames.
    startAfterNupdates : int 
        Minimum number of DANSE filter updates before first SRO estimation
    estimationMethod : str
        SRO estimation methods once frequency-wise estimates are obtained.
        Options: "gs" (golden section search in time domain [1]), 
                "mean" (similar to Online WACD implementation [2]),
                "ls" (least-squares estimate over frequency bins [3])
    alphaEps : float
        Residual SRO incrementation factor:
        $\\hat{\\varepsilon}^i = \\hat{\\varepsilon}^{i-1} + `alphaEps` * 
        \\Delta\\varepsilon^i$

    References
    ----------
    [1] Gburrek, Tobias, Joerg Schmalenstroeer, and Reinhold Haeb-Umbach.
        "On Synchronization of Wireless Acoustic Sensor Networks in the
        Presence of Time-Varying Sampling Rate Offsets and Speaker Changes."
        ICASSP 2022-2022 IEEE International Conference on Acoustics,
        Speech and Signal Processing (ICASSP). IEEE, 2022.
        
    [2] Chinaev, Aleksej, et al. "Online Estimation of Sampling Rate
        Offsets in Wireless Acoustic Sensor Networks with Packet Loss."
        2021 29th European Signal Processing Conference (EUSIPCO). IEEE, 2021.
        
    [3] Bahari, Mohamad Hasan, Alexander Bertrand, and Marc Moonen.
        "Blind sampling rate offset estimation for wireless acoustic sensor
        networks through weighted least-squares coherence drift estimation."
        IEEE/ACM Transactions on Audio, Speech, and Language Processing 25.3
        (2017): 674-686.
    """
    alpha : float = .95                 
    segLength : int = 10                # segment length: use phase angle
        # bw. values spaced by `segLength` signal frames to estimate the SRO.
    estEvery : int = 1                  # estimate SRO every `estEvery` frames.
    startAfterNups : int = 11       # only start estimating the SRO after 
        # `startAfterNupdates` signal frames.
    estimationMethod : str = 'gs'       # options: "gs" (golden section [1]),
                                        # "ls" (least-squares [3])
    alphaEps : float = .05              # residual SRO incrementation factor
    loop : str = 'closed'               # SRO estimation + compensation loop
        # - "closed": feedback loop, using SRO-compensated signals
        # - "open": no feedback, using SRO-uncompensated signals
    
@dataclass
class Hyperparameters:
    # Efficiency
    efficientSpSBC: bool = True     # if True, perform efficient sample-per-
        # sample broadcast by adapting the DANSE-events creation mechanism.
    # Other
    bypassUpdates: bool = False   # if True, do not update filters.

@dataclass
class PrintoutsAndPlotting:
    verbose: bool = True   # if True, enable print outs during DANSE.
    showWASNs : bool = False    # if True, shows a 3-D plot of the WASN every
        # time the topology changes (e.g., when a new tree is formed, or a new
        # root is set, in TI-DANSE).
    # Printouts
    printout_batch_updates: bool = True     # controls printouts of batch
                                            # updates.
    printout_profiler: bool = True      # controls printouts of Profiler.
    printout_eventsParser: bool = True      # controls printouts in
                                            # `events_parser()` function.
    printout_eventsParserNoBC: bool = False     # if True, do not print out the
                                                # broadcasts in event parser.
    printout_externalFilterUpdate: bool = True      # controls printouts of
                                                    # external filter updates.
    def __post_init__(self):
        # Adapt printouts depending on the verbosity
        self.printout_eventsParser = self.printout_eventsParser and self.verbose
        self.printout_eventsParserNoBC = self.printout_eventsParserNoBC and \
            self.verbose
        self.printout_profiler = self.printout_profiler and self.verbose
        self.printout_externalFilterUpdate = self.printout_externalFilterUpdate \
            and self.verbose

@dataclass
class DANSEparameters(Hyperparameters):
    """
    Parameters for the DANSE algorithm.

    References
    ----------
    - [1] Bertrand, A., & Moonen, M. (2010). Distributed adaptive node-specific
    signal estimation in fully connected sensor networks—Part I: Sequential
    node updating. IEEE Transactions on Signal Processing, 58(10), 5277-5291.

    - [2] Bertrand, A., & Moonen, M. (2010). Distributed adaptive node-specific
    signal estimation in fully connected sensor networks—Part II: Simultaneous
    and asynchronous node updating. IEEE Transactions on Signal Processing,
    58(10), 5292-5306.

    - [3] P. Didier, T. van Waterschoot, S. Doclo, and M. Moonen (2022).
    Sampling rate offset estimation and compensation for distributed adaptive 
    node-specific signal estimation in wireless acoustic sensor networks.
    arXiv preprint arXiv:2211.02489.

    - [4] A. Chinaev, P. Thüne, and G. Enzner (2021). Double-cross-correlation
    processing for blind sampling-rate and time-offset estimation. IEEE/ACM
    Transactions on Audio, Speech, and Language Processing, vol. 29,
    pp. 1881-1896.

    - [5] J. Szurley, A. Bertrand and M. Moonen, "Topology-Independent
    Distributed Adaptive Node-Specific Signal Estimation in Wireless
    Sensor Networks," in IEEE Transactions on Signal and Information
    Processing over Networks, vol. 3, no. 1, pp. 130-144, March 2017,
    doi: 10.1109/TSIPN.2016.2623095.
    """
    # --- General
    simType: str = 'batch'  # simulation type
        # - "batch": batch DANSE [1]
        # - "batch_wola_estimation": batch DANSE filter updates but WOLA-based
        #           ('online') estimation of the desired signals.
        # - "online": online DANSE
    maxBatchUpdates: int = 10   # maximum number of batch DANSE updates
        # (used iff `simType == 'batch'`)
    DFTsize: int = 1024     # DFT size
        # (used iff `simType == 'online'`)
    WOLAovlp: float = .5    # WOLA window overlap [*100%]
        # (used iff `simType == 'online'`)
    nodeUpdating: str = 'seq'   # node-updating strategy
        # - "seq": round-robin updating [1]
        # - "sim": simultaneous updating [2]
        # - "asy": asynchronous updating [2]
        # - "topo-indep_seq": round-robin updating for TI-DANSE [5]
        # - "topo-indep_sim": simultaneous updating for TI-DANSE [5]
        # - "topo-indep_asy": asynchronous updating for TI-DANSE [5]
    seqUpdateStartNodeIdx : int = 0  # index of node updating first.
        # Used iff `'seq' in nodeUpdating`.
    broadcastType: str = 'wholeChunk'    # type of broadcast
        # -- 'wholeChunk': chunks of compressed signals in time-domain,
        # -- 'fewSamples': T(z)-approximation of WOLA compression process.
        # broadcast L ≪ Ns samples at a time.
    broadcastLength: float = None  # [samples] length of broadcast
        # (used iff `broadcastType == 'fewSamples'`)
        # If None, is automatically set to 1 in __post_init__.
    winWOLAanalysisType: str = 'sqrthann'    # type of analysis window
        # - 'sqrthann': sqrt(hann)
        # - 'rect': rectangular window
        # - 'rect_normNs': rectangular window, normalized by WOLA overlap
    winWOLAsynthesisType: str = 'sqrthann'    # type of synthesis window
        # The valid values are the same as for `winWOLAanalysisType`.
    upTDfilterEvery: float = 1. # [s] duration of pause between two 
        # consecutive time-domain filter updates.
    noFusionAtSingleSensorNodes: bool = False  # if True, do not fuse the
        # DANSE filters at nodes with only one sensor, leave them as is.
    # ---- SROs
    compensateSROs: bool = False    # if True, compensate for SROs
    estimateSROs: str = 'Oracle'    # SRO estimation method.
        # If 'Oracle', no estimation: using oracle if `compensateSROs == True`.
        # If "CohDrift", use coherence-drift method (see [3]).
        # If "DXCPPhaT", use DXCP-PhaT method from [4] (implemented based on
        #   https://github.com/fgnt/asnsig).
    compensationStrategy: str = 'node-specific'    # type of SRO compensation
        # - 'node-specific': node-specific compensation
        # - 'network-wide': WASN-wide compensation
        #       ^^^ WASN-wide can only be used with TI-DANSE.
    cohDrift: CohDriftParameters = CohDriftParameters()
    # ---- General
    performGEVD: bool = False    # if True, perform GEVD
    GEVDrank: int = 1           # GEVD rank
    noExternalFilterRelaxation: bool = False  # if True, bypass external
        # filter updates.
    timeBtwExternalFiltUpdates: float = 0.  # [s] bw. external filter updates.
    alphaExternalFilters: float = 1.    # exponential averaging constant
        # for external filter _target_ update using internal DANSE filters.
    t_expAvg50pExternalFilters: float = 2.  # [s] Time in the past at which
        # the target external asynchronous node-updating DANSE filters are
        # weighted by 50% via exponential averaging to compute the external 
        # DANSE filters (not target).
    t_expAvg50p: float = 2.     # [s] Time in the past at which the spatial
        # covariance matrices values are weighted by 50% via
        # exponential averaging.
    forcedBeta: float = None    # if not None, use this value for the
        # forgetting factor beta.
    forcedBetaExternalFilters: float = None    # if not None, use this value
        # for the forgetting factor beta for external filters.
    filterInitType: str = 'selectFirstSensor'
        # type of complex filter vector initialization:
        # -- 'selectFirstSensor' == [1, 0, ..., 0]^T,
        # -- 'random' == random vector between -0.5 and 0.5
        # -- 'fixedValue' == [e, ..., e]^T with e = `filterInitFixedValue`.
        # -- 'selectFirstSensor_andFixedValue' == [1, e, ..., e]^T,
        #   with e = `filterInitFixedValue`.
    filterInitFixedValue: float = 0.
    # ---- Covariance matrices initialization parameters
    covMatInitType: str = 'fully_random'
        # type of complex covariance filter initialization:
        # -- 'fully_random' == all entries of the covariance matrix are random (but the same for each frequency)
        # -- 'eye' == identity matrix
        # -- 'eye_and_random' == identity matrix to which is added an all-random matrix
        # -- 'batch_estimates' == `Ryy = yy^H`, with `y` the entire (batch)
        #       signal, using the initial DANSE filters.
    covMatEyeInitScaling: float = 1.
        # Value by which to scale the diagonal terms of the initial covariance
        # matrices - used iff `'eye' in covMatInitType`.
    covMatRandomInitScaling: float = np.finfo(float).eps
        # Value by which to scale the [-1,1] normal distribution when
        # initializing the covariance matrices with random numbers.
    covMatSameInitForAllNodes: bool = True
        # If True, the same initial covariance matrices are used for all nodes.
        # Else, all nodes get a different initialization.
    covMatSameInitForAllFreqs: bool = True
        # If True, the same initial covariance matrices are used for all
        # frequency bins. Else, all bins get a different initialization.
    use1stFrameAsBasis: bool = False  # if True, use conditional SCM
        # updates, i.e., use the 1st available frame as basis for the 
        # subsequent exponential averaging.
    # ---- Plotting
    printoutsAndPlotting : PrintoutsAndPlotting = PrintoutsAndPlotting()
    # ---- Desired signal estimation
    desSigProcessingType: str = 'wola'  # processing scheme used to compute
        # the desired signal estimates: "wola": WOLA synthesis,
                                    # "conv": T(z)-approximation.
    computeCentralised: bool = False    # if True, compute centralised
        # estimate (using all microphone signals in network).
    computeLocal: bool = False  # if True, compute local estimate at each node.
    computeNoiseFree: bool = False  # if True, compute estimate with the
        # DANSE filters, but using noise-free signals (y_k's and z_q's).
        # TODO: TODO: TODO: TODO: TODO: TODO: TODO: 
    # ---- Metrics
    dynMetrics: DynamicMetricsParameters = DynamicMetricsParameters()
    gammafwSNRseg: float = 0.2  # gamma exponent for fwSNRseg
    frameLenfwSNRseg: float = 0.03  # [s] time window duration for fwSNRseg
    minNoSpeechDurEndUtterance: float = 0.2  # [s] min. duration of silence
        # at the end of an utterance (200 ms by default).
    startComputeMetricsAt: str = 'beginning_2nd_utterance'
        # ^^^ When to start computing the speech enhancement metrics.
        # Valid values:
        # -- 'beginning_2nd_utterance': start computing metrics at the
        # beginning of the 2nd utterance.
        # -- 'beginning_1st_utterance': start computing metrics at the
        # beginning of the 1st utterance.
        # -- 'end_1st_utterance': start computing metrics at the end of the
        # 1st utterance.
        # -- 'after_beginning_2nd_utterance_<x>': start computing metrics
        # `x` milliseconds after the beginning of the 2nd utterance.
        # -- 'after_<x><y>': start computing metrics `x` ms after the
        # beginning of the simulation if `y == 'ms'`, or `x` s after the
        # beginning of the simulation if `y == 's'`.
    endComputeMetricsAt: str = None    # When to stop computing the speech
        # enhancement metrics. Valid values: same as for
        # `startComputeMetricsAt`.
    # ---- TI-DANSE specific
    treeFormationAlgorithm: str = 'prim'    # algorithm to prune ad-hoc WASN
        # Valid values (from NetworkX toolbox): 'kruskal', 'prim', 'boruvka'.
            # NB: according to Paul Didier's testings from December 2022,
            # 'kruskal' and 'prim' are faster and more scalable than 'boruvka'.
    keepOriginalTree: bool = False  # if True, keep the original tree (do not
        # form new ones at every iteration).
    # ---- Debugging
    saveConditionNumber: bool = False   # if True, save condition numbers of
        # relevant covariance matrices.
    saveConditionNumberEvery: int = 999    # [iter] bw. condition number saves.
    wasnInfoInitiated: bool = False     # if True, the WASN info has been
        # initiated (i.e. the WASN has been created).
    startUpdatesAfterAtLeast: float = 0.   # [s] min. time before starting
        # the filter updates.
    
    def __post_init__(self):
        """Adapt some fields depending on the value of others, after 
        dataclass instance initialisation."""
        # Re-run post-init method on nested dataclasses
        self.printoutsAndPlotting.__post_init__()

        self.Ns = int(self.DFTsize * (1 - self.WOLAovlp))
        # Create windows
        if self.winWOLAanalysisType == 'sqrthann':
            self.winWOLAanalysis = np.sqrt(np.hanning(self.DFTsize))      # window
        elif self.winWOLAanalysisType == 'rect':
            self.winWOLAanalysis = np.ones(self.DFTsize)
        elif self.winWOLAanalysisType == 'rect_normNs':
            self.winWOLAanalysis = np.ones(self.DFTsize) * np.sqrt(self.WOLAovlp)
        else:
            raise ValueError(f'Unknown analysis window type: {self.winWOLAanalysisType}')
        if self.winWOLAsynthesisType == 'sqrthann':
            self.winWOLAsynthesis = np.sqrt(np.hanning(self.DFTsize))
        elif self.winWOLAsynthesisType == 'rect':
            self.winWOLAsynthesis = np.ones(self.DFTsize)
        elif self.winWOLAsynthesisType == 'rect_normNs':
            self.winWOLAsynthesis = np.ones(self.DFTsize) * np.sqrt(self.WOLAovlp)
        else:
            raise ValueError(f'Unknown synthesis window type: {self.winWOLAsynthesisType}')
        #
        self.normFactWOLA: float = self.Ns / sum(self.winWOLAanalysis)  # WOLA normalization factor -- see, e.g., https://de.mathworks.com/help/dsp/ref/dsp.istft.html#mw_6c14e754-02d8-4416-b20c-776b065aef98
        # ---- T(z)-approximation | Sample-wise broadcasts
        if self.broadcastType == 'wholeChunk':
            self.broadcastLength = self.Ns
        elif self.broadcastType == 'fewSamples' and self.broadcastLength is None:
            self.broadcastLength = 1
        elif self.broadcastType == 'fewSamples' and self.broadcastLength is not None:
            if self.broadcastLength > self.Ns:
                raise ValueError(f'Broadcast length ({self.broadcastLength}) cannot be larger than the WOLA frame size ({self.Ns}).')
            else:
                if self.Ns % self.broadcastLength != 0:
                    possibleDivisors = get_divisors(self.Ns)
                    raise ValueError(f'Broadcast length ({self.broadcastLength}) must be a divisor of the WOLA frame size ({self.Ns}). Possible divisors: {list(possibleDivisors)}.')
        if self.estimateSROs not in ['Oracle', 'CohDrift', 'DXCPPhaT']:
            raise ValueError(f'The field "estimateSROs" accepts values ["Oracle", "CohDrift", "DXCPPhaT"]. Current value: "{self.estimateSROs}".')
        if self.noExternalFilterRelaxation:
            # no differentiating between external and internal filters
            # (bypass-studies).
            self.timeBtwExternalFiltUpdates = 0.
        if self.simType not in ['batch', 'online']:
            raise ValueError(f'Unknown simulation type: {self.simType}. Valid values: ["batch", "online"].')
        self.compensationStrategy = self.compensationStrategy.lower()
        if self.compensationStrategy not in ['network-wide', 'node-specific']:
            raise ValueError(f'Unknown compensation strategy: {self.compensationStrategy}. Valid values: ["network-wide", "node-specific"].')

    def get_wasn_info(self, wasnParams: WASNparameters):
        """Adds useful info to DANSEparameters object from WASNparameters
        object."""
        self.nNodes = wasnParams.nNodes
        self.nSensorPerNode = wasnParams.nSensorPerNode
        self.referenceSensor = wasnParams.referenceSensor
        self.baseFs = wasnParams.fs
        self.seed = wasnParams.topologyParams.seed
        #
        self.wasnInfoInitiated = True
    
# def prep_sigs_for_FFT(y, N, Ns, t):
#     """
#     Zero-padding and signals length adaptation to ensure correct
#     FFT/IFFT operation. Based on FFT implementation by `scipy.signal` module.
#     -- Based on `prep_for_ffts` by Paul Didier
#     (`01_algorithms/01_NR/02_distributed/danse_utilities/setup.py`).
    
#     Parameters
#     ----------
#     y : [Nt x Nsensors] np.ndarray (float)
#         The microphone signals.
#     N : int
#         WOLA-DANSE frame size [samples].
#     Ns : int
#         Number of new samples per frame (`N * (1 - ovlp)`,
#         with `ovlp` the WOLA window overlap) [samples].
#     t : [N x 1] np.ndarray (float)
#         Sensor-specific time stamps vector.

#     Returns
#     -------
#     yout : np.ndarray (float)
#         Prepped signals.
#     tout : np.ndarray (float)
#         Corresponding time stamps.
#     nadd : int
#         Number of zeros added at the of signal after
#         frame-extension (step 2 below).
#     """

#     # 1) Extend signal on both ends to ensure that the first frame is centred
#     # at t = 0 -- see <scipy.signal.stft>'s `boundary` argument
#     # (default: `zeros`).
#     y = zero_ext(y, N // 2, axis=0)
#     # --- Also adapt timeInstants vector
#     dt = np.diff(t)[0]   # delta t between each time instant
#     tpre = np.linspace(start=-dt*(N//2), stop=-dt, num=N//2)
#     tpost = np.linspace(start=t[-1]+dt, stop=t[-1]+dt*(N//2), num=N//2)
#     t = np.concatenate((tpre, t, tpost), axis=0)

#     # 2) Zero-pad signal if necessary to include an
#     # integer number of frames in the signal.
#     nadd = 0
#     if not (y.shape[0] - N) % Ns == 0:
#         # vvv See <scipy.signal.stft>'s `padded` argument (default: `True`)
#         nadd = (-(y.shape[0] - N) % Ns) % N
#         print(f'Padding {nadd} zeros to the signals in order to fit FFT size')
#         y = np.concatenate((y, np.zeros([nadd, y.shape[-1]])), axis=0)
#         # Adapt time vector too
#         tzp = np.linspace(start=t[-1] + dt, stop=t[-1] + dt * nadd, num=nadd)
#         t = np.concatenate((t, tzp), axis=0)
#         if not (y.shape[0] - N) % Ns == 0:   # double-check
#             raise ValueError('There is a problem with the zero-padding...')

#     # Prepare for output 
#     yout = copy.copy(y)
#     tout = copy.copy(t)

#     return yout, tout, nadd


def check_clock_jitter(timeInstants: np.ndarray, nNodes: int):
    """
    Check for clock jitter and save sampling frequencies.

    Parameters
    ----------
    timeInstants : [Nt x nNodes] np.ndarray (floats)
        Time instants corresponding to the samples of each of the `nNodes` nodes.
    nNodes : int
        Number of nodes.

    Returns
    -------
    fs : [Nn x 1] np.ndarray[float]
        Sampling frequency at each node [Hz].
    """
    fs = np.zeros(nNodes)
    for k in range(nNodes):
        deltas = np.diff(timeInstants[:, k])
        # vvv Allowing computer precision errors down to 1e-4*mean delta.
        precision = int(np.ceil(np.abs(np.log10(np.mean(deltas) / 1e4))))
        if len(np.unique(np.round(deltas, precision))) > 1:
            raise ValueError(f'[NOT IMPLEMENTED] Clock jitter detected: {len(np.unique(np.round(deltas, precision)))} different sample intervals detected for node {k + 1}.')
        # np.round(): not going below 1 PPM precision for typical fs >= 8 kHz.
        fs[k] = np.round(1 / np.unique(np.round(deltas, precision))[0], 3)

    return fs


def ensure_shortfat_orientation(x: np.ndarray):
    """
    Ensures that the input array is a matrix with more columns than rows.
    """
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))
    elif len(x.shape) > 2:
        raise ValueError('Input array must have 2 dimensions or less.')
    if x.shape[0] < x.shape[1]:
        x = x.T
    return x


def base_event_checks(timeInstants: np.ndarray, p: DANSEparameters):
    """
    Conducts basic checks on the event instants matrix.
    """
    # Make sure time stamps matrix is indeed a matrix, correctly oriented
    timeInstants = ensure_shortfat_orientation(timeInstants)
    # Number of nodes
    nNodes = timeInstants.shape[1]
    # Check for clock jitter and save sampling frequencies
    fs = check_clock_jitter(timeInstants, nNodes)
    # Check consistency
    if 'sim' in p.nodeUpdating and any(fs != fs[p.referenceSensor]):
        raise ValueError('Simultaneous node-updating impossible in the presence of SROs.')

    return timeInstants, nNodes, fs


def initialize_events(
        timeInstants: np.ndarray,
        p: DANSEparameters,
        wasnObj: WASN = WASN()
    ) -> tuple[list[DANSEeventInstant], np.ndarray, list[WASN]]:
    """
    Returns the matrix the columns of which to loop over in (SRO-affected)
    asynchronous node-updating  DANSE. For each event instant, the matrix
    contains the instant itself (in [s]), the node indices concerned by
    this instant, and the corresponding event type ("bc" for broadcast,
    "up" for update).
    
    Parameters
    ----------
    timeInstants : [Nt x Nn] np.ndarray (floats)
        Time instants corresponding to the samples of each of the `Nn` nodes.
    p : DANSEparameters object
        Parameters.
    wasnObj : WASN class instance
        Only used for TI-DANSE: WASN under consideration, state at WASN
        initialization (before any pruning to a tree topology).

    Returns
    -------
    outputEvents : [Ne x 1] list of DANSEeventInstant objects
        Event instants matrix.
    fs : [Nn x 1] np.ndarray[float]
        Sampling frequency at each node [Hz].
    wasnObjList : list[WASN class instances]
        Only used for TI-DANSE: List of the WASNs topologies that exist
        through the entire signal duration.
    """
    # Basic checks (clock jitter, etc.) + infer sampling frequencies
    timeInstants, nNodes, fs = base_event_checks(timeInstants, p)
    
    # Total signal duration [s] per node (after truncation during sig. gen.).
    Ttot = timeInstants[-1, :]

    # Prepare events matrix building
    prepOutput = prep_evmat_build(p, nNodes, wasnObj, fs, Ttot)

    # Build event matrix
    outputEvents = build_events_matrix(
        up_t=prepOutput['upInstants'],
        bc_t=prepOutput['bcInstants'],
        nodeUpdating=p.nodeUpdating,
        fuse_t=prepOutput['fuInstants'],
        re_t=prepOutput['reInstants'],
        tr_t=prepOutput['trInstantsArranged'],
        leafToRootOrderings=prepOutput['leafToRootOrderings'],
        firstUpdatingNode=p.seqUpdateStartNodeIdx
    )

    return outputEvents, fs, prepOutput['wasnObjList']


def initialize_events_batch(
    timeInstants: np.ndarray,
    p: DANSEparameters,
    wasnObj: WASN = None
    ) -> tuple[list[DANSEeventInstant], np.ndarray, list[WASN]]:
    """
    Returns the matrix the columns of which to loop over in (SRO-affected)
    asynchronous node-updating DANSE, batch implemenation.
    For each event instant, the matrix contains the instant
    itself (in [s]), the node indices concerned by this instant, and the
    corresponding event type (only one type: "up" for update).
    
    Parameters
    ----------
    timeInstants : [Nt x Nn] np.ndarray (floats)
        Time instants corresponding to the samples of each of the `Nn` nodes.
    p : DANSEparameters object
        Parameters.
    wasnObj : WASN class instance
        Only used for TI-DANSE: WASN under consideration, state at WASN
        initialization (before any pruning to a tree topology).

    Returns
    -------
    outputEvents : [Ne x 1] list of DANSEeventInstant objects
        Event instants matrix.
    fs : [Nn x 1] np.ndarray[float]
        Sampling frequency at each node [Hz].
    wasnObjList : list[WASN class instances]
        Only used for TI-DANSE: List of the WASNs topologies that exist
        through the entire signal duration.
    """
    if wasnObj is not None:
        raise NotImplementedError('Batch TI-DANSE not implemented yet.')
    wasnObjList = None

    # Basic checks (clock jitter, etc.) + infer sampling frequencies
    timeInstants, nNodes, fs = base_event_checks(timeInstants, p)

    # Total signal duration [s] per node (after truncation during sig. gen.).
    Ttot = timeInstants[-1, :]

    # Prepare events matrix building
    if p.simType == 'batch':
        # Expected number of DANSE update per node over total signal length
        numUpInTtot = np.floor(Ttot * fs / p.Ns)

        if 'topo-indep' in p.nodeUpdating:   # ad-hoc (non fully connected) WASN
            raise NotImplementedError('Batch TI-DANSE not implemented yet.')
        else: 
            # Expected DANSE update instants
            upInstants = [
                np.arange(np.ceil((p.DFTsize + p.Ns) / p.Ns),
                int(numUpInTtot[k])) * p.Ns/fs[k] for k in range(nNodes)
            ]
    else:
        raise ValueError(f'Unexpected `simType`: {p.simType} (should be "batch" for batch DANSE).')
    
    # Compile events
    outputEvents = build_events_matrix(
        up_t=upInstants,
        bc_t=[],
        nodeUpdating=p.nodeUpdating,
        fuse_t=[],
        re_t= [],
        tr_t=[],
        leafToRootOrderings=[],
        firstUpdatingNode=p.seqUpdateStartNodeIdx
    )
    
    return outputEvents, fs, wasnObjList


def prep_evmat_build(
        p: DANSEparameters,
        nNodes: int,
        wasnObj: WASN,
        fs: np.ndarray,
        Ttot: float
    ):
    """
    Prepare for event matrix building in online DANSE
    (also used for batch DANSE).

    Parameters
    ----------
    p : DANSEparameters object
        Parameters.
    nNodes : int
        Number of nodes.
    wasnObj : WASN class instance
        Only used for TI-DANSE: WASN under consideration, state at WASN
        initialization (before any pruning to a tree topology).
    fs : [Nn x 1] np.ndarray[float]
        Sampling frequency at each node [Hz].
    Ttot : [Nn x 1] np.ndarray[float]
        Total signal duration [s] per node (after truncation during sig. gen.).

    Returns
    -------
    out : dict
    """
    
    # Expected number of DANSE update per node over total signal length
    numUpInTtot = np.floor(Ttot * fs / p.Ns)
    # Expected number of broadcasts per node over total signal length
    numBcInTtot = np.floor(Ttot * fs / p.broadcastLength)

    if 'topo-indep' in p.nodeUpdating:   # ad-hoc (non fully connected) WASN
        
        # Arange node indices starting from the sequential-update start-node
        orderedSeqUpNodes = np.mod(np.arange(
            start=p.seqUpdateStartNodeIdx,
            stop=p.seqUpdateStartNodeIdx + nNodes
        ), nNodes)
        # Get leaf-to-root orderings for all possible root indices
        leafToRootOrderings = []
        possibleWASNs = []
        for ii, k in enumerate(orderedSeqUpNodes):
            # Update WASN with new root
            updatingWasnObj = prune_wasn_to_tree(wasnObj, forcedRoot=k)
            leafToRootOrderings.append(updatingWasnObj.leafToRootOrdering)
            possibleWASNs.append(updatingWasnObj)
        # Cycling through a list (https://stackoverflow.com/a/8940984)
        orderingsCycled = cycle(leafToRootOrderings)
        wasnObjListCycled = cycle(possibleWASNs)
        
        # Get expected broadcast instants
        if 'wholeChunk' in p.broadcastType:
            # Expected DANSE update instants
            # upInstants = [
            #     np.arange(np.ceil((p.DFTsize + p.Ns) / p.Ns),
            #     int(numUpInTtot[k])) * p.Ns/fs[k] for k in range(nNodes)
            # ]

            # Edit on 2023.05.08 we only start updating when we have enough
            # samples so that the first update is not affected by the WOLA
            # analysis window (no "fading in" of the data) -- see journal 2023
            # week19 MON entry --------vvvv--------
            upInstants = [
                np.arange(np.ceil(2 * p.DFTsize / p.Ns),
                int(numUpInTtot[k])) * p.Ns/fs[k] for k in range(nNodes)
            ]
            # ^ note that we only start updating when we have enough samples.
            fuInstants = [
                np.arange(p.DFTsize/p.broadcastLength, int(numBcInTtot[k])) *\
                    p.broadcastLength/fs[k] for k in range(nNodes)
            ]
            # ^ note that we only start fusing when we have enough samples.

            bcInstants = copy.deepcopy(fuInstants)
            # ^ we differentiate fusion instants from broadcast instants to 
            # ensure fusion occurs in all nodes before broadcasting (necessary
            # to compute partial in-network sums).
            # ^ /!\ /!\ assuming no computational/communication delays /!\ /!\
            # if 'seq' in p.nodeUpdating and not p.keepOriginalTree:
            reInstants = copy.deepcopy(fuInstants)  # same relay instants
                # raise ValueError('WRONG: the relay instants depend on which node is the root!!!')
            # ^ /!\ /!\ assuming no computational/communication delays /!\ /!\
            if 'seq' in p.nodeUpdating and not p.keepOriginalTree:
                # form new tree at every DANSE update
                seqUpInstants, upNodeIndices = get_sequential_update_instants(
                    upInstants=upInstants,
                    startNodeIdx=p.seqUpdateStartNodeIdx
                )
                trInstants = copy.deepcopy(seqUpInstants)
                trInstants[0] = 0  # the first tree formation happens at t=0 s.
            else:
                # form tree at WASN initialization only
                trInstants = np.array([0])
                upNodeIndices = np.array([p.seqUpdateStartNodeIdx])
            trInstantsArranged = [trInstants[upNodeIndices == k] for k in range(nNodes)]
            # Keep only the relevant leaf-to-root orderings
            leafToRootOrderings = list(
                islice(orderingsCycled, None, len(trInstants))
            )
            # Keep only the relevant WASNs
            wasnObjList = list(
                islice(wasnObjListCycled, None, len(trInstants))
            )
        else:
            raise ValueError('fewSamples - Not yet implemented for TI-DANSE.')

    else:   # fully connected WASN
        fuInstants = [np.array([]) for _ in range(nNodes)]  # no fusion instants
        reInstants = [np.array([]) for _ in range(nNodes)]  # no relay instants
        trInstantsArranged = [np.array([]) for _ in range(nNodes)]  # no tree-formation instants
        leafToRootOrderings = None
        wasnObjList = None
        # Expected DANSE update instants
        upInstants = [
            np.arange(np.ceil((p.DFTsize + p.Ns) / p.Ns),
            int(numUpInTtot[k])) * p.Ns/fs[k] for k in range(nNodes)
        ]
        # upInstants = [
        #     np.arange(np.ceil(p.DFTsize / p.Ns),
        #     int(numUpInTtot[k])) * p.Ns/fs[k] for k in range(nNodes)
        # ]
        # Make the update instants match actual sample instants
        for k in range(nNodes):
            timeInstants = wasnObj.wasn[k].timeStamps
            for ii in range(len(upInstants[k])):
                if upInstants[k][ii] not in timeInstants:
                    # Replace by closest future sample instant
                    possibleTimeInstants = timeInstants[timeInstants > upInstants[k][ii]]
                    if len(possibleTimeInstants) > 0:
                        upInstants[k][ii] = possibleTimeInstants[0]
                    else:
                        upInstants[k][ii] = timeInstants[-1]
        # ^ note that we only start updating when we have enough samples.
        # Get expected broadcast instants
        if 'wholeChunk' in p.broadcastType:
            bcInstants = [
                np.arange(p.DFTsize/p.broadcastLength, int(numBcInTtot[k])) *\
                    p.broadcastLength/fs[k] for k in range(nNodes)
            ]
            # ^ note that we only start broadcasting when we have enough
            # samples to perform compression.
        elif 'fewSamples' in p.broadcastType:
            if p.efficientSpSBC:
                bcInstants = []
                # The broadcast instants at node `k` are the union of the
                # update instants of all its neighbors.
                for k in range(nNodes):
                    combinedUpInstants = []
                    for q in wasnObj.wasn[k].neighborsIdx:
                        # vvv include the first update instant (t=0 s)
                        # combinedUpInstants.append(0)
                        # combinedUpInstants.append(p.Ns/fs[q])
                        for ii in range(len(upInstants[q])):
                            if upInstants[q][ii] not in combinedUpInstants:
                                combinedUpInstants.append(upInstants[q][ii])
                    bcInstants.append(np.sort(np.array(combinedUpInstants)))
                # Ensure that the broadcast instants include the time corresponding to Ns samples
                # for k in range(nNodes):
                #     bcInstants[k] = np.concatenate(
                #         (
                #             np.array([0, p.Ns/fs[k]]),
                #             bcInstants[k]
                #         )
                #     )
                    # bcInstants[k] = np.insert(
                    #     bcInstants[k],
                    #     0,
                    #     p.Ns/fs[k]
                    # )

                # # Combine update instants across nodes
                # combinedUpInstants = list(upInstants[0])
                # for k in range(1, nNodes):
                #     for ii in range(len(upInstants[k])):
                #         if upInstants[k][ii] not in combinedUpInstants:
                #             combinedUpInstants.append(upInstants[k][ii])
                # combinedUpInstants = np.sort(np.array(combinedUpInstants))
                # # Same BC instants for all nodes
                # bcInstants = [combinedUpInstants for _ in range(nNodes)]
            else:
                bcInstants = [
                    np.arange(1, int(numBcInTtot[k])) *\
                        p.broadcastLength / fs[k] for k in range(nNodes)
                ]
                # ^ note that we start broadcasting sooner:
                # when we have `L` samples, enough for linear convolution.

    # Create output dictionary
    out = {
        'upInstants': upInstants,
        'fuInstants': fuInstants,
        'bcInstants': bcInstants,
        'reInstants': reInstants,
        'trInstantsArranged': trInstantsArranged,
        'leafToRootOrderings': leafToRootOrderings,
        'wasnObjList': wasnObjList
    }

    return out


def get_sequential_update_instants(
        upInstants: list[np.ndarray],
        startNodeIdx: int
    ):
    """
    Returns the update instants in a sequential node-updating DANSE scheme,
    based on the asynchronous update instants `upInstants` and a starting node
    index `startNodeIdx`. Can account for the presence of SROs.

    Returns
    -------
    seqUpInstants : np.ndarrray[float]
        Sequential updates time instants. 
    updatingNodeIndices : np.ndarrray[float]
        Corresponding updating node indices. 
    """
    currInstant = 0
    updatingNodeIdx = startNodeIdx
    seqUpInstants = []
    updatingNodeIndices = [startNodeIdx]
    while currInstant < np.amax(upInstants[updatingNodeIdx]):
        upInstantsUpdatingNode = upInstants[updatingNodeIdx]
        currSeqUpInstant = upInstantsUpdatingNode[
            upInstantsUpdatingNode > currInstant
        ][0]
        # Add to list
        seqUpInstants.append(currSeqUpInstant)
        currInstant = currSeqUpInstant
        if currInstant < np.amax(upInstants[updatingNodeIdx]):
            # Update updating node index
            updatingNodeIdx = np.mod(updatingNodeIdx + 1, len(upInstants))
            updatingNodeIndices.append(updatingNodeIdx)
    # Convert to arrays
    seqUpInstants = np.array(seqUpInstants)
    updatingNodeIndices = np.array(updatingNodeIndices)

    return seqUpInstants, updatingNodeIndices


def build_events_matrix(
        up_t,
        bc_t=[],
        nodeUpdating='seq',
        visualizeUps=False,
        fuse_t=[],
        re_t=[],
        tr_t=np.array([]),
        leafToRootOrderings=[],
        firstUpdatingNode=0
    ):
    """
    Builds the DANSE events matrix from the update and broadcast instants.
    
    Parameters
    ----------
    up_t : [nNodes x 1] list of np.ndarrays (float)
        Update instants per node [s].
    bc_t : [nNodes x 1] list of np.ndarrays (float)
        Broadcast instants per node [s].
    nodeUpdating : str
        Type of node updating ('seq', 'asy', 'sim', 'topo-indep_seq',
        'topo-indep_asy', 'topo-indep_sim').
    visualizeUps : bool
        If True, plot a visualization of the expected update instants for 
        each node in the WASN.
    vvvv Used iff `'topo-indep' in nodeUpdating` vvvv
    fuse_t : [nNodes x 1] list of np.ndarrays (float)
        Fusion instants per node [s].
    re_t : [nNodes x 1] list of np.ndarrays (float)
        Relay instants per node [s].
    tr_t : np.ndarray[float]
        Tree-formation instants [s].
    leafToRootOrderings : list of list[int or list[int]]
        Leaves to root ordering of node indices, for each tree-formation
        instants (elements of `tr_t`).
        Nodes that live on the same tree depth are groupped in lists.
    firstUpdatingNode : int
        Index of the first updating node (only used if `'seq' in nodeUpdating`).

    Returns
    -------
    outputEvents : [Ne x 1] list of DANSEeventInstant objects
        Event instants matrix.
    """
    def _flatten_instants(K, t, eventRef):
        """
        Helper function -- flatten instants array.
        
        Parameters
        ----------
        K : int
            Number of nodes.
        t : [K x 1] list[np.ndarray] (float)
            Time instants to consider [s].
        eventRef : int
            Event reference (integer code, see variable `eventsCodingDict`).

        Returns
        -------
        eventInstants : [N x 3] np.ndarray (float)
            Flattened event array.
            [:, 0] == instant;
            [:, 1] == node index;
            [:, 2] == event reference;
        """
        if len(t) == 0:   # no instants
            return np.zeros((0, 3))
        else:
            nInstants = sum(
                [len(np.unique(t[k])) for k in range(K)]
            )
            eventInstants = np.zeros((nInstants, 3))
            for k in range(K):
                idxStart = sum([len(t[q]) for q in range(k)])
                idxEnd = idxStart + len(t[k])
                eventInstants[idxStart:idxEnd, 0] = t[k]
                eventInstants[idxStart:idxEnd, 1] = k
                eventInstants[:, 2] = eventRef    # event reference
            return eventInstants
    
    # Events dictionary (in chronological order of events)
    # -- 'fu_ds': fuse local signals (only if `'topo-indep' in nodeUpdating`).
    # -- 'bc_ds': 
    #    -- if `'topo-indep' not in nodeUpdating`: 
    #    Fuse and broadcast local data to neighbors.
    #    -- elif `'topo-indep' in nodeUpdating`: 
    #    Compute partial in-network sum and broadcast downstream (towards root).
    # -- 'up_us': perform DANSE filter update.
    # >> General code: `ds` means "downstream" (from leaves towards root),
    #       `us` means "upstream" (from root towards leaves).
    #       The `ds` events should occur through the whole tree before the
    #       `us` events can occur.
    # >> Coding format: '<ref string>': [priority level, up/downstream]
    eventsCodesDict = dict([
        ('tr', [-1, 'ds']), # 0) form tree topology;
        ('fu', [0, 'ds']),  # 1) fuse local signals;
        ('bc', [1, 'ds']),  # 2) compute partial in-network sum and broadcast;
        ('re', [2, 'us']),  # 3) relay in-network sum from root upstream;
        ('up', [3, 'us']),  # 4) update node-specific filter estimates.
    ])

    # Make sure the event codes dictionary is correctly ordered
    if not [eventsCodesDict[list(eventsCodesDict.keys())[ii]] <\
        eventsCodesDict[list(eventsCodesDict.keys())[ii + 1]]\
        for ii in range(len(list(eventsCodesDict.keys())) - 1)]:
        raise ValueError(
            'The events code dictionary is not correclty ordered.'
        )

    # Useful variables
    nNodes = len(up_t)  # number of nodes in WASN
    reversedEventsCodingDict = dict(
        [(eventsCodesDict[a][0], a) for a in eventsCodesDict]
    )
    # TI-DANSE flag
    tiFlag = any(np.array([len(ts) > 0 for ts in fuse_t])) and\
        len(fuse_t) == nNodes  

    # Create event lists
    upInstants = _flatten_instants(nNodes, up_t, eventsCodesDict['up'][0])
    bcInstants = _flatten_instants(nNodes, bc_t, eventsCodesDict['bc'][0])
    # Total number of unique events
    numEventInstants = upInstants.shape[0] + bcInstants.shape[0]
    if tiFlag:
        # Consider fusion, relay, and tree-formation instants (TI-DANSE)
        fuInstants = _flatten_instants(nNodes, fuse_t, eventsCodesDict['fu'][0])
        reInstants = _flatten_instants(nNodes, re_t, eventsCodesDict['re'][0])
        trInstants = _flatten_instants(nNodes, tr_t, eventsCodesDict['tr'][0])
        numEventInstants += fuInstants.shape[0] + reInstants.shape[0] +\
            trInstants.shape[0]
    else:
        fuInstants = np.zeros((0, 3))  # default: no fusion instants
        reInstants = np.zeros((0, 3))  # default: no relay instants
        trInstants = np.zeros((0, 3))  # default: no tree-formation instants
    
    # Combine
    eventInstants = np.concatenate(
        (trInstants, upInstants, bcInstants, fuInstants, reInstants),
        axis=0
    )
    # Sort
    idxSort = np.argsort(eventInstants[:, 0], axis=0)
    eventInstants = eventInstants[idxSort, :]

    # Group
    outputInstants: list[DANSEeventInstant] = []
    eventIdx = 0    # init while-loop
    nodesConcerned = []             # init
    evTypesConcerned = []        # init
    lastUpNode = firstUpdatingNode - 1     # index of latest updating node
    # ^^^ init. at `firstUpdatingNode - 1` so that, first,
    #       `lastUpNode + 1 == firstUpdatingNode`.
    treeFormationCounter = -1
    # ^^^ init. at -1 so that `treeFormationIdx + 1 == 0`
    lastTreeFormationInstant = None
    while eventIdx < numEventInstants:

        currInstant = eventInstants[eventIdx, 0]
        nodesConcerned.append(int(eventInstants[eventIdx, 1]))
        evTypesConcerned.append(int(eventInstants[eventIdx, 2]))

        # Check if a new tree topology must be formed
        if tiFlag:
            if currInstant in trInstants[:, 0] and\
                currInstant != lastTreeFormationInstant:
                treeFormationCounter += 1
                lastTreeFormationInstant = currInstant
            # Set leaves-to-root ordering for current event groupping
            leafToNodeOrder = leafToRootOrderings[treeFormationCounter]
        else:
            leafToNodeOrder = []  # no need for an order in a fully conn. WASN

        # Group same-time-instant events
        eventIdx, nodesConcerned, evTypesConcerned = events_groupping_check(
            evIdx=eventIdx,
            numEv=numEventInstants,
            ev_t=eventInstants,
            nodes=nodesConcerned,
            evTypes=evTypesConcerned
        )
        # Transform to np.ndarray
        nodesConcerned = np.array(nodesConcerned, dtype=int)
        evTypesConcerned = np.array(evTypesConcerned, dtype=int)

        # Sort same-time-instant events
        indices = sort_simultaneous_events(
            evTypes=evTypesConcerned,
            nodes=nodesConcerned,
            eventsCodesDict=eventsCodesDict,
            tidanseFlag=tiFlag,
            leafToRootOrder=leafToNodeOrder
        )
        nodesConcerned = nodesConcerned[indices]
        evTypesConcerned = evTypesConcerned[indices]

        # Event types (broadcast or update)
        # types = ['bc' if ii == 0 else 'up' for ii in eventTypesConcerned]
        types = [reversedEventsCodingDict[ii] for ii in evTypesConcerned]

        # Address node-updating strategy
        bypassUpdate = [None for _ in types]  # default: no bypass
        if 'up' in types and 'seq' in nodeUpdating:
            lastUpNodeUpdated = lastUpNode
            for ii in range(len(types)):
                if types[ii] == 'up':
                    if nodesConcerned[ii] == np.mod(lastUpNode + 1, nNodes):
                        # Increment last updating node index
                        lastUpNodeUpdated = nodesConcerned[ii]
                        bypassUpdate[ii] = False
                    else:
                        # Bypass update in other nodes
                        bypassUpdate[ii] = True
            lastUpNode = lastUpNodeUpdated

        # Build events matrix
        outputInstants.append(DANSEeventInstant(
            t=currInstant,
            nodes=nodesConcerned,
            type=types,
            bypassUpdate=bypassUpdate
        ))

        nodesConcerned = []         # reset list
        evTypesConcerned = []    # reset list

    # Visualize update instants
    if visualizeUps:
        fig, axes = plt.subplots(1,1)
        fig.set_size_inches(6.5, 1.5)
        t = 0
        idxInstant = 0
        while t < 1:    # hard-coded, but ok
            currEvents = outputInstants[idxInstant]
            t = currEvents.t
            for ii in range(len(currEvents.type)):
                k = currEvents.nodes[ii]
                if currEvents.type[ii] == 'up':
                    alpha = 1
                    if currEvents.bypassUpdate[ii]:
                        alpha = 0.2
                    axes.vlines(
                        x=t, ymin=k, ymax=k+0.9, colors=f'C{k}', alpha=alpha)
            idxInstant += 1
        axes.set_yticks(np.arange(nNodes) + 0.5)
        axes.set_yticklabels([f'Node {k+1}' for k in range(nNodes)])
        axes.grid()
        axes.set_title('Update instants')
        plt.tight_layout()	
        plt.show()

    return outputInstants


def sort_simultaneous_events(
        evTypes,
        nodes,
        eventsCodesDict: dict,
        tidanseFlag=False,
        leafToRootOrder: list=[]
    ):
    """
    Sorts DANSE events occurring at the same time instants.
    -- Some specific DANSE events should occur (from a pure programming /
        computations point of view) before/after others.

    Parameters
    ----------
    evTypes : list[int]
        Event types concerned during current loop (outside the function).
    nodes : list[int]
        Nodes concerned during current loop (outside the function).
    eventsCodesDict : dict{str : int}
        Dictionary for correspondence between event codes in integer format and
        event codes in string format.
    tidanseFlag : bool
        If True, configure for TI-DANSE. Else, for fully connected DANSE.
    leafToRootOrder : list[int or list[int]]
        Leaves to root ordering of node indices. Nodes that live on the same
        tree depth are groupped in lists.


    Returns
    -------
    indices : [len(evTypes) x 1] np.ndarray (int)
        Ordering indices (to be applied subsequently to, e.g.,
        `evTypes` and `nodes`).
    """
    def _compile_indices(idxEv, nodeTreeOrder, indices, nodes):
        """
        Helper function: orders the provided `idxEvent` indices of TI-DANSE
        events (of a specific type) based on the provided tree-topology WASN
        node ordering (leave-to-root (downstream) or root-to-leaves
        (upstream)).

        Parameters:
        -----------
        idxEv : np.ndarray (int)
            Event indices corresponding to the currently considered type of
            TI-DANSE event.
        nodeTreeOrder : list[int or list[int]]
            Leaves-to-root or root-to-leaves ordering of node indices.
            Nodes that live on the same tree depth are groupped in lists.
        indices : np.ndarray (int)
            Previously computed TI-DANSE event ordering indices.
        
        Returns
        -------
        indices : np.ndarray (int)
            Updated/completed TI-DANSE event ordering indices.
        """
        for treeLevel in nodeTreeOrder:
            # Loop over branch-depth levels in the tree topology
            if not isinstance(treeLevel, list):
                idxCurrLevel = idxEv[nodes[idxEv] == treeLevel]
                indices = np.concatenate((indices, idxCurrLevel))
            else:
                treeLevel.sort()  # sort list in ascending node index order
                for k in treeLevel: # consider each node in the level
                    idxCurrLevel = idxEv[nodes[idxEv] == k]
                    indices = np.concatenate((indices, idxCurrLevel))
        return indices
    
    indices = np.empty(0, dtype=int)  # init
    baseIndices = np.arange(len(evTypes))

    if tidanseFlag:  # ad-hoc WASN topology case
        # Order downstream (`ds`) events first.
        for key in eventsCodesDict.keys():
            if eventsCodesDict[key][1] == 'ds':  # select downstream events only
                idxEvent = baseIndices[evTypes == eventsCodesDict[key][0]]
                # Order depending on leaves-to-root structure
                indices = _compile_indices(
                    idxEvent,
                    leafToRootOrder,
                    indices,
                    nodes
                )
        # Then, order upstream (`ds`) events.
        for key in eventsCodesDict.keys():
            if eventsCodesDict[key][1] == 'us':  # select upstream events only
                idxEvent = baseIndices[evTypes == eventsCodesDict[key][0]]
                # Order depending on root-to-leaves structure
                reversedOrder = copy.deepcopy(leafToRootOrder)
                reversedOrder.reverse()  # /!\ REVERSE: root to leaves
                indices = _compile_indices(
                    idxEvent,
                    reversedOrder,
                    indices,
                    nodes
                )

    else:  # fully connected WASN topology case
        for key in eventsCodesDict.keys():
            idxEvent = baseIndices[evTypes == eventsCodesDict[key][0]]
            # Order by node index
            if len(idxEvent) > 0:
                idxEvent = idxEvent[np.argsort(nodes[idxEvent])]
            indices = np.concatenate((indices, idxEvent))

    return indices


def events_groupping_check(evIdx, numEv, ev_t, nodes: list, evTypes: list):
    """
    Checks whether events are occurring at the same time instant and should,
    therefore, be groupped together.

    Parameters
    ----------
    evIdx : int
        Event index (used to keep track of events outside the function).
    numEv : int
        Total number of events looped over outside the function.
    ev_t : [N x 3] np.ndarray (float)
        Flattened event array.
        [:, 0] == instant;
        [:, 1] == node index;
        [:, 2] == event reference.
    nodes : list[int]
        Nodes concerned during current loop (outside the function).
    evTypes : list[int]
        Event types concerned during current loop (outside the function).

    Returns
    -------
    evIdx : int
        Updated event index.
    nodes : list[int]
        Updated nodes concerned during current loop.
    evTypes : list[int]
        Updated event types concerned during current loop.
    """
    
    curr_t = ev_t[evIdx, 0]  # current instant
    if evIdx < numEv - 1:
        next_t = ev_t[evIdx + 1, 0]
        while curr_t == next_t:
            evIdx += 1
            curr_t = ev_t[evIdx, 0]
            nodes.append(int(ev_t[evIdx, 1]))
            evTypes.append(int(ev_t[evIdx, 2]))
            # Check whether the next instant is the same and
            # should be grouped with the current instant.
            if evIdx < numEv - 1:
                next_t = ev_t[evIdx + 1, 0]
            else:
                evIdx += 1
                break
        else:
            evIdx += 1
    else:
        evIdx += 1

    return evIdx, nodes, evTypes


def local_chunk_for_broadcast(y: np.ndarray, t, fs, DFTsize):
    """
    Extract correct chunk of local signals for broadcasting.
    
    Parameters
    ----------
    y : [Ntot x Mk] np.ndarray (float)
        Time-domain locally recorded signal at `Mk` sensors.
    t : float
        Current time instant [s].
    fs : int or float
        Transmitting node's sampling frequency [Hz].
    N : int
        Frame size (= FFT size in DANSE).

    Returns
    -------
    chunk : [N x Mk] np.ndarray (float)
        Time chunk of local sensor signals.
    """

    # vvv -- np.round() used to avoid issues with previous
    # rounding/precision errors (see Word journal week 32, THU, 2022).
    # idxEnd = int(np.floor(np.round(t * fs, 5)))  # COMMENTED AGAIN ON 2023.05.05
    idxEnd = int(np.floor(t * fs))
    # vvv -- don't go into negative sample indices!
    idxBeg = np.amax([idxEnd - DFTsize, 0])
    chunk = y[idxBeg:idxEnd, :]
    # Pad zeros at beginning if needed
    if idxEnd - idxBeg < DFTsize:
        chunk = np.concatenate((
            np.zeros((DFTsize - chunk.shape[0], chunk.shape[1])),
            chunk
        ))

    return chunk, idxBeg, idxEnd


def local_chunk_for_update(y, t, fs, bd, Ndft, Ns):
    """
    Extract correct chunk of local signals for DANSE updates.
    
    Parameters
    ----------
    y : [Ntot x Mk] np.ndaray (float)
        Time-domain locally recorded signal (at `Mk` sensors).
    t : float
        Current time instant [s].
    fs : int or float
        Transmitting node's sampling frequency [Hz].
    bd : str
        Broadcast type ("fewSamples" or "wholeChunk").
    Ndft : int
        DFT size.
    Ns : int
        Number of new samples at each new WOLA frame.

    Returns
    -------
    chunk : [N x Mk] np.ndarray (float)
        Time chunk of local sensor signals.
    idxBeg : int
        Start index of chunk (w.r.t. `y`).
    idxEnd : int
        End index of chunk (w.r.t. `y`).
    """

    # Broadcast scheme: block-wise, in freq.-domain
    # <or> Broadcast scheme: few samples at a time, in time-domain
    if bd == 'fewSamples':
        # idxEnd = int(np.floor(np.round(t * fs, 5)))  # COMMENTED AGAIN ON 2023.05.05
        idxEnd = int(np.floor(t * fs))
    # Broadcast scheme: block-wise, in time-domain
    elif bd == 'wholeChunk':
        # `N - Ns` samples delay due to WOLA
        # idxEnd = int(np.floor(np.round(t * fs, 5))) - (Ndft - Ns)    # COMMENTED AGAIN ON 2023.05.05
        idxEnd = int(np.floor(t * fs)) - (Ndft - Ns)

    # vvv -- don't go into negative sample indices!
    idxBeg = np.amax([idxEnd - Ndft, 0])
    chunk = y[idxBeg:idxEnd, :]
    # Pad zeros at beginning if needed (occurs at algorithm's startup)
    if idxEnd - idxBeg < Ndft:
        chunk = np.concatenate((
            np.zeros((Ndft - chunk.shape[0], chunk.shape[1])), chunk
        ))

    return chunk, idxBeg, idxEnd


def back_to_time_domain(x, n, axis=0):
    """
    Performs an IFFT after pre-processing of a frequency-domain
    signal chunk.
    
    Parameters
    ----------
    x : np.ndarray of complex
        Frequency-domain signal to be transferred back to time domain.
    n : int
        IFFT order.
    axis : int (0 or 1)
        Array axis where to perform IFFT.
        -- not implemented for more than 2-D arrays.

    Returns
    -------
    xout : np.ndarray of floats
        Time-domain version of signal.
    """

    # Interpret `axis` parameter
    flagSingleton = False
    if x.ndim == 1:
        x = x[:, np.newaxis]
        flagSingleton = True
    elif x.ndim > 2:
        raise np.AxisError(f'{x.ndim}-D arrays not permitted.')
    if axis not in [0,1]:
        raise np.AxisError(f'`axis={axis}` is not permitted.')

    if axis == 1:
        x = x.T

    # Check dimension
    if x.shape[0] != n/2 + 1:
        raise ValueError('`x` should be (n/2+1)-long along the IFFT axis.')

    x[0, :] = x[0, :].real      # Set DC to real value
    x[-1, :] = x[-1, :].real    # Set Nyquist to real value
    x = np.concatenate((x, np.flip(x[:-1, :].conj(), axis=0)[:-1, :]), axis=0)
    
    # vvv -- go back to original input dimensionality before FFT
    # (bias of np.fft.fft with (n, 1)-dimensioned input).
    if flagSingleton:
        x = np.squeeze(x)

    # Back to time-domain
    xout = np.fft.ifft(x, n, axis=0)

    if axis == 1:
        xout = xout.T
    
    return xout


def extract_few_samples_from_convolution(idDesired, a, b):
    """
    Manually computes convolution between `a` and `b`
    only for the output indices desired (`idDesired`). 

    Parameters
    ----------
    idDesired : [L x 1] np.ndarray (float)
        Indices desired from convolution output.
    a : [N x 1] np.ndarray (float)
        FIR filter (time-domain).
    b : [M x 1] np.ndarray (float)
        Signal to be used for convolution.

    Returns
    -------
    out : [L x 1] np.ndarray (float)
        Output samples from convolution between `a` and `b`.
    """

    out = np.zeros(len(idDesired))
    yqzp = np.concatenate((np.zeros(len(a)), b, np.zeros(len(a))))
    for ii in range(len(idDesired)):
        out[ii] = np.dot(
            yqzp[idDesired[ii] + 1:idDesired[ii] + 1 + len(a)],
            np.flip(a)
        )
    
    return out


def fill_buffers_td_few_samples(k, neighs, zBuffer, zLocalK, L):
    """
    Fill in buffers -- simulating broadcast of compressed signals
    from one node (`k`) to its neighbours.
    
    Parameters
    ----------
    k : int
        Current node index.
    neighs : [numNodes x 1] list of [nNeighbours[n] x 1] lists (int)
        Network indices of neighbours, per node.
    zBuffer : [numNodes x 1] list of [nNeighbours[n] x 1] ... 
            ... lists of [variable length] np.ndarrays (float)
        Compressed signals buffers for each node and its neighbours.
    zLocalK : [N x 1] np.ndarray (float)
        Latest compressed local signals to be broadcasted from node `k`.
    L : int
        Broadcast chunk length.

    Returns
    -------
    zBuffer : [numNodes x 1] list of [nNeighbours[n] x 1] ...
            ... lists of [variable length] np.ndarrays (float)
        Updated compressed signals buffers for each node and its neighbours.
    """

    # Check for sample-per-sample broadcast scheme (`L==1`)
    if isinstance(zLocalK, float):
        zLocalK = [zLocalK]
        if L != 1:
            raise ValueError(f'Incoherence: float `zLocalK` but L = {L}.')

    # Loop over neighbors of node `k`
    for idxq in range(len(neighs[k])):
        # Network-wide index of node `q` (one of node `k`'s neighbors)
        q = neighs[k][idxq]
        idxKforNeighborQ = [i for i, x in enumerate(neighs[q]) if x == k]
        # Node `k`'s "neighbor index", from node `q`'s perspective
        idxKforNeighborQ = idxKforNeighborQ[0]
        # Only broadcast the `L` last samples of local compressed signals
        zBuffer[q][idxKforNeighborQ] = np.concatenate(
            (zBuffer[q][idxKforNeighborQ], zLocalK[-L:]),
            axis=0
        )
        
    return zBuffer


def events_parser(
        events: DANSEeventInstant,
        startUpdates,
        p: DANSEparameters
    ):
    """
    Printouts to inform user of DANSE events.
    
    Parameters
    ----------
    events : `DANSEeventInstant` object
        Event instants. One element of the output of the
        `get_events_matrix` function.
    startUpdates : list of bools
        Node-specific flags to indicate whether DANSE updates have started. 
    p : `DANSEparameters` object
        DANSE parameters.
    """
    if p.printoutsAndPlotting.printout_eventsParser:
        if 'up' in events.type:
            txt = f'[{p.simType}] [{p.nodeUpdating}] t={np.round(events.t, 3):.3f}s -- '
            updatesTxt = 'Updating nodes: '
            updatesFlag = False
            if p.printoutsAndPlotting.printout_eventsParserNoBC:
                broadcastsTxt = ''
            else:
                broadcastsTxt = 'Broadcasting nodes: '
            # vvv -- little flag to add a comma (`,`) at the right spot.
            flagCommaUpdating = False
            for idxEvent in range(len(events.type)):
                k = int(events.nodes[idxEvent])   # node index
                if events.type[idxEvent] == 'bc' and\
                    not p.printoutsAndPlotting.printout_eventsParserNoBC:
                    if idxEvent > 0:
                        broadcastsTxt += ','
                    broadcastsTxt += f'{k + 1}'
                elif events.type[idxEvent] == 'up' and\
                    not events.bypassUpdate[idxEvent]:
                    # Only print if the node actually has started updating
                    # (i.e. there has been sufficiently many autocorrelation
                    # matrices updates since the start of recording).
                    if startUpdates[k]:
                        updatesFlag = True
                        if not flagCommaUpdating:
                            flagCommaUpdating = True
                        else:
                            updatesTxt += ','
                        updatesTxt += f'{k + 1}'
            # Get ready to print
            if not updatesFlag:
                updatesTxt = ''
            fullTxt = txt + broadcastsTxt + '; ' + updatesTxt
            if is_interactive():  # if we are running from a notebook
                # Print on the same line
                print(f"\r{fullTxt}", end="")
            else:
                # Print on the next line
                print(fullTxt)


def events_parser_ti_danse(
        events: DANSEeventInstant,
        startUpdates,
    ):
    """
    Printouts to inform user of TI-DANSE events.
    
    Parameters
    ----------
    events : `DANSEeventInstant` object
        Event instants. One element of the output of the
        `get_events_matrix` function.
    startUpdates : list of bools
        Node-specific flags to indicate whether DANSE updates have started.
    """
    # Hard-coded
    strCodes = dict([
        ('fu', "Local signals fusion -> z'_k"),
        ('bc', 'PiNS and downstream broadcast of z_k'),
        ('re', 'Upstream relay of η'),
        ('up', 'Filter update w_k[i+1]'),
        ('tr', 'Tree topology formation'),
    ])

    print(f'TI-DANSE: ---- t = {events.t} s ----')
    # Init while-loop
    flagEnd = False
    prevType = events.type[0]
    currNodes = []
    ii = 0
    while not flagEnd:
        if events.type[ii] != prevType:
            currNodes.append(events.nodes[ii])
            ii += 1
            if ii >= len(events.type):
                flagEnd = True
        else:
            # address all identical events
            counter = 0
            while events.type[ii] == prevType:
                if not events.bypassUpdate[ii] and\
                    not ((not startUpdates[events.nodes[ii]])\
                        and events.type[ii] == 'up'):
                    currNodes.append(events.nodes[ii])
                ii += 1
                counter += 1
                if ii >= len(events.type):
                    flagEnd = True
                    break  # break sub while-loop
        # Prepare printout
        nodesStr = ''
        for k in currNodes:
            nodesStr += f'{k} -> '
        if prevType == 'tr':  # different text for tree-formation
            fullTxt =\
                f'TI-DANSE -- {strCodes[prevType]} (root: Node {nodesStr[:-4]})'
        else:
            fullTxt =\
                f'TI-DANSE -- Node(s) {nodesStr[:-4]}: {strCodes[prevType]}'
        if is_interactive():  # if we are running from a notebook
            # Print on the same line
            print(f"\r{fullTxt}", end="")
        elif nodesStr != '':
            # Print on the next line
            print(fullTxt)
        # Prepare next while-loop pass
        if not flagEnd:
            prevType = events.type[ii]
            currNodes = []


def is_interactive():  # https://stackoverflow.com/a/22424821
    import __main__ as main
    return not hasattr(main, '__file__')


def danse_compression_whole_chunk(
        yq,
        wHat,
        h,
        f,
        zqPrevious=None,
        Ns=None
    ):
    """Performs local signals compression in the frequency domain.

    Parameters
    ----------
    yq : [N x nSensors] np.ndarray (float)
        Local sensor signals.
    wHat : [N/2 x nSensors] np.ndarray (float or complex)
        Frequency-domain local filter estimate (from latest DANSE iteration).
    h : [`n` x 1] np.ndarray (float)
        WOLA analysis window (time-domain to WOLA-domain).
    f : [`n` x 1] np.ndarray (float)
        WOLA synthesis window (WOLA-domain to time-domain).
    zqPrevious : [N x 1] np.ndarray (float)
        Previous time-domain chunk of compressed signal.
    Ns : int
        WOLA hop size (number of samples).

    Returns
    -------
    zqHat : [N/2 x 1] np.ndarray (complex)
        Frequency-domain compressed signal for current frame.
    zq : [N x 1] np.ndarray (float)
        Time-domain latest WOLA chunk of compressed signal (after OLA).
    """
    
    # Useful variables
    DFTsize = len(yq)
    normFactWOLA = Ns / sum(f)  # normalization factor for WOLA

    # Checking input arguments
    if Ns is None:
        # Assume 50% overlap 
        print("WARNING: `Ns` not specified. Assuming 50 percent WOLA overlap.")
        Ns = DFTsize // 2

    # Check for single-sensor case
    flagSingleSensor = False
    if wHat.shape[-1] == 1:
        wHat = np.squeeze(wHat)
        yq = np.squeeze(yq)
        flagSingleSensor = True

    # WOLA analysis stage
    if flagSingleSensor:
        yqHat = np.fft.fft(
            np.squeeze(yq) * h, DFTsize, axis=0
        ) / np.sqrt(Ns)
        # Keep only positive frequencies
        yqHat = yqHat[:int(DFTsize // 2 + 1)]
        # Apply linear combination to form compressed signal.
        # -- single sensor = simple element-wise multiplication.
        # 2023.06.26 TRIAL: no compression for single-sensor nodes
        zqHat = wHat.conj() * yqHat
        # zqHat = copy.deepcopy(yqHat)
    else:
        yqHat = np.fft.fft(
            np.squeeze(yq) * h[:, np.newaxis], DFTsize, axis=0
        ) / np.sqrt(Ns)
        # Keep only positive frequencies
        yqHat = yqHat[:int(DFTsize // 2 + 1), :]
        # Apply linear combination to form compressed signal.
        zqHat = np.einsum('ij,ij->i', wHat.conj(), yqHat)

    # WOLA synthesis stage
    if zqPrevious is not None:
        # IDFT
        zqCurr = np.sqrt(Ns) * back_to_time_domain(zqHat, DFTsize, axis=0)
        zqCurr = np.real_if_close(zqCurr)
        zqCurr *= f    # multiply by synthesis window
        # zqCurr *= normFactWOLA    # multiply by WOLA normalization factor

        if not np.any(zqPrevious):
            # No previous frame, keep current frame
            zq = zqCurr
        else:
            # Overlap-add
            zq = np.zeros(DFTsize)
            zq[:(DFTsize - Ns)] = zqPrevious[-(DFTsize - Ns):]
            zq += zqCurr
    else:
        print('Cannot compute time-domain overlap-added chunk of local compressed signal.')
        zq = None

    stop = 1
    
    return zqHat, zq


def danse_compression_few_samples(
        yq,
        wqqHat,
        L,
        wIRprevious,
        winWOLAanalysis,
        winWOLAsynthesis,
        Ns, 
        updateBroadcastFilter=False
    ):
    """
    Performs local signals compression according to DANSE theory [1],
    in the time-domain (to be able to broadcast the compressed signal sample
    per sample between nodes).
    Approximate WOLA filtering process via linear convolution.
    
    Parameters
    ----------
    yq : [Ntotal x nSensors] np.ndarray (real)
        Local sensor signals.
    wqqHat : [N/2 x nSensors] np.ndarray (real or complex)
        Frequency-domain local filter estimate (from latest DANSE iteration).
    L : int
        Broadcast length [samples].
    winWOLAanalysis : [`n` x 1] np.ndarray (float)
        WOLA analysis window (time-domain to WOLA-domain).
    winWOLAsynthesis : [`n` x 1] np.ndarray (float)
        WOLA synthesis window (WOLA-domain to time-domain).
    Ns : int
        Sample shift between adjacent windows.
    updateBroadcastFilter : bool
        If True, update TD filter for broadcast.

    Returns
    -------
    zq : [N x 1] np.ndarray (real)
        Compress local sensor signals (1-D).
    """

    # Profiling
    if updateBroadcastFilter:
        wIR = dist_fct_approx(
            wqqHat,
            winWOLAanalysis,
            winWOLAsynthesis,
            Ns
        )
    else:
        wIR = wIRprevious
    
    # Perform convolution
    yfiltLastSamples = np.zeros((L, yq.shape[-1]))
    for idxSensor in range(yq.shape[-1]):
        # Indices required from convolution output
        idDesired = np.arange(
            start=len(wIR) - L + 1,
            stop=len(wIR) + 1
        )
        tmp = extract_few_samples_from_convolution(
            idDesired,
            wIR[:, idxSensor],
            yq[:, idxSensor]
        )
        yfiltLastSamples[:, idxSensor] = tmp

    zq = np.sum(yfiltLastSamples, axis=1)

    return zq, wIR


@njit
def get_trace_jitted(A, ofst):
    """ 
    JIT-ting NumPy's `trace()` function.
    """
    return np.trace(A, ofst)


def dist_fct_approx(wHat, h, f, R, jitted=True):
    """
    Distortion function approximation of the WOLA filtering process.
    -- See Word journal 2022, weeks 30-33.

    Parameters
    ----------
    wHat : [Nf x M] np.ndarry (complex)
        Frequency-domain filter coefficients for each of the `M` channels
        (>0 freqs. only) used in the WOLA process to modify the
        short-term spectrum.
    h : [N x 1] np.ndarray (float)
        WOLA analysis window (time-domain).
    f : [N x 1] np.ndarray (float)
        WOLA synthesis window (time-domain).
    R : int
        Window shift [samples].
    jitted : bool
        If True, use numba to speed some computations up. 

    Returns
    -------
    wIR_out : [(2 * N + 1) x 1] np.ndarray (float)
        Time-domain distortion function approx. of the WOLA filtering process.
    """

    n = len(h)

    wTD = back_to_time_domain(wHat.conj(), n, axis=0)
    wTD = np.real_if_close(wTD)         
    wIR_out = np.zeros((2 * n - 1, wTD.shape[1]))
    for m in range(wTD.shape[1]):
        Hmat = sla.circulant(np.flip(wTD[:, m]))
        Amat = np.diag(f) @ Hmat @ np.diag(h)

        for ii in np.arange(start=-n+1, stop=n):
            if jitted:
                wIR_out[ii + n - 1, m] = get_trace_jitted(Amat, ii)
            else:
                wIR_out[ii + n - 1, m] = np.sum(np.diagonal(Amat, ii))

    wIR_out /= R

    return wIR_out


def perform_update_noforloop(Ryy, Rnn, refSensorIdx=0):
    """
    Regular DANSE update computations, `for`-loop free.
    No GEVD involved here.
    
    Parameters
    ----------
    Ryy : [M x N x N] np.ndarray (complex)
        Autocorrelation matrix between the sensor signals.
    Rnn : [M x N x N] np.ndarray (complex)
        Autocorrelation matrix between the noise signals.
    refSensorIdx : int
        Index of the reference sensor (>=0).

    Returns
    -------
    w : [M x N] np.ndarray (complex)
        Regular DANSE filter coefficients.
    """
    # Reference sensor selection vector
    Evect = np.zeros((Ryy.shape[-1],))
    Evect[refSensorIdx] = 1

    # Cross-correlation matrix update 
    ryd = np.matmul(Ryy - Rnn, Evect)
    # Update node-specific parameters of node k
    Ryyinv = np.linalg.inv(Ryy)
    w = np.matmul(Ryyinv, ryd[:,:,np.newaxis])
    w = w[:, :, 0]  # get rid of singleton dimension
    
    return w


def get_desired_sig_chunk(
        desSigProcessingType,
        w,
        y,
        win,
        normFactWOLA,
        dChunk,
        Ns,
        yTD
    ):
    """
    Computes STFT-domain frame and time-domain frame of desired
    signal estimate.
    
    Parameters
    ----------
    desSigProcessingType : str
        Type of processing to be used to build the time-domain signal:
        "wola" for WOLA processing /or/ "conv" for linear convolution via
        T(z)-approximation of WOLA processing.
    w : [N x M] np.ndarray (complex)
        STFT-domain multichannel (`M` channels) filter coefficients
        at current frame.
    y : [N x M] np.ndarray (complex)
        STFT-domain multichannel (`M` channels) signals at current frame.
    win : [N x 1] np.ndarray (float)
        WOLA analysis and synthesis window (option to use different windows
        for analysis and synthesis not yet implemented).
    normFactWOLA : float
        WOLA normalisation factor for transforms bw. frequency- & time-domain.
    dChunk : [N x 1] np.ndarray (float)
        Previous frame of time-domain desired signal estimate constructed
        from a WOLA process (only used if `desSigProcessingType == 'wola'`).
    Ns : int
        Number of new samples at each new WOLA frame.
    yTD : [N x M] np.ndarray (float)
        Time-domain frame of multichannel (`M` channels) signals. 
    
    Returns
    -------
    dChunk : [N x 1] np.ndarray (float)
        Latest time-domain frame of desired signal estimate.
    dhatCurr : [N x 1] np.ndarray (complex)
        Latest STFT-domain frame of desired signal estimate.
    """
    dhatCurr = None  # if `self.desSigProcessingType == 'conv'`
    if desSigProcessingType == 'wola':
        # Compute desired signal chunk estimate using WOLA
        dhatCurr = np.einsum('ij,ij->i', w.conj(), y)
        # Transform back to time domain (WOLA processing)
        dChunkCurr = normFactWOLA * win *\
            back_to_time_domain(dhatCurr, len(win))
        # Overlap and add construction of output time-domain signal
        if len(dChunk) < len(win):
            dChunk += np.real_if_close(dChunkCurr[-len(dChunk):])
        else:
            dChunk += np.real_if_close(dChunkCurr)

    elif desSigProcessingType == 'conv':
        # Compute desired signal chunk estimate using T(z) approximation
        wIR = dist_fct_approx(w, win, win, Ns)
        # Perform convolution
        yfiltLastSamples = np.zeros((Ns, yTD.shape[-1]))
        for m in range(yTD.shape[-1]):
            # Indices required from convolution output vvv
            idDesired = np.arange(start=len(wIR) - Ns, stop=len(wIR))
            tmp = extract_few_samples_from_convolution(
                idDesired, wIR[:, m], yTD[:, m]
            )
            yfiltLastSamples[:, m] = tmp

        dChunk = np.sum(yfiltLastSamples, axis=1)
    
    return dChunk, dhatCurr


def prune_wasn_to_tree(
    wasnObj: WASN,
    algorithm='prim',
    plotit=False,
    forcedRoot=None
    ) -> WASN:
    """
    Prunes a WASN to a tree topology.
    
    Parameters
    ----------
    wasnObj : `WASN` object
        WASN under consideration.
    algorithm : str
        Minimum-Spanning-Tree algorithm to be used.
        Valid values (from NetworkX toolbox): 'kruskal', 'prim', 'boruvka'.
        >> According to Paul Didier's testings from December 2022: 
            'kruskal' and 'prim' are faster and more scalable than 'boruvka'.
    plotit : bool
        If True, plots a visualization of the original graph and pruned graphs.
    forcedRoot : int
        If not None: tree root node index.

    Returns
    -------
    prunedWasnObj : `WASN` object
        WASN pruned to a tree topology.
    """
    # Generate NetworkX graph
    # Gnx = generate_graph_for_wasn(wasnObj.wasn)
    Gnx = nx.from_numpy_array(wasnObj.adjacencyMatrix)
    # Get node positions 
    nodesPos = dict(
        [(k, wasnObj.wasn[k].nodePosition) for k in range(len(wasnObj.wasn))]
    )
    
    # Add edge weights based on inter-node distance # TODO: is that a correct approach? TODO:
    if nodesPos[0] is not None:
        for e in Gnx.edges():
            weight = np.linalg.norm(nodesPos[e[0]] - nodesPos[e[1]])
            Gnx[e[0]][e[1]]['weight'] = weight
    else:
        for e in Gnx.edges():
            Gnx[e[0]][e[1]]['weight'] = 1  # not "true-room" scenario
    
    # Compute minimum spanning tree
    prunedWasnNX = nx.minimum_spanning_tree(
        Gnx,
        weight='weight',
        algorithm=algorithm
    )

    # Translate back to `WASN` object
    prunedWasnObj = update_wasn_object_from_nxgraph(wasnObj, prunedWasnNX)

    # Set node types
    for k in range(len(prunedWasnObj.wasn)):
        if len(prunedWasnObj.wasn[k].neighborsIdx) == 1:
            prunedWasnObj.wasn[k].nodeType = 'leaf'
    # Set root
    if forcedRoot is None:
        prunedWasnObj.set_tree_root()
    else:
        prunedWasnObj.rootIdx = forcedRoot
        prunedWasnObj.wasn[forcedRoot].nodeType = 'root'
    # Add graph orientation from leaves to root
    prunedWasnObj.orientate()

    if plotit:
        # Plot original and pruned WASN
        fig = visualise_pruning(wasnObj, prunedWasnObj)
        plt.show()

    return prunedWasnObj


def update_wasn_object_from_nxgraph(
    originalWASN: WASN,
    Gnx: nx.Graph
    ) -> WASN:
    """
    Updates connectivity parameters in a list of `Node` object instances
    based on a Networkx `Graph` object.

    Parameters
    ----------
    originalWASN : `WASN` object
        WASN under consideration.
    Gnx : nx.Graph object
        NetworkX graph object of updated WASN.

    Returns
    -------
    WASNout : `WASN` object
        Updated WASN.
    """
    with warnings.catch_warnings():
        # Avoiding the NetworkX FutureWarning:
        #   `adjacency_matrix will return a scipy.sparse array instead of a
        #   matrix in Networkx 3.0.`
        warnings.simplefilter(action='ignore', category=FutureWarning)
        #
        adjMat = nx.adjacency_matrix(Gnx).toarray()
    nNodes = adjMat.shape[0]

    WASNout = copy.deepcopy(originalWASN)
    allNodeIndices = np.arange(nNodes)
    for k in range(nNodes):
        WASNout.wasn[k].neighborsIdx = allNodeIndices[adjMat[:, k] > 0]

    WASNout.adjacencyMatrix = adjMat

    return WASNout


def visualise_pruning(wasnObj: WASN, prunedWasnObj: WASN):
    """
    Plots a 3D graph of the WASNs before/after pruning to a tree topology.

    Parameters
    ----------
    wasnObj : `WASN` object
        WASN (ad-hoc topology).
    prunedWasnObj : `WASN` object
        WASN (tree topology).

    Returns
    -------
    fig : plt figure handle
        Figure handle for further post-processing.
    """
    
    # Generate figure
    fig = plt.figure()
    fig.set_size_inches(8.5, 3.5)
    #
    axes = fig.add_subplot(1, 2, 1, projection='3d')
    wasnObj.plot_me(axes)
    axes.set_title('Original topology')
    #
    axes = fig.add_subplot(1, 2, 2, projection='3d')
    prunedWasnObj.plot_me(axes)
    axes.set_title('Pruned topology')

    return fig


def generate_graph_for_wasn(wasn: list[Node]) -> nx.Graph:
    """
    Generates a NetworkX `Graph` object from the `wasn` list of `Node` objects.
    
    Parameters
    ----------
    wasn : list of `Node` objects
        WASN under consideration.

    Returns
    -------
    Gnx : nx.Graph object
        NetworkX Graph object.
    """

    nNodes = len(wasn)
    adjMat = np.eye(nNodes)
    for k in range(nNodes):
        for q in range(nNodes):
            if k > q:
                if q in wasn[k].neighborsIdx:
                    adjMat[k, q] = 1
                    adjMat[q, k] = 1
                    

    Gnx = nx.from_numpy_array(adjMat)

    return Gnx


def get_stft(x, fs, win, ovlp, boundary='zeros'):
    """
    Derives time-domain signals' STFT representation
    given certain settings.

    Parameters
    ----------
    x : [N x C] np.ndarray (float)
        Time-domain signal(s).
    fs : int
        Sampling frequency [samples/s].
    win : np.ndarray[float]
        Analysis window.
    ovlp : float
        Amount of window overlap.
    boundary : str
        Boundary extension mode.

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
            return_onesided=True,
            boundary=boundary
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


def get_istft(x, fs, win, ovlp, boundary='zeros'):
    # Perform inverse operation as get_stft()

    if x.ndim == 2:
        x = x[:, :, np.newaxis]

    for channel in range(x.shape[-1]):

        t, tmp = sig.istft(
            x[:, :, channel],
            fs=fs,
            window=win,
            nperseg=len(win),
            noverlap=int(ovlp * len(win)),
            boundary=boundary
        )
        if channel == 0:
            out = np.zeros((tmp.shape[0], x.shape[-1]))
        out[:, channel] = tmp

    # Flatten array in case of single-channel data
    if x.shape[-1] == 1:
        out = np.array([i[0] for i in out])

    return out, t


def init_complex_filter(
        size,
        refIdx=0,
        initType='selectFirstSensor',
        fixedValue=0.,
        seed=0,
    ):
    """
    Returns an initialized STFT-domain filter vector,
    i.e., a selector of the reference sensor at node 1.
    
    Parameters
    ----------
    size : tuple
        Filter size.
    refIdx : int, optional
        Index of the reference sensor, by default 0.
    initType : str, optional
        Initialization type, by default 'selectFirstSensor'.
    fixedValue : float, optional
        Fixed value for the filter vector, by default 0. Used iff `initType`
        includes 'fixedValue'.
    seed : int, optional
        Random number generator seed, by default 0.

    Returns
    -------
    wInit : np.ndarray (complex)
        Initialized filter vector.
    """
    if initType == 'selectFirstSensor':
        wInit = np.zeros(size, dtype=complex)
        if len(size) == 3:
            wInit[:, :, refIdx] = 1
        elif len(size) == 2:
            wInit[:, refIdx] = 1
    elif initType == 'random':
        rng = np.random.default_rng(seed)
        wInit = (rng.random(size) - 0.5) + 1j * (rng.random(size) - 0.5)
    elif initType == 'fixedValue':
        wInit = np.full(size, fill_value=fixedValue, dtype=complex)
    elif initType == 'selectFirstSensor_andFixedValue':
        wInit = np.full(size, fill_value=fixedValue, dtype=complex)
        if len(size) == 3:
            wInit[:, :, refIdx] = 1
        elif len(size) == 2:
            wInit[:, refIdx] = 1
    return wInit


def init_covmats(
        dims: tuple,
        rng,
        covMatInitType,
        covMatRandomInitScaling,
        covMatEyeInitScaling
    ) -> np.ndarray:
    """
    Helper function to initialize the covariance matrices.
    
    Parameters
    ----------
    dims : tuple
        Covariance matrix dimensions.
    rng : np.random.Generator
        Random number generator.
    covMatInitType : str
        Covariance matrix initialization type.
    covMatRandomInitScaling : float
        Scaling factor for the random part of the covariance matrix.
    covMatEyeInitScaling : float
        Scaling factor for the identity part of the covariance matrix.

    Returns
    -------
    fullSlice : np.ndarray
        Initialized covariance matrix.
    """
    # Generate a basis random array for random initializations
    randArray = 2 * rng.random(dims) - 1 +\
        1j * (2 * rng.random(dims) - 1)
    if covMatInitType == 'fully_random':
        # Scale the random array
        fullSlice = covMatRandomInitScaling * randArray
    elif covMatInitType == 'eye_and_random':
        if len(dims) == 2:  # for single freq-bin slice
            eyePart = np.eye(dims[-1]) * covMatEyeInitScaling
        elif len(dims) == 3:  # for multiple bins
            eyePart = np.tile(
                np.eye(dims[-1]) * covMatEyeInitScaling,
                (dims[0], 1, 1)
            )
        # Scale the random array and add an identity matrix
        # to each slice.
        fullSlice = eyePart + covMatRandomInitScaling * randArray
    elif covMatInitType == 'eye':
        if len(dims) == 2:
            fullSlice = np.eye(dims[-1]) * covMatEyeInitScaling
        elif len(dims) == 3:
            fullSlice = np.tile(
                np.eye(dims[-1]) * covMatEyeInitScaling,
                (dims[0], 1, 1)
            )
    return fullSlice


def get_y_tilde_batch(
        tidanseFlag=False,
        yinSTFT=None,
        nNodes=0,
        wTildeExt=None,
        nSensorPerNode=None,
        neighbors=None,
        i=0,
        k=0,
        yinSTFT_s=None,
        yinSTFT_n=None,
        computeSpeechAndNoiseOnly=False,
        useThisFilter=None
    ):
    """
    Compute complete yTilde for all nodes, all frames, using
    current (external) DANSE filters. Works for both fully connected
    DANSE and TI-DANSE.
    """
    if tidanseFlag:  # TI-DANSE case
        # raise ValueError('TODO: `useThisFilter` parameter for TI-DANSE')
        etaMkBatch = np.zeros((
            yinSTFT[k].shape[0],
            yinSTFT[k].shape[1]
        ), dtype=complex)
        if computeSpeechAndNoiseOnly:
            etaMkBatch_s = copy.deepcopy(etaMkBatch)
            etaMkBatch_n = copy.deepcopy(etaMkBatch)
            # ^^^ `yinSTFT_s` and `yinSTFT_n` have the same shape
            # as `yinSTFT`.
        for idxNode in range(nNodes):
            if idxNode != k:  # only sum over the other nodes
                if useThisFilter is not None:
                    # Bypass `wTildeExt`
                    fusionFilter = useThisFilter[idxNode]
                else:
                    fusionFilter = wTildeExt[idxNode][:, i[idxNode], :]
                # TI-DANSE fusion vector
                # p = fusionFilter[:, :nSensorPerNode[idxNode]] /\
                #     fusionFilter[:, -1:]
                p = fusionFilter[:, :nSensorPerNode[idxNode]]
                # Compute sum
                etaMkBatch += np.einsum(   # <-- `+=` is important
                    'ij,ikj->ik',
                    p.conj(),
                    yinSTFT[idxNode]
                )
                if computeSpeechAndNoiseOnly:
                    etaMkBatch_s += np.einsum(   # <-- `+=` is important
                        'ij,ikj->ik',
                        p.conj(),
                        yinSTFT_s[idxNode]
                    )
                    etaMkBatch_n += np.einsum(   # <-- `+=` is important
                        'ij,ikj->ik',
                        p.conj(),
                        yinSTFT_n[idxNode]
                    )
        # Construct yTilde
        yTildeBatch = np.concatenate(
            (yinSTFT[k], etaMkBatch[:, :, np.newaxis]),
            axis=-1
        )
        if computeSpeechAndNoiseOnly:
            yTildeBatch_s = np.concatenate(
                (yinSTFT_s[k], etaMkBatch_s[:, :, np.newaxis]),
                axis=-1
            )
            yTildeBatch_n= np.concatenate(
                (yinSTFT_n[k], etaMkBatch_n[:, :, np.newaxis]),
                axis=-1
            )
    else:
        # Compute batch fused signals using current (external) DANSE filters
        zBatch = np.zeros((
            yinSTFT[k].shape[0],
            yinSTFT[k].shape[1],
            len(neighbors[k])
        ), dtype=complex)
        if computeSpeechAndNoiseOnly:
            zBatch_s = copy.deepcopy(zBatch)
            zBatch_n = copy.deepcopy(zBatch)
            # ^^^ ok because `yinSTFT_s` and `yinSTFT_n` have the same
            # shape as `yinSTFT`.
        for ii, idxNode in enumerate(neighbors[k]):
            if useThisFilter is not None:
                # Bypass `wTildeExt`
                fusionFilter = useThisFilter[idxNode]
            else:
                fusionFilter = wTildeExt[idxNode][:, i[idxNode], :]
            
            zBatch[:, :, ii] = np.einsum(
                'ij,ikj->ik',
                fusionFilter.conj(),
                yinSTFT[idxNode]
            )
            if computeSpeechAndNoiseOnly:
                zBatch_s[:, :, ii] = np.einsum(
                    'ij,ikj->ik',
                    fusionFilter.conj(),
                    yinSTFT_s[idxNode]
                )
                zBatch_n[:, :, ii] = np.einsum(
                    'ij,ikj->ik',
                    fusionFilter.conj(),
                    yinSTFT_n[idxNode]
                )
        # Construct yTilde
        yTildeBatch = np.concatenate(
            (yinSTFT[k], zBatch),
            axis=-1
        )
        if computeSpeechAndNoiseOnly:
            yTildeBatch_s = np.concatenate(
                (yinSTFT_s[k], zBatch_s),
                axis=-1
            )
            yTildeBatch_n = np.concatenate(
                (yinSTFT_n[k], zBatch_n),
                axis=-1
            )
    
    # Conditional exports
    if computeSpeechAndNoiseOnly:
        return yTildeBatch, yTildeBatch_s, yTildeBatch_n
    else:
        return yTildeBatch


# ----------------------------------
# 3 functions below:
# from https://alexwlchan.net/2019/finding-divisors-with-python/
# (accessed 2023.09.13)
def prime_factors(n):
    i = 2
    while i * i <= n:
        if n % i == 0:
            n /= i
            yield i
        else:
            i += 1

    if n > 1:
        yield n

def prod(iterable):
    result = 1
    for i in iterable:
        result *= i
    return result

def get_divisors(n):
    pf = prime_factors(n)

    pf_with_multiplicity = collections.Counter(pf)

    powers = [
        [factor ** i for i in range(count + 1)]
        for factor, count in pf_with_multiplicity.items()
    ]

    for prime_power_combo in itertools.product(*powers):
        yield prod(prime_power_combo)
# ----------------------------------