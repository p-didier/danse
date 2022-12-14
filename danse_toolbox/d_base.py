# Basic functions necessary for the good functioning of DANSE.
# -- Mostly surrounding the most important functions in `d_core.py`.
#
# ~created on 19.10.2022 by Paul Didier

import copy
import numpy as np
from numba import njit
import scipy.linalg as sla
import matplotlib.pyplot as plt
from danse_toolbox.d_classes import *
from scipy.signal._arraytools import zero_ext
from danse_toolbox.d_eval import DynamicMetricsParameters
from siggen.classes import WASNparameters


@dataclass
class DANSEeventInstant:
    t: float = 0.   # event time instant [s]
    nodes: np.ndarray = np.array([0])   # node(s) concerned
    type: list[str] = field(default_factory=list)   # event type
    bypass: list[bool] = field(default_factory=list)  
        # ^^^ if True, bypass event at that instant and for that node.
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
    # Printouts
    printout_profiler: bool = True      # controls printouts of Profiler.
    printout_eventsParser: bool = True      # controls printouts in
                                            # `events_parser()` function.
    printout_eventsParserNoBC: bool = False     # if True, do not print out the
                                                # broadcasts in event parser.
    printout_externalFilterUpdate: bool = True      # controls printouts of
                                                    # external filter updates.
    # Other
    bypassUpdates: bool = False   # if True, do not update filters.

@dataclass
class DANSEparameters(Hyperparameters):
    """
    Parameters for the DANSE algorithm.

    References
    ----------
    - [1] Bertrand, A., & Moonen, M. (2010). Distributed adaptive node-specific
    signal estimation in fully connected sensor networks???Part I: Sequential
    node updating. IEEE Transactions on Signal Processing, 58(10), 5277-5291.

    - [2] Bertrand, A., & Moonen, M. (2010). Distributed adaptive node-specific
    signal estimation in fully connected sensor networks???Part II: Simultaneous
    and asynchronous node updating. IEEE Transactions on Signal Processing,
    58(10), 5292-5306.

    """
    DFTsize: int = 1024    # DFT size
    WOLAovlp: float = .5   # WOLA window overlap [*100%]
    nodeUpdating: str = 'seq'   # node-updating strategy
        # - "seq": round-robin updating [1]
        # - "sim": simultaneous updating [2]
        # - "asy": asynchronous updating [2]
    broadcastType: str = 'wholeChunk'    # type of broadcast
        # -- 'wholeChunk': chunks of compressed signals in time-domain,
        # -- 'fewSamples': T(z)-approximation of WOLA compression process.
        # broadcast L ??? Ns samples at a time.
    winWOLAanalysis: np.ndarray = np.sqrt(np.hanning(DFTsize))      # window
    winWOLAsynthesis: np.ndarray = np.sqrt(np.hanning(DFTsize))     # window
    normFactWOLA: float = sum(winWOLAanalysis)  # (I)FFT normalization factor
    # T(z)-approximation | Sample-wise broadcasts
    upTDfilterEvery: float = 1. # [s] duration of pause between two 
                                    # consecutive time-domain filter updates.
    # SROs
    compensateSROs: bool = False    # if True, compensate for SROs
    estimateSROs: str = 'Oracle'    # SRO estimation method. If 'Oracle',
        # no estimation: using oracle if `compensateSROs == True`.
    cohDrift: CohDriftParameters = CohDriftParameters()
    # General
    performGEVD: bool = True    # if True, perform GEVD
    GEVDrank: int = 1           # GEVD rank
    timeBtwExternalFiltUpdates: float = 1.  # [s] bw. external filter updates.
    # ^ TODO: make that used only if the node-updating is sim/asy.
    alphaExternalFilters: float = .5    # exponential averaging constant
                                        # for external filter target update.
    t_expAvg50p: float = 2.     # [s] Time in the past at which the value is
                                # weighted by 50% via exponential averaging.
    # Desired signal estimation
    desSigProcessingType: str = 'wola'  # processing scheme used to compute
        # the desired signal estimates: "wola": WOLA synthesis,
                                    # "conv": T(z)-approximation.
    computeCentralised: bool = False    # if True, compute centralised
        # estimate (using all microphone signals in network). TODO: TODO: TODO: TODO: 
    computeLocal: bool = False  # if True, compute local estimate at each node TODO: TODO: TODO: TODO: 
    # Metrics
    dynMetrics: DynamicMetricsParameters = DynamicMetricsParameters()
    gammafwSNRseg: float = 0.2  # gamma exponent for fwSNRseg
    frameLenfwSNRseg: float = 0.03  # [s] time window duration for fwSNRseg
    minFiltUpdatesForMetrics: int = 10   # minimum number of DANSE
    # updates before start of speech enhancement metrics computation
    
    def __post_init__(self):
        """
        Adapt some fields depending on the value of others, after 
        dataclass instance initialisation.
        """
        self.Ns = int(self.DFTsize * (1 - self.WOLAovlp))
        if self.broadcastType == 'wholeChunk':
            self.broadcastLength = self.Ns
        elif self.broadcastType == 'fewSamples':
            self.broadcastLength = 1

    def get_wasn_info(self, wasn: WASNparameters):
        """
        Adds useful info to DANSEparameters object from WASNparameters
        object. 
        """
        self.nNodes = wasn.nNodes
        self.nSensorPerNode = wasn.nSensorPerNode
        self.referenceSensor = wasn.referenceSensor

def prep_sigs_for_FFT(y, N, Ns, t):
    """
    Zero-padding and signals length adaptation to ensure correct
    FFT/IFFT operation. Based on FFT implementation by `scipy.signal` module.
    -- Based on `prep_for_ffts` by Paul Didier
    (`01_algorithms/01_NR/02_distributed/danse_utilities/setup.py`).
    
    Parameters
    ----------
    y : [Nt x Nsensors] np.ndarray (float)
        The microphone signals.
    N : int
        WOLA-DANSE frame size [samples].
    Ns : int
        Number of new samples per frame (`N * (1 - ovlp)`,
        with `ovlp` the WOLA window overlap) [samples].
    t : [N x 1] np.ndarray (float)
        Sensor-specific time stamps vector.

    Returns
    -------
    yout : np.ndarray (float)
        Prepped signals.
    tout : np.ndarray (float)
        Corresponding time stamps.
    nadd : int
        Number of zeros added at the of signal after
        frame-extension (step 2 below).
    """

    # 1) Extend signal on both ends to ensure that the first frame is centred
    # at t = 0 -- see <scipy.signal.stft>'s `boundary` argument
    # (default: `zeros`).
    y = zero_ext(y, N // 2, axis=0)
    # --- Also adapt timeInstants vector
    # TODO: what if clock jitter?
    dt = np.diff(t)[0]   # delta t between each time instant
    tpre = np.linspace(start=-dt*(N//2), stop=-dt, num=N//2)
    tpost = np.linspace(start=t[-1]+dt, stop=t[-1]+dt*(N//2), num=N//2)
    t = np.concatenate((tpre, t, tpost), axis=0)

    # 2) Zero-pad signal if necessary to include an
    # integer number of frames in the signal.
    nadd = 0
    if not (y.shape[0] - N) % Ns == 0:
        # vvv See <scipy.signal.stft>'s `padded` argument (default: `True`)
        nadd = (-(y.shape[0] - N) % Ns) % N
        print(f'Padding {nadd} zeros to the signals in order to fit FFT size')
        y = np.concatenate((y, np.zeros([nadd, y.shape[-1]])), axis=0)
        # Adapt time vector too
        # TODO: what if clock jitter?
        tzp = np.linspace(start=t[-1] + dt, stop=t[-1] + dt * nadd, num=nadd)
        t = np.concatenate((t, tzp), axis=0)
        if not (y.shape[0] - N) % Ns == 0:   # double-check
            raise ValueError('There is a problem with the zero-padding...')

    # Prepare for output 
    yout = copy.copy(y)
    tout = copy.copy(t)

    return yout, tout, nadd


def initialize_events(timeInstants: np.ndarray, p: DANSEparameters):
    """
    Returns the matrix the columns of which to loop over in SRO-affected
    simultaneous DANSE. For each event instant, the matrix contains the instant
    itself (in [s]), the node indices concerned by this instant, and the
    corresponding event type ("bc" for broadcast, "up" for update).
    
    Parameters
    ----------
    timeInstants : [Nt x Nn] np.ndarray (floats)
        Time instants corresponding to the samples of each of the `Nn` nodes.
    p : DANSEparameters object
        Parameters.

    Returns
    -------
    outputEvents : [Ne x 1] list of DANSEeventInstant objects
        Event instants matrix.
    """
    
    # Make sure time stamps matrix is indeed a matrix, correctly oriented
    if timeInstants.ndim != 2:
        if timeInstants.ndim == 1:
            timeInstants = timeInstants[:, np.newaxis]
        else:
            raise ValueError('Unexpected dimensions for `timeInstants`.')
    if timeInstants.shape[0] < timeInstants.shape[1]:
        timeInstants = timeInstants.T

    # Number of nodes
    nNodes = timeInstants.shape[1]

    # Check for clock jitter and save sampling frequencies
    fs = np.zeros(nNodes)
    for k in range(nNodes):
        deltas = np.diff(timeInstants[:, k])
        # vvv Allowing computer precision errors down to 1e-4*mean delta.
        precision = int(np.ceil(np.abs(np.log10(np.mean(deltas) / 1e6))))
        if len(np.unique(np.round(deltas, precision))) > 1:
            raise ValueError(f'[NOT IMPLEMENTED] Clock jitter detected: \
                {len(np.unique(np.round(deltas, precision)))} different \
                    sample intervals detected for node {k+1}.')
        # np.round(): not going below 1 PPM precision for typical fs >= 8 kHz.
        fs[k] = np.round(1 / np.unique(np.round(deltas, precision))[0], 3)

    # Check consistency
    if p.nodeUpdating == 'sim' and any(fs != fs[p.referenceSensor]):
        raise ValueError('Simultaneous node-updating is impossible\
            in the presence of SROs.')

    # Total signal duration [s] per node (after truncation during sig. gen.).
    Ttot = timeInstants[-1, :]
    
    # Useful renaming (compact code)
    Ndft = p.DFTsize
    Ns = p.Ns 
    Lbc = p.broadcastLength
    bcType = p.broadcastType
    efficient = p.efficientSpSBC
    
    # Expected number of DANSE update per node over total signal length
    numUpInTtot = np.floor(Ttot * fs / Ns)
    # Expected DANSE update instants
    upInstants = [
        np.arange(np.ceil(Ndft / Ns),
        int(numUpInTtot[k])) * Ns/fs[k] for k in range(nNodes)
    ]  
    # ^ note that we only start updating when we have enough samples.
    
    # Expected number of broadcasts per node over total signal length
    numBcInTtot = np.floor(Ttot * fs / Lbc)
    # Get expected broadcast instants
    if 'wholeChunk' in bcType:
        bcInstants = [
            np.arange(Ndft/Lbc, int(numBcInTtot[k])) * Lbc/fs[k]\
                for k in range(nNodes)
        ]
        # ^ note that we only start broadcasting when we have enough
        # samples to perform compression.
    elif 'fewSamples' in bcType:
        if efficient:
            # Combine update instants
            combinedUpInstants = list(upInstants[0])
            for k in range(nNodes):
                if k > 0:
                    for ii in range(len(upInstants[k])):
                        if upInstants[k][ii] not in combinedUpInstants:
                            combinedUpInstants.append(upInstants[k][ii])
            combinedUpInstants = np.sort(np.array(combinedUpInstants))
            # Same BC instants for all nodes
            bcInstants = [combinedUpInstants for _ in range(nNodes)]
        else:
            bcInstants = [
                np.arange(1, int(numBcInTtot[k])) * Lbc/fs[k]\
                    for k in range(nNodes)
            ]
            # ^ note that we start broadcasting sooner:
            # when we have `L` samples, enough for linear convolution.
    
    # Build event matrix
    outputEvents = build_events_matrix(
        upInstants,
        bcInstants,
        p.nodeUpdating
    )

    return outputEvents, fs


def build_events_matrix(up_t, bc_t, nodeUpdating, visualizeUps=False):
    """
    Sub-function of `get_events_matrix`, building the events matrix
    from the update and broadcast instants.
    
    Parameters
    ----------
    up_t : [nNodes x 1] list of np.ndarrays (float)
        Update instants per node [s].
    bc_t : [nNodes x 1] list of np.ndarrays (float)
        Broadcast instants per node [s].

    Returns
    -------
    outputEvents : [Ne x 1] list of DANSEeventInstant objects
        Event instants matrix.
    """
    # Useful variables
    nNodes = len(up_t)

    numUniqueUpInstants = sum(
        [len(np.unique(up_t[k])) for k in range(nNodes)]
    )
    # Number of unique broadcast instants across the WASN
    numUniqueBcInstants = sum(
        [len(np.unique(bc_t[k])) for k in range(nNodes)]
    )
    # Number of unique update _or_ broadcast instants across the WASN
    numEventInstants = numUniqueBcInstants + numUniqueUpInstants

    # Arrange into matrix
    flattenedUpInstants = np.zeros((numUniqueUpInstants, 3))
    flattenedBcInstants = np.zeros((numUniqueBcInstants, 3))
    for k in range(nNodes):
        idxStart_u = sum([len(up_t[q]) for q in range(k)])
        idxEnd_u = idxStart_u + len(up_t[k])
        flattenedUpInstants[idxStart_u:idxEnd_u, 0] = up_t[k]
        flattenedUpInstants[idxStart_u:idxEnd_u, 1] = k
        flattenedUpInstants[:, 2] = 1    # event reference "1" for updates

        idxStart_b = sum([len(bc_t[q]) for q in range(k)])
        idxEnd_b = idxStart_b + len(bc_t[k])
        flattenedBcInstants[idxStart_b:idxEnd_b, 0] = bc_t[k]
        flattenedBcInstants[idxStart_b:idxEnd_b, 1] = k
        flattenedBcInstants[:, 2] = 0    # event reference "0" for broadcasts
    # Combine
    eventInstants = np.concatenate(
        (flattenedUpInstants, flattenedBcInstants),
        axis=0
    )
    # Sort
    idxSort = np.argsort(eventInstants[:, 0], axis=0)
    eventInstants = eventInstants[idxSort, :]
    # Group
    outputInstants: list[DANSEeventInstant] = []
    eventIdx = 0    # init while-loop
    nodesConcerned = []             # init
    eventTypesConcerned = []        # init
    lastUpNode = -1     # index of latest updating node
    # ^^^ (init. at -1 so that `lastUpNode + 1 == 0`)
    while eventIdx < numEventInstants:

        currInstant = eventInstants[eventIdx, 0]
        nodesConcerned.append(int(eventInstants[eventIdx, 1]))
        eventTypesConcerned.append(int(eventInstants[eventIdx, 2]))

        # Check whether the next instant is the same and
        # should be grouped with the current instant.
        if eventIdx < numEventInstants - 1:
            nextInstant = eventInstants[eventIdx + 1, 0]
            while currInstant == nextInstant:
                eventIdx += 1
                currInstant = eventInstants[eventIdx, 0]
                nodesConcerned.append(int(eventInstants[eventIdx, 1]))
                eventTypesConcerned.append(int(eventInstants[eventIdx, 2]))
                # Check whether the next instant is the same and
                # should be grouped with the current instant.
                if eventIdx < numEventInstants - 1:
                    nextInstant = eventInstants[eventIdx + 1, 0]
                else:
                    eventIdx += 1
                    break
            else:
                eventIdx += 1
        else:
            eventIdx += 1

        # Sort events at current instant
        nodesConcerned = np.array(nodesConcerned, dtype=int)
        eventTypesConcerned = np.array(eventTypesConcerned, dtype=int)
        # 1) First broadcasts, then updates
        originalIndices = np.arange(len(nodesConcerned))
        idxUpEvent = originalIndices[eventTypesConcerned == 1]
        idxBcEvent = originalIndices[eventTypesConcerned == 0]
        # 2) Order by node index
        if len(idxUpEvent) > 0:
            idxUpEvent = idxUpEvent[np.argsort(nodesConcerned[idxUpEvent])]
        if len(idxBcEvent) > 0:
            idxBcEvent = idxBcEvent[np.argsort(nodesConcerned[idxBcEvent])]
        # 3) Re-combine
        indices = np.concatenate((idxBcEvent, idxUpEvent))
        # 4) Sort
        nodesConcerned = nodesConcerned[indices]
        eventTypesConcerned = eventTypesConcerned[indices]

        # Event types (broadcast or update)
        types = ['bc' if ii == 0 else 'up' for ii in eventTypesConcerned]

        # Address node-updating strategy
        bypass = [False for _ in eventTypesConcerned]  # default: no bypass
        if 'up' in types and nodeUpdating == 'seq':
            lastUpNodeUpdated = lastUpNode
            for ii in range(len(eventTypesConcerned)):
                if types[ii] == 'up':
                    if nodesConcerned[ii] == np.mod(lastUpNode + 1, nNodes):
                        # Increment last updating node index
                        lastUpNodeUpdated = nodesConcerned[ii]
                        # break  # break `for`-loop
                    else:
                        # Bypass update in other nodes
                        bypass[ii] = True
            lastUpNode = lastUpNodeUpdated

        # Build events matrix
        outputInstants.append(DANSEeventInstant(
            t=currInstant,
            nodes=nodesConcerned,
            type=types,
            bypass=bypass
        ))

        nodesConcerned = []         # reset list
        eventTypesConcerned = []    # reset list

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
                    if currEvents.bypass[ii]:
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


def local_chunk_for_broadcast(y, t, fs, N):
    """
    Extract correct chunk of local signals for broadcasting.
    
    Parameters
    ----------
    y : [Ntot x Mk] np.ndarray (float)
        Time-domain locally recorded signal (at `Mk` sensors).
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
    idxEnd = int(np.floor(np.round(t * fs, 5)))
    # vvv -- don't go into negative sample indices!
    idxBeg = np.amax([idxEnd - N, 0])
    chunk = y[idxBeg:idxEnd, :]
    # Pad zeros at beginning if needed
    if idxEnd - idxBeg < N:
        chunk = np.concatenate((
            np.zeros((N - chunk.shape[0], chunk.shape[1])), chunk
        ))

    return chunk


def local_chunk_for_update(y, t, fs, bd, Ndft, Ns):
    """
    Extract correct chunk of local signals for DANSE updates.
    
    Parameters
    ----------
    y : [Ntot x Mk] np.ndarray (float)
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
        idxEnd = int(np.floor(np.round(t * fs, 5)))
    # Broadcast scheme: block-wise, in time-domain
    elif bd == 'wholeChunk':
        # `N - Ns` samples delay due to time-domain WOLA
        idxEnd = int(np.floor(np.round(t * fs, 5))) - (Ndft - Ns)

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
    
    # vvv -- important to go back to original input dimensionalitybefore FFT
    # (bias of np.fft.fft with (n,1)-dimensioned input).
    if flagSingleton:
        x = np.squeeze(x)

    # Back to time-domain
    xout = np.fft.ifft(x, n, axis=0)

    if axis == 1:
        xout = xout.T

    # Check before output
    if not np.allclose(np.fft.fft(xout, n, axis=0), x):
        raise ValueError('Issue in return to time-domain')

    return xout


def fill_buffers_whole_chunk(k, neighs, zBuffer, zLocalK):
    """
    Fills neighbors nodes' buffers, using frequency domain data.
    Data comes from compression via function `danse_compression_freq_domain`.
    
        Parameters
    ----------
    k : int
        Current node index.
    neighs : [numNodes x 1] list of [nNeighbours[n] x 1] lists (int)
        Network indices of neighbours, per node.
    zBuffer : [numNodes x 1] list of [nNeighbours[n] x 1] ...
            ... lists of [variable length] np.ndarrays (complex)
        Compressed signals buffers for each node and its neighbours.
    zLocal : [N x 1] np.ndarray (float)
        Latest compressed local signals to be broadcasted from node `k`.

    Returns
    -------
    zBuffer : [numNodes x 1] list of [nNeighbours[n] x 1] ...
            ... lists of [N x 1] np.ndarrays (complex)
        Updated compressed signals buffers for each node and its neighbours.
    """

    # Loop over neighbors of `k`
    for idxq in range(len(neighs[k])):
        # Network-wide index of node `q` (one of node `k`'s neighbors)
        q = neighs[k][idxq]
        idxKforNeighborQ = [i for i, x in enumerate(neighs[q]) if x == k]
        # Node `k`'s "neighbor index", from node `q`'s perspective
        idxKforNeighborQ = idxKforNeighborQ[0]
        # Fill in neighbor's buffer
        zBuffer[q][idxKforNeighborQ] = zLocalK
        
    return zBuffer


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
            raise ValueError(f'Incoherence: float `zLocalK` but L = {L}')

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

        stop = 1
        
    return zBuffer


def events_parser(
    events: DANSEeventInstant,
    startUpdates,
    printouts=False,
    doNotPrintBCs=False):
    """
    Printouts to inform user of DANSE events.
    
    Parameters
    ----------
    events : [Ne x 1] list of [3 x 1] np.ndarrays containing lists of floats
        Event instants matrix. One column per event instant.
        Output of `get_events_matrix` function.
    startUpdates : list of bools
        Node-specific flags to indicate whether DANSE updates have started. 
    printouts : bool
        If True, inform user about current events after parsing. 
    doNotPrintBCs : bool
        If True, do not print the broadcast events
        (only used if `printouts == True`).
    """

    if printouts:
        if 'up' in events.type:
            txt = f't={np.round(events.t, 3)}s -- '
            updatesTxt = 'Updating nodes: '
            if doNotPrintBCs:
                broadcastsTxt = ''
            else:
                broadcastsTxt = 'Broadcasting nodes: '
            # vvv -- little flag to add a comma (`,`) at the right spot.
            flagCommaUpdating = False
            for idxEvent in range(len(events.type)):
                k = int(events.nodes[idxEvent])   # node index
                if events.type[idxEvent] == 'bc' and not doNotPrintBCs:
                    if idxEvent > 0:
                        broadcastsTxt += ','
                    broadcastsTxt += f'{k + 1}'
                elif events.type[idxEvent] == 'up':
                    # Only print if the node actually has started updating
                    # (i.e. there has been sufficiently many autocorrelation
                    # matrices updates since the start of recording).
                    if startUpdates[k]:
                        if not flagCommaUpdating:
                            flagCommaUpdating = True
                        else:
                            updatesTxt += ','
                        updatesTxt += f'{k + 1}'
            print(txt + broadcastsTxt + '; ' + updatesTxt)


def danse_compression_whole_chunk(yq, wHat, h, f, zqPrevious=None):
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

    Returns
    -------
    zqHat : [N/2 x 1] np.ndarray (complex)
        Frequency-domain compressed signal for current frame.
    zq : [N x 1] np.ndarray (float)
        Time-domain latest WOLA chunk of compressed signal (after OLA).
    """

    # Check for single-sensor case
    flagSingleSensor = False
    if wHat.shape[-1] == 1:
        wHat = np.squeeze(wHat)
        yq = np.squeeze(yq)
        flagSingleSensor = True
    
    # Transfer local observations to frequency domain
    n = len(yq)     # DFT order

    # WOLA analysis stage
    if flagSingleSensor:
        yqHat = np.fft.fft(np.squeeze(yq) * h, n, axis=0)
        # Keep only positive frequencies
        yqHat = yqHat[:int(n/2 + 1)]
        # Apply linear combination to form compressed signal.
        # -- single sensor = simple element-wise multiplication.
        zqHat = wHat.conj() * yqHat
    else:
        yqHat = np.fft.fft(np.squeeze(yq) * h[:, np.newaxis], n, axis=0)
        # Keep only positive frequencies
        yqHat = yqHat[:int(n/2 + 1), :]
        # Apply linear combination to form compressed signal.
        zqHat = np.einsum('ij,ij->i', wHat.conj(), yqHat)

    # WOLA synthesis stage
    if zqPrevious is not None:
        # IDFT
        zqCurr = back_to_time_domain(zqHat, n, axis=0)
        zqCurr = np.real_if_close(zqCurr)
        zqCurr *= f    # multiply by synthesis window

        if not np.any(zqPrevious):
            # No previous frame, keep current frame
            zq = zqCurr
        else:
            # Overlap-add
            zq = np.zeros(n)
            # TODO: consider case with multiple neighbor nodes
            # (`len(zqPrevious) > 1`).
            zq[:(n // 2)] = zqPrevious[-(n // 2):]
            zq += zqCurr
    else:
        zq = None
    
    return zqHat, zq


def danse_compression_few_samples(
    yq, wqqHat, L, wIRprevious,
    winWOLAanalysis, winWOLAsynthesis, R, 
    updateBroadcastFilter=False):
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
    R : int
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
            R
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
    w, y, win, normFactWOLA,
    dChunk, Ns, yTD
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
        dChunCurr = normFactWOLA * win *\
            back_to_time_domain(dhatCurr, len(win))
        # Overlap and add construction of output time-domain signal
        if len(dChunk) < len(win):
            dChunk += np.real_if_close(dChunCurr[-len(dChunk):])
        else:
            dChunk += np.real_if_close(dChunCurr)

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