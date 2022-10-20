# Basic functions necessary for the good functioning of DANSE.
# -- Mostly surrounding the most important functions in `d_core.py`.
#
# ~created on 19.10.2022 by Paul Didier

import copy
from platform import node
import numpy as np
from danse.danse_toolbox.d_classes import *
from danse.danse_toolbox.d_core import dist_fct_approx
from scipy.signal._arraytools import zero_ext


def prep_sigs_for_FFT(y, N, Ns, t):
    """
    Zero-padding and signals length adaptation to ensure correct FFT/IFFT operation.
    Based on FFT implementation by `scipy.signal` module.
    -- Based on `prep_for_ffts` by Paul Didier (`01_algorithms/01_NR/02_distributed/danse_utilities/setup.py`)
    
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

    # 1) Extend signal on both ends to ensure that the first frame is centred on t = 0 -- see <scipy.signal.stft>'s `boundary` argument (default: `zeros`)
    y = zero_ext(y, N // 2, axis=0)
    # --- Also adapt timeInstants vector
    dt = np.diff(t)[0]   # delta t between each time instant   # TODO what if clock jitter?
    tpre = np.linspace(start= - dt * (N // 2), stop=-dt, num=N // 2)
    tpost = np.linspace(start= t[-1] + dt, stop=t[-1] + dt * (N // 2), num=N // 2)
    t = np.concatenate((tpre, t, tpost), axis=0)

    # 2) Zero-pad signal if necessary to include an integer number of frames in the signal
    nadd = 0
    if not (y.shape[0] - N) % Ns == 0:
        nadd = (-(y.shape[0] - N) % Ns) % N  # see <scipy.signal.stft>'s `padded` argument (default: `True`)
        print(f'Padding {nadd} zeros to the signals in order to fit FFT size')
        y = np.concatenate((y, np.zeros([nadd, y.shape[-1]])), axis=0)
        # Adapt time vector too
        tzp = np.linspace(start=t[-1] + dt, stop=t[-1] + dt * nadd, num=nadd)     # TODO what if clock jitter?
        t = np.concatenate((t, tzp), axis=0)
        if not (y.shape[0] - N) % Ns == 0:   # double-check
            raise ValueError('There is a problem with the zero-padding...')

    # Prepare for output 
    yout = copy.copy(y)
    tout = copy.copy(t)

    return yout, tout, nadd


def initialize_events(timeInstants, N, Ns, L, bd, efficient=False):
    """Returns the matrix the columns of which to loop over in SRO-affected simultaneous DANSE.
    For each event instant, the matrix contains the instant itself (in [s]),
    the node indices concerned by this instant, and the corresponding event
    flag: "0" for broadcast, "1" for update, "2" for end of signal. 
    
    Parameters
    ----------
    timeInstants : [Nt x Nn] np.ndarray (floats)
        Time instants corresponding to the samples of each of the Nn nodes in the network.
    N : int
        Number of samples used for compression / for updating the DANSE filters.
    Ns : int
        Number of new samples per time frame (used in SRO-free sequential DANSE with frame overlap) (Ns < N).
    L : int
        Number of (compressed) signal samples to be broadcasted at a time to other nodes.
    bd : str
        Inter-node data broadcasting domain:
        -- 'wholeChunk_td': broadcast whole chunks of compressed signals in the time-domain,
        -- 'wholeChunk_fd': broadcast whole chunks of compressed signals in the WOLA-domain,
        -- 'fewSamples_td': linear-convolution approximation of WOLA compression process, broadcast L ≪ Ns samples at a time.
    efficient : bool
        If True, create "efficient" events. Only changing the `bd == 'fewSamples_td'` case.
        Avoids generation of unnecessary one-sample broadcast events --> saves a lot of computation time.

    Returns
    -------
    outputEvents : [Ne x 1] list of [3 x 1] np.ndarrays containing lists of floats
        Event instants matrix. One column per event instant.
    fs : [Nn x 1] list of floats
        Sampling frequency of each node.
    --------------------- vvv UNUSED vvv ---------------------
    initialTimeBiases : [Nn x 1] np.ndarray (floats)
        [s] Initial time difference between first update at node `row index`
        and latest broadcast instant at node `column index` (diagonal elements are
        all set to zero: there is no bias w.r.t. locally recorded data).
    """

    # Make sure time stamps matrix is indeed a matrix, correctly oriented
    if timeInstants.ndim != 2:
        if timeInstants.ndim == 1:
            timeInstants = timeInstants[:, np.newaxis]
        else:
            raise ValueError('Unexpected number of dimensions for input `timeInstants`.')
    if timeInstants.shape[0] < timeInstants.shape[1]:
        timeInstants = timeInstants.T

    # Number of nodes
    nNodes = timeInstants.shape[1]

    # Check for clock jitter and save sampling frequencies
    fs = np.zeros(nNodes)
    for k in range(nNodes):
        deltas = np.diff(timeInstants[:, k])
        precision = int(np.ceil(np.abs(np.log10(np.mean(deltas) / 1e6))))  # allowing computer precision errors down to 1e-4*mean delta.
        if len(np.unique(np.round(deltas, precision))) > 1:
            raise ValueError(f'[NOT IMPLEMENTED] Clock jitter detected: {len(np.unique(np.round(deltas, precision)))} different sample intervals detected for node {k+1}.')
        fs[k] = np.round(1 / np.unique(np.round(deltas, precision))[0], 3)  # np.round(): not going below 1PPM precision for typical fs >= 8 kHz

    # Total signal duration [s] per node (after truncation during signal generation)
    Ttot = timeInstants[-1, :]
    
    # Get expected DANSE update instants
    numUpdatesInTtot = np.floor(Ttot * fs / Ns)   # expected number of DANSE update per node over total signal length
    updateInstants = [np.arange(np.ceil(N / Ns), int(numUpdatesInTtot[k])) * Ns/fs[k] for k in range(nNodes)]  # expected DANSE update instants
    #                               ^ note that we only start updating when we have enough samples
    
    numBroadcastsInTtot = np.floor(Ttot * fs / L)   # expected number of broadcasts per node over total signal length
    # Get expected broadcast instants
    if 'wholeChunk' in bd:
        broadcastInstants = [np.arange(N/L, int(numBroadcastsInTtot[k])) * L/fs[k] for k in range(nNodes)]
        #                              ^ note that we only start broadcasting when we have enough samples to perform compression
    elif 'fewSamples' in bd:
        if efficient:
            # Combine update instants
            combinedUpdateInstants = list(updateInstants[0])
            for k in range(nNodes):
                if k > 0:
                    for ii in range(len(updateInstants[k])):
                        if updateInstants[k][ii] not in combinedUpdateInstants:
                            combinedUpdateInstants.append(updateInstants[k][ii])
            combinedUpdateInstants = np.sort(np.array(combinedUpdateInstants))
            broadcastInstants = [combinedUpdateInstants for _ in range(nNodes)]  # same BC instants for all nodes
        else:
            broadcastInstants = [np.arange(1, int(numBroadcastsInTtot[k])) * L/fs[k] for k in range(nNodes)]
            #                              ^ note that we start broadcasting sooner: when we have `L` samples, enough for linear convolution

    # Build event matrix
    outputEvents = build_events_matrix(updateInstants, broadcastInstants)

    return outputEvents, fs


def build_events_matrix(updateInstants, broadcastInstants):
    """Sub-function of `get_events_matrix`, building the events matrix
    from the update and broadcast instants.
    
    Parameters
    ----------
    updateInstants : [nNodes x 1] list of np.ndarrays (float)
        Update instants per node [s].
    broadcastInstants : [nNodes x 1] list of np.ndarrays (float)
        Broadcast instants per node [s].

    Returns
    -------
    outputEvents : [Ne x 1] list of [3 x 1] np.ndarrays containing lists of floats
        Event instants matrix. One column per event instant.
    """

    nNodes = len(updateInstants)

    numUniqueUpdateInstants = sum([len(np.unique(updateInstants[k])) for k in range(nNodes)])
    # Number of unique broadcast instants across the WASN
    numUniqueBroadcastInstants = sum([len(np.unique(broadcastInstants[k])) for k in range(nNodes)])
    # Number of unique update _or_ broadcast instants across the WASN
    numEventInstants = numUniqueBroadcastInstants + numUniqueUpdateInstants

    # Arrange into matrix
    flattenedUpdateInstants = np.zeros((numUniqueUpdateInstants, 3))
    flattenedBroadcastInstants = np.zeros((numUniqueBroadcastInstants, 3))
    for k in range(nNodes):
        idxStart_u = sum([len(updateInstants[q]) for q in range(k)])
        idxEnd_u = idxStart_u + len(updateInstants[k])
        flattenedUpdateInstants[idxStart_u:idxEnd_u, 0] = updateInstants[k]
        flattenedUpdateInstants[idxStart_u:idxEnd_u, 1] = k
        flattenedUpdateInstants[:, 2] = 1    # event reference "1" for updates

        idxStart_b = sum([len(broadcastInstants[q]) for q in range(k)])
        idxEnd_b = idxStart_b + len(broadcastInstants[k])
        flattenedBroadcastInstants[idxStart_b:idxEnd_b, 0] = broadcastInstants[k]
        flattenedBroadcastInstants[idxStart_b:idxEnd_b, 1] = k
        flattenedBroadcastInstants[:, 2] = 0    # event reference "0" for broadcasts
    # Combine
    eventInstants = np.concatenate((flattenedUpdateInstants, flattenedBroadcastInstants), axis=0)
    # Sort
    idxSort = np.argsort(eventInstants[:, 0], axis=0)
    eventInstants = eventInstants[idxSort, :]
    # Group
    outputEvents: list[DANSEeventInstant] = []
    eventIdx = 0    # init while-loop
    nodesConcerned = []             # init
    eventTypesConcerned = []        # init
    while eventIdx < numEventInstants:

        currInstant = eventInstants[eventIdx, 0]
        nodesConcerned.append(int(eventInstants[eventIdx, 1]))
        eventTypesConcerned.append(int(eventInstants[eventIdx, 2]))

        if eventIdx < numEventInstants - 1:   # check whether the next instant is the same and should be grouped with the current instant
            nextInstant = eventInstants[eventIdx + 1, 0]
            while currInstant == nextInstant:
                eventIdx += 1
                currInstant = eventInstants[eventIdx, 0]
                nodesConcerned.append(int(eventInstants[eventIdx, 1]))
                eventTypesConcerned.append(int(eventInstants[eventIdx, 2]))
                if eventIdx < numEventInstants - 1:   # check whether the next instant is the same and should be grouped with the current instant
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
        idxUpdateEvent = originalIndices[eventTypesConcerned == 1]
        idxBroadcastEvent = originalIndices[eventTypesConcerned == 0]
        # 2) Order by node index
        if len(idxUpdateEvent) > 0:
            idxUpdateEvent = idxUpdateEvent[np.argsort(nodesConcerned[idxUpdateEvent])]
        if len(idxBroadcastEvent) > 0:
            idxBroadcastEvent = idxBroadcastEvent[np.argsort(nodesConcerned[idxBroadcastEvent])]
        # 3) Re-combine
        indices = np.concatenate((idxBroadcastEvent, idxUpdateEvent))
        # 4) Sort
        nodesConcerned = nodesConcerned[indices]
        eventTypesConcerned = eventTypesConcerned[indices]

        # Build events matrix
        outputEvents.append(DANSEeventInstant(
            t=currInstant,
            nodes=nodesConcerned,
            type=['bc' if ii == 0 else 'up' for ii in eventTypesConcerned]
        ))
        nodesConcerned = []         # reset
        eventTypesConcerned = []    # reset

    return outputEvents


def local_chunk_for_broadcast(y, t, fs, N):
    """Extract correct chunk of local signals for broadcasting.
    
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

    idxEnd = int(np.floor(np.round(t * fs, 5)))  # np.round() used to avoid issues with previous rounding/precision errors (see Word journal week 32, THU, 2022)
    idxBeg = np.amax([idxEnd - N, 0])   # don't go into negative sample indices!
    chunk = y[idxBeg:idxEnd, :]
    # Pad zeros at beginning if needed
    if idxEnd - idxBeg < N:
        chunk = np.concatenate((np.zeros((N - chunk.shape[0], chunk.shape[1])), chunk))

    return chunk


def local_chunk_for_update(y, t, fs, p: DANSEparameters):
    """Extract correct chunk of local signals for DANSE updates.
    
    Parameters
    ----------
    y : [Ntot x Mk] np.ndarray (float)
        Time-domain locally recorded signal (at `Mk` sensors).
    t : float
        Current time instant [s].
    fs : int or float
        Transmitting node's sampling frequency [Hz].
    p : DANSEparameters object
        DANSE parameters.

    Returns
    -------
    chunk : [N x Mk] np.ndarray (float)
        Time chunk of local sensor signals.
    idxBeg : int
        Start index of chunk (w.r.t. `y`).
    idxEnd : int
        End index of chunk (w.r.t. `y`).
    """

    # Useful renaming
    bd = p.broadcastType
    Ndft = p.DFTsize
    Ns = p.Ns

    # Broadcast scheme: block-wise, in freq.-domain
    # <or> Broadcast scheme: few samples at a time, in time-domain
    if bd in ['wholeChunk_fd', 'fewSamples_td']:
        idxEnd = int(np.floor(np.round(t * fs, 5)))
    # Broadcast scheme: block-wise, in time-domain
    elif bd == 'wholeChunk_td':
        idxEnd = int(np.floor(np.round(t * fs, 5))) - (Ndft - Ns)     # `N - Ns` samples delay due to time-domain WOLA

    idxBeg = np.amax([idxEnd - Ndft, 0])       # don't go into negative sample indices!
    chunk = y[idxBeg:idxEnd, :]
    # Pad zeros at beginning if needed (occurs at algorithm's startup)
    if idxEnd - idxBeg < Ndft:
        chunk = np.concatenate((np.zeros((Ndft - chunk.shape[0], chunk.shape[1])), chunk))

    return chunk, idxBeg, idxEnd


def back_to_time_domain(x, n, axis=0):
    """Performs an IFFT after pre-processing of a frequency-domain
    signal chunk.
    
    Parameters
    ----------
    x : np.ndarray of complex
        Frequency-domain signal to be transferred back to time domain.
    n : int
        IFFT order.
    axis : int (0 or 1)
        Array axis where to perform IFFT -- not implemented for more than 2-D arrays.

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
        raise ValueError('`x` should be (n/2 + 1)-dimensioned along the IFFT axis.')

    x[0, :] = x[0, :].real      # Set DC to real value
    x[-1, :] = x[-1, :].real    # Set Nyquist to real value
    x = np.concatenate((x, np.flip(x[:-1, :].conj(), axis=0)[:-1, :]), axis=0)
    
    if flagSingleton: # important to go back to original input dimensionality before FFT (bias of np.fft.fft with (n,1)-dimensioned input)
        x = np.squeeze(x)

    # Back to time-domain
    xout = np.fft.ifft(x, n, axis=0)

    if axis == 1:
        xout = xout.T

    # Check before output
    if not np.allclose(np.fft.fft(xout, n, axis=0), x):
        raise ValueError('Issue in return to time-domain')

    return xout


def fill_buffers_whole_chunk(k, neighbourNodes, zBuffer, zLocalK):
    """Fills neighbors nodes' buffers, using frequency domain data.
    Data comes from compression via function `danse_compression_freq_domain`.
    
        Parameters
    ----------
    k : int
        Current node index.
    neighbourNodes : [numNodes x 1] list of [nNeighbours[n] x 1] lists (int)
        Network indices of neighbours, per node.
    zBuffer : [numNodes x 1] list of [nNeighbours[n] x 1] lists of [variable length] np.ndarrays (complex)
        Compressed signals buffers for each node and its neighbours.
    zLocal : [N x 1] np.ndarray (float)
        Latest compressed local signals to be broadcasted from node `k`.

    Returns
    -------
    zBuffer : [numNodes x 1] list of [nNeighbours[n] x 1] lists of [N x 1] np.ndarrays (complex)
        Updated compressed signals buffers for each node and its neighbours.
    """

    # Loop over neighbors of `k`
    for idxq in range(len(neighbourNodes[k])):
        q = neighbourNodes[k][idxq]                 # network-wide index of node `q` (one of node `k`'s neighbors)
        idxKforNeighborQ = [i for i, x in enumerate(neighbourNodes[q]) if x == k]
        idxKforNeighborQ = idxKforNeighborQ[0]      # node `k`'s "neighbor index", from node `q`'s perspective
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
        out[ii] = np.dot(yqzp[idDesired[ii] + 1:idDesired[ii] + 1 + len(a)], np.flip(a))
    
    return out


def fill_buffers_td_few_samples(k, neighbourNodes, zBuffer, zLocalK, L):
    """Fill in buffers -- simulating broadcast of compressed signals
    from one node (`k`) to its neighbours.
    
    Parameters
    ----------
    k : int
        Current node index.
    neighbourNodes : [numNodes x 1] list of [nNeighbours[n] x 1] lists (int)
        Network indices of neighbours, per node.
    zBuffer : [numNodes x 1] list of [nNeighbours[n] x 1] lists of [variable length] np.ndarrays (float)
        Compressed signals buffers for each node and its neighbours.
    zLocalK : [N x 1] np.ndarray (float)
        Latest compressed local signals to be broadcasted from node `k`.
    L : int
        Broadcast chunk length.

    Returns
    -------
    zBuffer : [numNodes x 1] list of [nNeighbours[n] x 1] lists of [variable length] np.ndarrays (float)
        Updated compressed signals buffers for each node and its neighbours.
    """

    # Check for sample-per-sample broadcast scheme (`L==1`)
    if isinstance(zLocalK, float):
        zLocalK = [zLocalK]
        if L != 1:
            raise ValueError(f'Incoherence: float `zLocalK` but L = {L}')

    # Loop over neighbors of node `k`
    for idxq in range(len(neighbourNodes[k])):
        q = neighbourNodes[k][idxq]                 # network-wide index of node `q` (one of node `k`'s neighbors)
        idxKforNeighborQ = [i for i, x in enumerate(neighbourNodes[q]) if x == k]
        idxKforNeighborQ = idxKforNeighborQ[0]      # node `k`'s "neighbor index", from node `q`'s perspective
        # Only broadcast the `L` last samples of local compressed signals
        zBuffer[q][idxKforNeighborQ] = np.concatenate((zBuffer[q][idxKforNeighborQ], zLocalK[-L:]), axis=0)
        
    return zBuffer


def process_incoming_signals_buffers(k, dv: DANSEvariables, p: DANSEparameters, t):
    """When called, processes the incoming data from other nodes, as stored in local node's buffers.
    Called whenever a DANSE update can be performed (`N` new local samples were captured since last update).
    
    Parameters
    ----------
    k : int
        Receiving node index.
    dv : DANSEvariables object
        DANSE variables to be updated.
    p : DANSEparameters object
        DANSE parameters.
    t : float
        Current time instant [s].
    
    Returns
    -------
    dv : DANSEvariables object
        Updated DANSE variables.
    """

    # Useful renaming
    Ndft = p.DFTsize
    Ns = p.Ns
    Lbc = p.broadcastLength

    # Initialize compressed signal matrix ($\mathbf{z}_{-k}$ in [1]'s notation)
    if p.broadcastType == 'wholeChunk_fd':
        zk = np.empty((dv.nPosFreqs, 0), dtype=complex)
    elif p.broadcastType == 'wholeChunk_td':
        zk = np.empty((p.DFTsize, 0), dtype=float)
    elif p.broadcastType == 'fewSamples_td':
        zk = np.empty((p.DFTsize, 0), dtype=float)

    # Initialise flags vector (overflow: >0; underflow: <0; or none: ==0)
    bufferFlags = np.zeros(len(dv.neighbors[k]))

    for idxq in range(len(dv.neighbors[k])):
        
        Bq = len(dv.zBuffer[k][idxq])  # buffer size for neighbour node `q`

        # Frequency-domain chunks broadcasting (naïve DANSE, simplest to visualize/implement, but inefficient)
        if p.broadcastType == 'wholeChunk_fd':
            if Bq == 0:     # under-flow with chunk-wise FD broadcasting
                if dv.DANSEiter[k] == 0:
                    raise ValueError(f'Unexpected buffer under-flow at first iteration in FD-broadcasting.')
                else:
                    print('FD BROADCASTING: BUFFER UNDERFLOW')
                    zCurrBuffer = dv.z[k][:, idxq]  # re-use previous z...
            elif Bq == 2 * dv.nPosFreqs:  # over-flow with chunk-wise FD broadcasting
                print('FD BROADCASTING: BUFFER OVERFLOW')
                zCurrBuffer = dv.zBuffer[k][idxq][:Ns]
            elif Bq == dv.nPosFreqs:     # correctly filled-in buffer, no under-/over-flow
                zCurrBuffer = dv.zBuffer[k][idxq]
            else:
                raise ValueError(f'Unexpected buffer over-/under-flow in FD-broadcasting: {Bq} samples instead of {dv.nPosFreqs} expected.')
        
        # Time-domain chunks broadcasting (naïve DANSE, simplest to visualize/implement, but inefficient)
        elif p.broadcastType == 'wholeChunk_td':
            if dv.DANSEiter[k] == 0:
                if Bq == Ns:
                    # Not yet any previous buffer -- need to appstart zeros
                    zCurrBuffer = np.concatenate((np.zeros(Ndft - Bq), dv.zBuffer[k][idxq]))
                elif Bq == 0:
                    # Node `q` has not yet transmitted enough data to node `k`, but node `k` has already reached its first update instant.
                    # Interpretation: Node `q` samples slower than node `k`. 
                    # Response: ...
                    print(f'[b- @ t={np.round(t, 3)}s] Buffer underflow at current node`s B_{dv.neighbors[k][idxq]+1} buffer | -1 broadcast')
                    bufferFlags[idxq] = -1      # raise negative flag
                    zCurrBuffer = np.zeros(Ndft)
            else:
                if Bq == Ns:
                    # All good, no under-/over-flows
                    if not np.any(dv.z[k]):
                        # Not yet any previous buffer -- need to appstart zeros
                        zCurrBuffer = np.concatenate((np.zeros(Ndft - Bq), dv.zBuffer[k][idxq]))
                    else:
                        # Concatenate last `Ns` samples of previous buffer with current buffer
                        zCurrBuffer = np.concatenate((dv.z[k][-Ns:, idxq], dv.zBuffer[k][idxq]))
                else:
                    # Under- or over-flow...
                    raise ValueError('[NOT YET IMPLEMENTED]')
                
        elif p.broadcastType == 'fewSamples_td':

            if dv.DANSEiter[k] == 0: # first DANSE iteration case -- we are expecting an abnormally full buffer, with an entire DANSE chunk size inside of it
                if Bq == Ndft: 
                    # There is no significant SRO between node `k` and `q`.
                    # Response: node `k` uses all samples in the `q`^th buffer.
                    zCurrBuffer = dv.zBuffer[k][idxq]
                elif (Ndft - Bq) % Lbc == 0 and Bq < Ndft:
                    # Node `q` has not yet transmitted enough data to node `k`, but node `k` has already reached its first update instant.
                    # Interpretation: Node `q` samples slower than node `k`. 
                    # Response: ...
                    nMissingBroadcasts = int(np.abs((Ndft - Bq) / Lbc))
                    print(f'[b- @ t={np.round(t, 3)}s] Buffer underflow at current node`s B_{dv.neighbors[k][idxq]+1} buffer | -{nMissingBroadcasts} broadcast(s)')
                    bufferFlags[idxq] = -1 * nMissingBroadcasts      # raise negative flag
                    zCurrBuffer = np.concatenate((np.zeros(Ndft - Bq), dv.zBuffer[k][idxq]), axis=0)
                    # raise ValueError('Unexpected edge case: Node `q` has not yet transmitted enough data to node `k`, but node `k` has already reached its first update instant.')
                elif (Ndft - Bq) % Lbc == 0 and Bq > Ndft:
                    # Node `q` has already transmitted too much data to node `k`.
                    # Interpretation: Node `q` samples faster than node `k`.
                    # Response: node `k` raises a positive flag and uses the last `frameSize` samples from the `q`^th buffer.
                    nExtraBroadcasts = int(np.abs((Ndft - Bq) / Lbc))
                    print(f'[b+ @ t={np.round(t, 3)}s] Buffer overflow at current node`s B_{dv.neighbors[k][idxq]+1} buffer | +{nExtraBroadcasts} broadcasts(s)')
                    bufferFlags[idxq] = +1 * nExtraBroadcasts       # raise positive flag
                    zCurrBuffer = dv.zBuffer[k][idxq][-Ndft:]

            else:   # not the first DANSE iteration -- we are expecting a normally full buffer, with a DANSE chunk size considering overlap
                if Bq == Ns:             # case 1: no broadcast frame mismatch between node `k` and node `q`
                    pass
                elif (Ns - Bq) % Lbc == 0 and Bq < Ns:       # case 2: negative broadcast frame mismatch between node `k` and node `q`
                    nMissingBroadcasts = int(np.abs((Ns - Bq) / Lbc))
                    print(f'[b- @ t={np.round(t, 3)}s] Buffer underflow at current node`s B_{dv.neighbors[k][idxq]+1} buffer | -{nMissingBroadcasts} broadcast(s)')
                    bufferFlags[idxq] = -1 * nMissingBroadcasts      # raise negative flag
                elif (Ns - Bq) % Lbc == 0 and Bq > Ns:       # case 3: positive broadcast frame mismatch between node `k` and node `q`
                    nExtraBroadcasts = int(np.abs((Ns - Bq) / Lbc))
                    print(f'[b+ @ t={np.round(t, 3)}s] Buffer overflow at current node`s B_{dv.neighbors[k][idxq]+1} buffer | +{nExtraBroadcasts} broadcasts(s)')
                    bufferFlags[idxq] = +1 * nExtraBroadcasts       # raise positive flag
                else:
                    if (Ns - Bq) % Lbc != 0 and np.abs(dv.DANSEiter[k] - (dv.numIterations - 1)) < 10:
                        print('[b! @ t={np.round(t, 3)}s] This is the last iteration -- not enough samples anymore due to cumulated SROs effect, skip update.')
                        bufferFlags[idxq] = np.NaN   # raise "end of signal" flag
                    else:
                        raise ValueError(f'Unexpected buffer size ({Bq} samples, with L={Lbc} and N={Ns}) for neighbor node q={dv.neighbors[k][idxq]+1}.')
                # Build current buffer
                if Ndft - Bq > 0:
                    zCurrBuffer = np.concatenate((dv.z[k][-(Ndft - Bq):, idxq], dv.zBuffer[k][idxq]), axis=0)
                else:   # edge case: no overlap between consecutive frames
                    zCurrBuffer = dv.zBuffer[k][idxq]

        # Stack compressed signals
        zk = np.concatenate((zk, zCurrBuffer[:, np.newaxis]), axis=1)

    # Update DANSE variables
    dv.z[k] = zk
    dv.bufferFlags[k][dv.DANSEiter[k], :] = bufferFlags

    return dv


def events_parser(events: DANSEeventInstant, startUpdates, printouts=False, doNotPrintBCs=False):
    """Printouts to inform user of DANSE events.
    
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
        If True, do not print the broadcast events (only used if `printouts == True`).

    Returns
    -------
    t : float
        Current time instant [s].
    eventTypes : list of str
        Events at current instant.
    nodesConcerned : list of ints
        Corresponding node indices.
    """

    if printouts:
        if 'up' in events.type:
            txt = f't={np.round(events.t, 3)}s -- '
            updatesTxt = 'Updating nodes: '
            if doNotPrintBCs:
                broadcastsTxt = ''
            else:
                broadcastsTxt = 'Broadcasting nodes: '
            flagCommaUpdating = False    # little flag to add a comma (`,`) at the right spot
            for idxEvent in range(len(events.type)):
                k = int(events.nodes[idxEvent])   # node index
                if events.type[idxEvent] == 'bc' and not doNotPrintBCs:
                    if idxEvent > 0:
                        broadcastsTxt += ','
                    broadcastsTxt += f'{k + 1}'
                elif events.type[idxEvent] == 'up':
                    if startUpdates[k]:  # only print if the node actually has started updating (i.e. there has been sufficiently many autocorrelation matrices updates since the start of recording)
                        if not flagCommaUpdating:
                            flagCommaUpdating = True
                        else:
                            updatesTxt += ','
                        updatesTxt += f'{k + 1}'
            print(txt + broadcastsTxt + '; ' + updatesTxt)

    return None


def build_ytilde(yk, tCurr, fs, k, dv: DANSEvariables, params: DANSEparameters):
    """
    
    Parameters
    ----------
    yk : [Nt x Nsensors] np.ndarray (float)
        Full time-domain local sensor signals at node `k`.
    tCurr : float
        Current time instant [s].
    fs : float
        Node `k`'s sampling frequency [s].
    k : int
        Receiving node index.
    dv : DANSEvariables object
        DANSE variables to be updated.
    p : DANSEparameters object
        DANSE parameters.

    Returns
    -------
    TODO:
    """

    # Extract current local data chunk
    yLocalCurr, dv.idxBegChunk, dv.idxEndChunk = local_chunk_for_update(yk, tCurr, fs, params)

    # Compute VAD
    VADinFrame = dv.fullVAD[np.amax([dv.idxBegChunk, 0]):dv.idxEndChunk, k]
    # If there is a majority of "VAD = 1" in the frame, set the frame-wise VAD to 1
    dv.oVADframes[dv.DANSEiter[k]] = sum(VADinFrame == 0) <= len(VADinFrame) // 2
    # Count number of spatial covariance matrices updates
    if dv.oVADframes[dv.DANSEiter[k]]:
        dv.numUpdatesRyy[k] += 1
    else:
        dv.numUpdatesRnn[k] += 1

    # Build `\tilde{y}_k`
    if params.broadcastType == 'wholeChunk_fd':
        # Broadcasting done in frequency-domain
        yLocalHatCurr = 1 / params.normFactWOLA * np.fft.fft(
            yLocalCurr * params.winWOLAanalysis[:, np.newaxis],
            params.DFTsize, axis=0
        )
        dv.yTildeHat[k][:, dv.DANSEiter[k], :] = np.concatenate(
            (yLocalHatCurr[:dv.nPosFreqs, :], 1 / params.normFactWOLA * dv.z[k]),
            axis=1
        )
    elif params.broadcastType in ['wholeChunk_td', 'fewSamples_td']:
        # Build full available observation vector
        yTildeCurr = np.concatenate((yLocalCurr, dv.z[k]), axis=1)
        dv.yTilde[k][:, dv.DANSEiter[k], :] = yTildeCurr
        # Go to frequency domain
        yTildeHatCurr = 1 / params.normFactWOLA * np.fft.fft(
            dv.yTilde[k][:, dv.DANSEiter[k], :] * params.winWOLAanalysis[:, np.newaxis],
            params.DFTsize, axis=0
        )
        # Keep only positive frequencies
        dv.yTildeHat[k][:, dv.DANSEiter[k], :] = yTildeHatCurr[:dv.nPosFreqs, :]

    return dv, yLocalCurr


def account_for_flags(yLocalCurr, k, dv: DANSEvariables, p: DANSEparameters, t):

    # Init
    skipUpdate = False
    extraPhaseShiftFactor = np.zeros(dv.dimYTilde[k])
    
    for q in range(len(dv.neighbors[k])):
        if not np.isnan(dv.bufferFlags[k][dv.DANSEiter[k], q]):
            extraPhaseShiftFactor[yLocalCurr.shape[-1] + q] =\
                dv.bufferFlags[k][dv.DANSEiter[k], q] * p.broadcastLength
            # ↑↑↑ if `bufferFlags[k][i[k], q] == 0`, `extraPhaseShiftFactor = 0` and no additional phase shift is applied
            if dv.bufferFlags[k][dv.DANSEiter[k], q] != 0:
                dv.flagIterations[k].append(dv.DANSEiter[k])  # keep flagging iterations in memory
                dv.flagInstants[k].append(t)       # keep flagging instants in memory
        else:
            # From `process_incoming_signals_buffers`: "Not enough samples anymore due to cumulated SROs effect, skip update"
            skipUpdate = True
    # Save uncompensated \tilde{y} for coherence-drift-based SRO estimation
    dv.yTildeHatUncomp[k][:, dv.DANSEiter[k], :] = copy.copy(dv.yTildeHat[k][:, dv.DANSEiter[k], :])
    dv.yyHuncomp[k][dv.DANSEiter[k], :, :, :] = np.einsum(
        'ij,ik->ijk',
        dv.yTildeHatUncomp[k][:, dv.DANSEiter[k], :],
        dv.yTildeHatUncomp[k][:, dv.DANSEiter[k], :].conj()
    )
    # Compensate SROs
    if p.compensateSROs:
        # Complete phase shift factors
        dv.phaseShiftFactors[k] += extraPhaseShiftFactor
        if k == 0:  # Save for plotting
            dv.phaseShiftFactorThroughTime[dv.DANSEiter[k]:] = dv.phaseShiftFactors[k][yLocalCurr.shape[-1] + q]
        # Apply phase shift factors
        dv.yTildeHat[k][:, dv.DANSEiter[k], :] *=\
            np.exp(-1 * 1j * 2 * np.pi / p.DFTsize *\
                np.outer(np.arange(dv.nPosFreqs), dv.phaseShiftFactors[k]))

    return dv, skipUpdate


def spatial_covariance_matrix_update(y, Ryy, Rnn, beta, vad):
    """Helper function: performs the spatial covariance matrices updates.
    
    Parameters
    ----------
    y : [N x M] np.ndarray (real or complex)
        Current input data chunk (if complex: in the frequency domain).
    Ryy : [N x M x M] np.ndarray (real or complex)
        Previous Ryy matrices (for each time frame /or/ each frequency line).
    Rnn : [N x M x M] np.ndarray (real or complex)
        Previous Rnn matrices (for each time frame /or/ each frequency line).
    beta : float (0 <= beta <= 1)
        Exponential averaging forgetting factor.
    vad : bool
        If True (=1), Ryy is updated. Otherwise, Rnn is updated.
    
    Returns
    -------
    Ryy : [N x M x M] np.ndarray (real or complex)
        New Ryy matrices (for each time frame /or/ each frequency line).
    Rnn : [N x M x M] np.ndarray (real or complex)
        New Rnn matrices (for each time frame /or/ each frequency line).
    yyH : [N x M x M] np.ndarray (real or complex)
        Instantaneous correlation outer product.
    """
    yyH = np.einsum('ij,ik->ijk', y, y.conj())

    if vad:
        Ryy = beta * Ryy + (1 - beta) * yyH  # update signal + noise matrix
    else:     
        Rnn = beta * Rnn + (1 - beta) * yyH  # update noise-only matrix

    return Ryy, Rnn, yyH


def get_desired_signal(k, dv: DANSEvariables, p: DANSEparameters):
    """
    Compute chunk of desired signal from DANSE freq.-domain filters
    and freq.-domain observation vector y_tilde.

    Parameters
    ----------
    w : [Nf x M] np.ndarray (complex)
        Frequency-domain filter coefficients for each of the `M` channels (>0 freqs. only).
    y : [Nf x M] np.ndarray (complex)
        Frequency-domain signals (full observations vector $\\tilde{\\mathbf{y}}_k$) (>0 freqs. only).
    win : [N x 1] np.ndarray (float)
        Synthesis window (time-domain).
    dChunk : [N x 1] np.ndarray (float)
        For WOLA processing: previous data chunk to use for overlap-add.
    normFactWOLA : float
        For WOLA processing: normalization factor (sum of window samples).
    winShift : int
        Window shift [samples].
    desSigEstChunkLength : int
        Output length (only used if `processingType == 'conv'`) [samples].
    processingType : str
        Processing type -- "wola": WOLA synthesis; "conv": linear convolution via T(z)-approximation.

    Returns
    -------
    dhatCurr : [Nf x 1] np.ndarray (complex)
        Current chunk frequency-domain estimate (>0 freqs. only).
    dChunk : [N x 1] np.ndarray (float)
        Current chunk time-domain estimate, incl. overlap-add.
    """
    
    # Useful renaming (compact code)
    w = dv.wTilde[k][:, dv.DANSEiter[k] + 1, :]
    y = dv.yTildeHat[k][:, dv.DANSEiter[k], :]
    win = p.winWOLAsynthesis
    dChunk = dv.d[dv.idxBegChunk:dv.idxEndChunk, k]
    yTD = dv.yTilde[k][:, dv.DANSEiter[k], :dv.nLocalSensors[k]]
        
    dhatCurr = None  # init

    if p.desSigProcessingType == 'wola':
        # ----- Compute desired signal chunk estimate using WOLA -----
        dhatCurr = np.einsum('ij,ij->i', w.conj(), y)   # vectorized way to do inner product on slices
        # Transform back to time domain (WOLA processing)
        dChunCurr = p.normFactWOLA * win * back_to_time_domain(dhatCurr, len(win))
        if len(dChunk) < len(win):   # fewSamples_td BC scheme: first iteration required zero-padding at the start of the chunk
            dChunk += np.real_if_close(dChunCurr[-len(dChunk):])   # overlap and add construction of output time-domain signal
        else:
            dChunk += np.real_if_close(dChunCurr)   # overlap and add construction of output time-domain signal

    elif p.desSigProcessingType == 'conv':
        # ----- Compute desired signal chunk estimate using T(z) approx. for linear convolution -----
        wIR = dist_fct_approx(w, win, win, p.Ns)
        # Perform convolution
        yfiltLastSamples = np.zeros((p.Ns, yTD.shape[-1]))
        for m in range(yTD.shape[-1]):
            idDesired = np.arange(start=len(wIR) - p.Ns, stop=len(wIR))   # indices required from convolution output
            tmp = extract_few_samples_from_convolution(idDesired, wIR[:, m], yTD[:, m])
            yfiltLastSamples[:, m] = tmp

        dChunk = np.sum(yfiltLastSamples, axis=1)

    return dhatCurr, dChunk