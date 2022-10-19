# Core functions for DANSE.
#
# ~created on 19.10.2022 by Paul Didier

# General TODO:'s
# -- Allow computation of local and centralised estimates

from numba import njit
import scipy.linalg as sla
from danse.danse_toolbox.d_base import *
from danse.danse_toolbox.d_classes import *

def danse(wasn: list[Node], p: DANSEparameters):
    """
    Main DANSE function.

    Parameters
    ----------
    wasn : list of `Node` objects
        WASN under consideration.
    p : DANSEparameters object
        Parameters.

    Returns
    -------
    out : DANSEoutputs object
        DANSE outputs.
    """

    # Initialize variables
    dv = DANSEvariables().fromWASN(wasn)

    # Events
    eventInstants, fs = initialize_events(
        timeInstants=dv.timeInstants,
        N=p.DFTsize,
        Ns=p.Ns,
        L=p.broadcastLength,
        bd=p.broadcastType,
        efficient=p.efficientSpSBC
    )


    for idx_t in range(len(eventInstants)):

        # Parse event matrix and inform user (is asked)
        events_parser(
            eventInstants[idx_t],
            dv.startUpdates,
            printouts=p.printout_eventsParser,
            doNotPrintBCs=p.printout_eventsParserNoBC
        )

        events = eventInstants[idx_t]  # events at current instant

        # Loop over events
        for idx_e in range(events.nEvents):

            k = events.nodes[idx_e]  # node index

            # Broadcast event
            if events.type[idx_e] == 'bc':

                dv = broadcast(
                    yk=wasn[k].data,
                    tCurr=events.t,
                    fs=fs[k],  # TODO: maybe not needed to recompute `fs` (already there in `wasn[k].fs`)
                    k=k,
                    dv=dv,
                    params=p
                )
            
            # Filter updates and desired signal estimates event
            elif events.type[idx_e] == 'up':

                dv = update_and_estimate(
                    yk=wasn[k].data,
                    tCurr=events.t,
                    fs=fs[k],
                    k=k,
                    dv=dv,
                    params=p
                )

            else:
                raise ValueError(f'Unknown event type: "{events.type[idx_e]}".')

    out = DANSEoutputs(
        #
    )

    return out


def broadcast(yk, tCurr, fs, k,
                dv: DANSEvariables,
                params: DANSEparameters):
    """
    Parameters
    ----------
    yk : [Nt x Nsensors] np.ndarray (float)
        Local sensors time-domain signals.
    tCurr : float
        Broadcast event global time instant [s].
    fs : float
        Node's sampling frequency [Hz].
    k : int
        Node index.
    dv : DANSEvariables object.
        DANSE variables to be updated during broadcast.
    params : DANSEparameters
        DANSE parameters.

    Returns
    -------
    dv : DANSEvariables object.
        Updated DANSE variables.
    """

    # Extract correct frame of local signals
    ykFrame = local_chunk_for_broadcast(yk, tCurr, fs, params.DFTsize)

    if len(ykFrame) < params.DFTsize:

        print('Cannot perform compression: not enough local signals samples.')

    else:

        if params.broadcastType == 'wholeChunk_fd':

            # Frequency-domain chunk-wise broadcasting
            zLocalHat, _ = danse_compression_whole_chunk(
                ykFrame,
                dv.wTildeExt[k],
                params.winWOLAanalysis,
                params.winWOLAsynthesis
            )  # local compressed signals (freq.-domain)

            # Fill buffers in
            dv.zBuffer = fill_buffers_whole_chunk(
                k,
                dv.neighbors,
                dv.zBuffer,
                zLocalHat
            ) 
            dv.zLocal[k] = None   # FD BC -- no `zLocal` variable computed
            
        elif params.broadcastType == 'wholeChunk_td':

            # Time-domain chunk-wise broadcasting
            _, dv.zLocal[k] = danse_compression_whole_chunk(
                ykFrame,
                dv.wTildeExt[k],
                params.winWOLAanalysis,
                params.winWOLAsynthesis,
                zqPrevious=dv.zLocal[k]
            )  # local compressed signals (time-domain)

            # Fill buffers in
            dv.zBuffer = fill_buffers_whole_chunk(
                k,
                dv.neighbors,
                dv.zBuffer,
                dv.zLocal[k][:(params.DFTsize // 2)]
            ) 
        
        elif params.broadcastType == 'fewSamples_td':
            # Time-domain broadcasting, `L` samples at a time,
            # via linear-convolution approximation of WOLA filtering process

            # Only update filter every so often
            updateBroadcastFilter = False
            if np.abs(tCurr - dv.previousTDfilterUpdate[k]) >= params.updateTDfilterEvery:
                updateBroadcastFilter = True
                dv.previousTDfilterUpdate[k] = tCurr

            dv.zLocal[k], dv.wIR[k] = danse_compression_few_samples(
                ykFrame,
                dv.wTildeExt[k],
                params.DFTsize,
                params.broadcastLength,
                dv.wIR[k],
                params.winWOLAanalysis,
                params.winWOLAsynthesis,
                params.Ns,
                updateBroadcastFilter
            )  # local compressed signals

            dv.zBuffer = fill_buffers_td_few_samples(
                k,
                dv.neighbors,
                dv.zBuffer,
                dv.zLocal[k],
                params.broadcastLength
            )

    return dv


def update_and_estimate(yk, tCurr, fs, k,
                dv: DANSEvariables,
                params: DANSEparameters):
    """
    Update filter coefficient at current node
    and estimate corresponding desired signal frame.
    """

    # Process buffers
    dv.z[k], dv.bufferFlags[k][dv.DANSEiter[k], :] = process_incoming_signals_buffers(dv, params, tCurr)
    # Wipe local buffers
    dv.zBuffer[k] = [np.array([]) for _ in range(len(dv.neighbors[k]))]
    # Construct `\tilde{y}_k` in frequency domain
    dv, yLocalCurr = build_ytilde(yk, tCurr, fs, k, dv, params)
    # Account for buffer flags
    account_for_flags(yLocalCurr, k, dv, params, tCurr)



    stop = 1

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
        # Apply linear combination to form compressed signal
        zqHat = wHat.conj() * yqHat     # single sensor = simple element-wise multiplication
    else:
        yqHat = np.fft.fft(np.squeeze(yq) * h[:, np.newaxis], n, axis=0)
        # Keep only positive frequencies
        yqHat = yqHat[:int(n/2 + 1), :]
        # Apply linear combination to form compressed signal
        zqHat = np.einsum('ij,ij->i', wHat.conj(), yqHat)  # vectorized way to do inner product on slices of a 3-D tensor https://stackoverflow.com/a/15622926/16870850

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
            zq[:(n // 2)] = zqPrevious[-(n // 2):]   # TODO: consider case with multiple neighbor nodes (`len(zqPrevious) > 1`)
            zq += zqCurr
    else:
        zq = None
    
    return zqHat, zq


def danse_compression_few_samples(
    yq, wqqHat, L, wIRprevious,
    winWOLAanalysis, winWOLAsynthesis, R, 
    updateBroadcastFilter=False):
    """Performs local signals compression according to DANSE theory [1],
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
        idDesired = np.arange(
            start=len(wIR) - L + 1,
            stop=len(wIR) + 1
        )   # indices required from convolution output
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
    return np.trace(A, ofst)


def dist_fct_approx(wHat, h, f, R, jitted=True):
    """
    Distortion function approximation of the WOLA filtering process.
    -- See Word journal 2022, weeks 30-33.

    Parameters
    ----------
    wHat : [Nf x M] np.ndarry (complex)
        Frequency-domain filter coefficients for each of the `M` channels (>0 freqs. only)
        used in the WOLA process to modify the short-term spectrum.
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
        Time-domain distortion function approximation of the WOLA filtering process.
    """

    n = len(h)

    wTD = back_to_time_domain(wHat.conj(), n, axis=0)
    wTD = np.real_if_close(wTD)         
    wIR_out = np.zeros((2 * n - 1, wTD.shape[1]))
    for m in range(wTD.shape[1]):
        Hmat = sla.circulant(np.flip(wTD[:, m]))
        # Amat = dist_fct_module.get_Amat_jitted(f, Hmat, h)
        Amat = np.diag(f) @ Hmat @ np.diag(h)

        for ii in np.arange(start=-n+1, stop=n):
            if jitted:
                wIR_out[ii + n - 1, m] = get_trace_jitted(Amat, ii)
                # wIR_out[ii + n - 1, m] = dist_fct_module.get_trace_jitted(Amat, ii)
            else:
                wIR_out[ii + n - 1, m] = np.sum(np.diagonal(Amat, ii))

    wIR_out /= R

    return wIR_out