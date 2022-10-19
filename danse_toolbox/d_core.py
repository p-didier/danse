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
    dv, skipUpdate = account_for_flags(yLocalCurr, k, dv, params, tCurr)
    # Ryy and Rnn updates
    dv.Ryytilde[k], dv.Rnntilde[k], dv.yyH[k][dv.DANSEiter[k], :, :, :] =\
        spatial_covariance_matrix_update(
            dv.yTildeHat[k][:, dv.DANSEiter[k], :],
            dv.Ryytilde[k],
            dv.Rnntilde[k],
            dv.expAvgBeta[k],
            dv.oVADframes[dv.DANSEiter[k]]
        )
    
    # Check quality of autocorrelations estimates 
    # -- once we start updating, do not check anymore.
    if not dv.startUpdates[k] and \
        dv.numUpdatesRyy[k] > np.amax(dv.dimYTilde) and \
            dv.numUpdatesRnn[k] > np.amax(dv.dimYTilde):
        dv.startUpdates[k] = True

    if dv.startUpdates[k] and not params.bypassFilterUpdates and not skipUpdate:
        if k == params.referenceSensor and dv.nInternalFilterUpdates[k] == 0:
            # Save first update instant (for, e.g., SRO plot)
            firstDANSEupdateRefSensor = tCurr
        # No `for`-loop versions 
        if params.performGEVD:    # GEVD update
            dv.wTilde[k][:, dv.DANSEiter[k] + 1, :], _ = perform_gevd_noforloop(
                dv.Ryytilde[k],
                dv.Rnntilde[k],
                params.GEVDrank,
                params.referenceSensor
            )
        else:                       # regular update (no GEVD)
            dv.wTilde[k][:, dv.DANSEiter[k] + 1, :] = perform_update_noforloop(
                dv.Ryytilde[k],
                dv.Rnntilde[k],
                params.referenceSensor
            )
        # Count the number of internal filter updates
        dv.nInternalFilterUpdates[k] += 1  

        # TODO: improve this ugly bit vvvv
        # # Useful export for enhancement metrics computations
        # if dv.nInternalFilterUpdates[k] >= s.minFiltUpdatesForMetricsComputation and\
        #     tStartForMetrics[k] is None:
        #     if s.asynchronicity.compensateSROs and s.asynchronicity.estimateSROs == 'CohDrift':
        #         # Make sure SRO compensation has started
        #         if nInternalFilterUpdates[k] > s.asynchronicity.cohDriftMethod.startAfterNupdates:
        #             tStartForMetrics[k] = t
        #     else:
        #         tStartForMetrics[k] = t
    else:
        # Do not update the filter coefficients
        dv.wTilde[k][:, dv.DANSEiter[k] + 1, :] = dv.wTilde[k][:, dv.DANSEiter[k], :]
        if skipUpdate:
            print(f'Node {k+1}: {dv.DANSEiter[k] + 1}^th update skipped.')
    if params.bypassFilterUpdates:
        print('!! User-forced bypass of filter coefficients updates !!')

    # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  Update external filters (for broadcasting)  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    dv.wTildeExt[k] = dv.expAvgBeta[k] * dv.wTildeExt[k] +\
        (1 - dv.expAvgBeta[k]) *  dv.wTildeExtTarget[k]
    # Update targets
    # TODO:TODO:TODO:TODO: stopped here on 19.10.2022
    # if tCurr - lastExternalFiltUpdateInstant[k] >= s.timeBtwExternalFiltUpdates:
    #     wTildeExternalTarget[k] = (1 - alphaExternalFilters) * wTildeExternalTarget[k] + alphaExternalFilters * wTilde[k][:, i[k] + 1, :yLocalCurr.shape[-1]]
    #     # Update last external filter update instant [s]
    #     lastExternalFiltUpdateInstant[k] = tCurr
    #     if s.printouts.externalFilterUpdates:    # inform user
    #         print(f't={np.round(t, 3)}s -- UPDATING EXTERNAL FILTERS for node {k+1} (scheduled every [at least] {s.timeBtwExternalFiltUpdates}s)')
    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑  Update external filters (for broadcasting)  ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


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


def perform_gevd_noforloop(Ryy, Rnn, rank=1, refSensorIdx=0):
    """GEVD computations for DANSE, `for`-loop free.
    
    Parameters
    ----------
    Ryy : [M x N x N] np.ndarray (complex)
        Autocorrelation matrix between the sensor signals.
    Rnn : [M x N x N] np.ndarray (complex)
        Autocorrelation matrix between the noise signals.
    rank : int
        GEVD rank approximation.
    refSensorIdx : int
        Index of the reference sensor (>=0).

    Returns
    -------
    w : [M x N] np.ndarray (complex)
        GEVD-DANSE filter coefficients.
    Qmat : [M x N x N] np.ndarray (complex)
        Hermitian conjugate inverse of the generalized eigenvectors matrix of the pencil {Ryy, Rnn}.
    Xmat : [M x N x N] np.ndarray (complex)
        Qeneralized eigenvectors matrix of the pencil {Ryy, Rnn}.
    """
    # ------------ for-loop-free estimate ------------
    n = Ryy.shape[-1]
    nFreqs = Ryy.shape[0]
    # Reference sensor selection vector 
    Evect = np.zeros((n,))
    Evect[refSensorIdx] = 1

    sigma = np.zeros((nFreqs, n))
    Xmat = np.zeros((nFreqs, n, n), dtype=complex)

    # t0 = time.perf_counter()
    for kappa in range(nFreqs):
        # Perform generalized eigenvalue decomposition -- as of 2022/02/17: scipy.linalg.eigh() seemingly cannot be jitted nor vectorized
        sigmacurr, Xmatcurr = sla.eigh(Ryy[kappa, :, :], Rnn[kappa, :, :], check_finite=False, driver='gvd')
        # Flip Xmat to sort eigenvalues in descending order
        idx = np.flip(np.argsort(sigmacurr))
        sigma[kappa, :] = sigmacurr[idx]
        Xmat[kappa, :, :] = Xmatcurr[:, idx]

    Qmat = np.linalg.inv(np.transpose(Xmat.conj(), axes=[0,2,1]))
    # GEVLs tensor
    Dmat = np.zeros((nFreqs, n, n))
    for ii in range(rank):
        Dmat[:, ii, ii] = np.squeeze(1 - 1/sigma[:, ii])
    # LMMSE weights
    Qhermitian = np.transpose(Qmat.conj(), axes=[0,2,1])
    w = np.matmul(np.matmul(np.matmul(Xmat, Dmat), Qhermitian), Evect)

    return w, Qmat


def perform_update_noforloop(Ryy, Rnn, refSensorIdx=0):
    """Regular DANSE update computations, `for`-loop free.
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