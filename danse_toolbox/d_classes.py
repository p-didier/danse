import copy
import numpy as np
import scipy.linalg as sla
from dataclasses import dataclass, field
from danse.siggen.classes import Node
import danse.danse_toolbox.d_base as base
import danse.danse_toolbox.d_sros as sros


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
    referenceSensor: int = 0    # index of reference sensor at each node
    DFTsize: int = 1024    # DFT size
    WOLAovlp: float = .5   # WOLA window overlap [*100%]
    broadcastLength: int = 1    # number of samples to be broadcasted at a time
    broadcastType: str = 'wholeChunk_td'    # type of broadcast
        # -- 'wholeChunk_td': chunks of compressed signals in time-domain,
        # -- 'fewSamples_td': T(z)-approximation of WOLA compression process.
        # broadcast L ≪ Ns samples at a time.
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
        # -- TODO: make that used only if the node-updating is simultaneous/asynchronous
    alphaExternalFilters: float = .5    # exponential averaging constant
                                        # for external filter target update.
    t_expAvg50p: float = 2.     # [s] Time in the past at which the value is
                                # weighted by 50% via exponential averaging.
    # Desired signal estimation
    desSigProcessingType: str = 'wola'  # processing scheme used to compute
        # the desired signal estimates: "wola": WOLA synthesis,
                                    # "conv": T(z)-approximation.
    # Metrics
    minFiltUpdatesForMetrics: int = 10   # minimum number of DANSE
        # updates before start of speech enhancement metrics computation
    

    def __post_init__(self):
        self.Ns = int(self.DFTsize * (1 - self.WOLAovlp))


@dataclass
class DANSEvariables(DANSEparameters):
    
    def fromWASN(self, wasn: list[Node]):
        """
        Initialize `DANSEvariables` object based on `wasn`
        list of `Node` objects.
        """
        nNodes = len(wasn)  # number of nodes in WASN
        self.nPosFreqs = int(self.DFTsize // 2 + 1)  # number of >0 freqs.
        # Expected number of DANSE iterations (==  # of signal frames)
        self.nIter = int((wasn[0].data.shape[0] - self.DFTsize) / self.Ns) + 1

        avgProdResiduals = []   # average residuals product coming out of
                                # filter-shift processing (SRO estimation).
        bufferFlags = []
        dimYTilde = np.zeros(nNodes, dtype=int)   # dimension of \tilde{y}_k
        phaseShiftFactors = []
        Rnntilde = []   # autocorrelation matrix when VAD=0 
        Ryytilde = []   # autocorrelation matrix when VAD=1
        SROsEstimates = []  # SRO estimates per node (for each neighbor)
        SROsResiduals = []  # SRO residuals per node (for each neighbor)
        t = np.zeros((len(wasn[0].timeStamps), nNodes))  # time stamps
        wIR = []
        wTilde = []
        wTildeExt = []
        wTildeExtTarget = []
        yyH = []
        yyHuncomp = []
        yTilde = []
        yTildeHat = []
        yTildeHatUncomp = []
        z = []
        zBuffer = []
        zLocal = []
        for k in range(nNodes):
            nNeighbors = len(wasn[k].neighborsIdx)
            #
            avgProdResiduals.append(np.zeros(
                (self.DFTsize, nNeighbors),dtype=complex
                ))
            # init all buffer flags at 0 (assuming no over- or under-flow)
            bufferFlags.append(np.zeros((self.nIter, nNeighbors)))    
            #
            dimYTilde[k] = wasn[k].nSensors + nNeighbors
            # initiate phase shift factors as 0's (no phase shift)
            phaseShiftFactors.append(np.zeros(dimYTilde[k]))   
            #
            sliceTilde = np.finfo(float).eps *\
                (np.random.random((dimYTilde[k], dimYTilde[k])) +\
                    1j * np.random.random((dimYTilde[k], dimYTilde[k]))) 
            Rnntilde.append(np.tile(sliceTilde, (self.nPosFreqs, 1, 1)))
            Ryytilde.append(np.tile(sliceTilde, (self.nPosFreqs, 1, 1)))
            #
            SROsEstimates.append(np.zeros((self.nIter, nNeighbors)))
            SROsResiduals.append(np.zeros((self.nIter, nNeighbors)))
            #
            t[:, k] = wasn[k].timeStamps
            #
            wtmp = np.zeros((2 * self.DFTsize - 1, wasn[k].nSensors))
            # initialize time-domain filter as Dirac for first sensor signal
            wtmp[self.DFTsize, 0] = 1   
            wIR.append(wtmp)
            wtmp = np.zeros(
                (self.nPosFreqs, self.nIter + 1, dimYTilde[k]), dtype=complex)
            # initialize filter as a selector of the first sensor signal
            wtmp[:, :, 0] = 1
            wTilde.append(wtmp)
            wtmp = np.zeros((self.nPosFreqs, wasn[k].nSensors), dtype=complex)
            # initialize filter as a selector of the first sensor signal
            wtmp[:, 0] = 1
            wTildeExt.append(wtmp)
            wTildeExtTarget.append(wtmp)
            #
            yyH.append(np.zeros((self.nIter, self.nPosFreqs, dimYTilde[k],
                dimYTilde[k]), dtype=complex))
            yyHuncomp.append(np.zeros((self.nIter, self.nPosFreqs,
                dimYTilde[k], dimYTilde[k]), dtype=complex))
            yTilde.append(np.zeros((self.DFTsize, self.nIter, dimYTilde[k])))
            yTildeHat.append(np.zeros(
                (self.nPosFreqs, self.nIter, dimYTilde[k]), dtype=complex))
            yTildeHatUncomp.append(np.zeros(
                (self.nPosFreqs, self.nIter, dimYTilde[k]), dtype=complex))
            #
            z.append(np.empty((self.DFTsize, 0), dtype=float))
            zBuffer.append([np.array([]) for _ in range(nNeighbors)])
            zLocal.append(np.array([]))

        # Create fields
        self.avgProdResiduals = avgProdResiduals
        self.bufferFlags = bufferFlags
        self.d = np.zeros((wasn[self.referenceSensor].data.shape[0], nNodes))
        self.i = np.zeros(nNodes, dtype=int)
        self.dimYTilde = dimYTilde
        self.dhat = np.zeros(
            (self.nPosFreqs, self.nIter, nNodes), dtype=complex)
        self.expAvgBeta = [node.beta for node in wasn]
        self.flagIterations = [[] for _ in range(nNodes)]
        self.flagInstants = [[] for _ in range(nNodes)]
        self.idxBegChunk = None
        self.idxEndChunk = None
        self.lastExtFiltUp = np.zeros(nNodes)
        self.neighbors = [node.neighborsIdx for node in wasn]
        self.nInternalFilterUps = np.zeros(nNodes)
        self.nLocalMic = [node.data.shape[-1] for node in wasn]
        self.numUpdatesRyy = np.zeros(nNodes, dtype=int)
        self.numUpdatesRnn = np.zeros(nNodes, dtype=int)
        self.oVADframes = np.zeros(self.nIter)
        self.phaseShiftFactors = phaseShiftFactors
        self.phaseShiftFactorThroughTime = np.zeros((self.nIter))
        self.lastTDfilterUp = np.zeros(nNodes)
        self.Rnntilde = Rnntilde
        self.Ryytilde = Ryytilde
        self.SROsppm = np.array([node.sro for node in wasn])
        self.SROsEstimates = SROsEstimates
        self.SROsResiduals = SROsResiduals
        self.startUpdates = np.full(shape=(nNodes,), fill_value=False)
        self.timeInstants = t
        self.tStartForMetrics = np.zeros(nNodes)
        self.yyH = yyH
        self.yyHuncomp = yyHuncomp
        self.yTilde = yTilde
        self.yTildeHat = yTildeHat
        self.yTildeHatUncomp = yTildeHatUncomp
        self.wIR = wIR
        self.wTilde = wTilde
        self.wTildeExt = wTildeExt
        self.wTildeExtTarget = wTildeExtTarget
        self.z = z
        self.zBuffer = zBuffer
        self.zLocal = zLocal

        # VAD
        if wasn[0].vad.shape[-1] > 1: #TODO:
            raise ValueError('/!\ VAD for multiple desired sources not yet treated as special case. Using VAD for source #1!')
        nNodes = len(wasn)
        fullVAD = np.zeros((wasn[0].vad.shape[0], nNodes))
        for k in range(nNodes):  # for each node
            # TODO: multiple desired sources case not considered
            fullVAD[:, k] = wasn[k].vad[:, 0]
        self.fullVAD = fullVAD

        return self

    def broadcast(self, yk, tCurr, fs, k, p: DANSEparameters):
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
        p : DANSEparameters
            DANSE parameters.
        """

        # Extract correct frame of local signals
        ykFrame = base.local_chunk_for_broadcast(yk, tCurr, fs, p.DFTsize)

        if len(ykFrame) < p.DFTsize:

            print('Cannot perform compression: not enough local signals samples.')

        elif p.broadcastType == 'wholeChunk_td':

            # Time-domain chunk-wise broadcasting
            _, self.zLocal[k] = base.danse_compression_whole_chunk(
                ykFrame,
                self.wTildeExt[k],
                p.winWOLAanalysis,
                p.winWOLAsynthesis,
                zqPrevious=self.zLocal[k]
            )  # local compressed signals (time-domain)

            # Fill buffers in
            self.zBuffer = base.fill_buffers_whole_chunk(
                k,
                self.neighbors,
                self.zBuffer,
                self.zLocal[k][:(p.DFTsize // 2)]
            ) 
        
        elif p.broadcastType == 'fewSamples_td':
            # Time-domain broadcasting, `L` samples at a time,
            # via linear-convolution approximation of WOLA filtering process

            # Only update filter every so often
            updateBroadcastFilter = False
            if np.abs(tCurr - self.lastTDfilterUp[k]) >= p.upTDfilterEvery:
                updateBroadcastFilter = True
                self.lastTDfilterUp[k] = tCurr

            self.zLocal[k], self.wIR[k] = base.danse_compression_few_samples(
                ykFrame,
                self.wTildeExt[k],
                p.DFTsize,
                p.broadcastLength,
                self.wIR[k],
                p.winWOLAanalysis,
                p.winWOLAsynthesis,
                p.Ns,
                updateBroadcastFilter
            )  # local compressed signals

            self.zBuffer = base.fill_buffers_td_few_samples(
                k,
                self.neighbors,
                self.zBuffer,
                self.zLocal[k],
                p.broadcastLength
            )

        
    def update_and_estimate(self, yk, tCurr, fs, k,
                    p: DANSEparameters):
        """
        Update filter coefficient at current node
        and estimate corresponding desired signal frame.
        """

        # Process buffers
        self.process_incoming_signals_buffers(k, tCurr, p)
        # Wipe local buffers
        self.zBuffer[k] = [np.array([]) for _ in range(len(self.neighbors[k]))]
        # Construct `\tilde{y}_k` in frequency domain
        yLocalCurr = self.build_ytilde(yk, tCurr, fs, k, p)
        # Account for buffer flags
        skipUpdate = self.compensate_sros(yLocalCurr, k, tCurr, p)
        # Ryy and Rnn updates
        self.spatial_covariance_matrix_update(k)
        
        # Check quality of autocorrelations estimates 
        # -- once we start updating, do not check anymore.
        if not self.startUpdates[k] and \
            self.numUpdatesRyy[k] > np.amax(self.dimYTilde) and \
                self.numUpdatesRnn[k] > np.amax(self.dimYTilde):
            self.startUpdates[k] = True

        if self.startUpdates[k] and not p.bypassUpdates and not skipUpdate:
            # No `for`-loop versions
            if p.performGEVD:   # GEVD update
                self.perform_gevd_noforloop(k, p.GEVDrank, p.referenceSensor)
            else:   # regular update (no GEVD)
                self.perform_update_noforloop(k, p.referenceSensor)
            # Count the number of internal filter updates
            self.nInternalFilterUps[k] += 1  

            # Useful export for enhancement metrics computations
            if self.nInternalFilterUps[k] >= p.minFiltUpdatesForMetrics\
                and self.tStartForMetrics[k] is None:
                if p.compensateSROs and p.estimateSROs == 'CohDrift':
                    # Make sure SRO compensation has started
                    if self.nInternalFilterUps[k] > p.cohDrift.startAfterNups:
                        self.tStartForMetrics[k] = tCurr
                else:
                    self.tStartForMetrics[k] = tCurr
        else:
            # Do not update the filter coefficients
            self.wTilde[k][:, self.i[k] + 1, :] =\
                self.wTilde[k][:, self.i[k], :]
            if skipUpdate:
                print(f'Node {k+1}: {self.i[k]+1}^th update skipped.')
        if p.bypassUpdates:
            print('!! User-forced bypass of filter coefficients updates !!')

        # Update external filters (for broadcasting)
        self.update_external_filters(k, tCurr, p)
        # Update SRO estimates
        self.update_sro_estimates(k, p)
        # Update phase shifts for SRO compensation
        if p.compensateSROs:
            self.build_phase_shifts_for_srocomp(k, p)
        # Compute desired signal chunk estimate
        self.get_desired_signal(k, p)


    def update_external_filters(self, k, t, p: DANSEparameters):
        """
        Update external filters for relaxed filter update.
        To be used when using simultaneous or asynchronous node-updating.
        
        Parameters
        ----------
        k : int
            Receiving node index.
        t : float
            Current time instant [s].
        p : DANSEparameters object
            DANSE parameters.
        """
        self.wTildeExt[k] = self.expAvgBeta[k] * self.wTildeExt[k] +\
            (1 - self.expAvgBeta[k]) *  self.wTildeExtTarget[k]
        # Update targets
        if t - self.lastExtFiltUp[k] >= p.timeBtwExternalFiltUpdates:
            self.wTildeExtTarget[k] = (1 - p.alphaExternalFilters) *\
                self.wTildeExtTarget[k] + p.alphaExternalFilters *\
                self.wTilde[k][:, self.i[k] + 1, :self.nLocalMic[k]]
            # Update last external filter update instant [s]
            self.lastExtFiltUp[k] = t
            if p.printout_externalFilterUpdate:    # inform user
                print(f't={np.round(t, 3)}s -- UPDATING EXTERNAL FILTERS for node {k+1} (scheduled every [at least] {p.timeBtwExternalFiltUpdates}s)')


    def process_incoming_signals_buffers(self, k, t, p: DANSEparameters):
        """
        Processes the incoming data from other nodes, as stored in local node's
        buffers. Called whenever a DANSE update can be performed
        (`N` new local samples were captured since last update).
        
        Parameters
        ----------
        k : int
            Receiving node index.
        t : float
            Current time instant [s].
        p : DANSEparameters object
            DANSE parameters.
        """

        # Useful renaming
        Ndft = p.DFTsize
        Ns = p.Ns
        Lbc = p.broadcastLength

        # Initialize compressed signal matrix
        # ($\mathbf{z}_{-k}$ in [1]'s notation)
        zk = np.empty((p.DFTsize, 0), dtype=float)

        # Initialise flags vector (overflow: >0; underflow: <0; or none: ==0)
        bufferFlags = np.zeros(len(self.neighbors[k]))

        for idxq in range(len(self.neighbors[k])):
            
            Bq = len(self.zBuffer[k][idxq])  # buffer size for neighbour `q`

            # Time-domain chunks broadcasting
            if p.broadcastType == 'wholeChunk_td':
                if self.i[k] == 0:
                    if Bq == Ns:
                        # Not yet any previous buffer
                        # -- need to appstart zeros.
                        zCurrBuffer = np.concatenate((
                            np.zeros(Ndft - Bq),
                            self.zBuffer[k][idxq]
                        ))
                    elif Bq == 0:
                        # Node `q` has not yet transmitted enough data to node
                        # `k`, but node `k` has already reached its first
                        # update instant. Interpretation: Node `q` samples
                        # slower than node `k`. 
                        print(f'[b- @ t={np.round(t, 3)}s] Buffer underflow at current node`s B_{self.neighbors[k][idxq]+1} buffer | -1 broadcast')
                        bufferFlags[idxq] = -1      # raise negative flag
                        zCurrBuffer = np.zeros(Ndft)
                else:
                    if Bq == Ns:
                        # All good, no under-/over-flows
                        if not np.any(self.z[k]):
                            # Not yet any previous buffer
                            # -- need to appstart zeros.
                            zCurrBuffer = np.concatenate((
                                np.zeros(Ndft - Bq), self.zBuffer[k][idxq]
                            ))
                        else:
                            # Concatenate last `Ns` samples of previous buffer
                            # with current buffer.
                            zCurrBuffer = np.concatenate((
                                self.z[k][-Ns:, idxq], self.zBuffer[k][idxq]
                            ))
                    else:
                        # Under- or over-flow...
                        raise ValueError('[NOT YET IMPLEMENTED]')
                    
            elif p.broadcastType == 'fewSamples_td':

                if self.i[k] == 0: # first DANSE iteration case 
                    # -- we are expecting an abnormally full buffer,
                    # with an entire DANSE chunk size inside of it
                    if Bq == Ndft: 
                        # There is no significant SRO between node `k` and `q`.
                        # Response: `k` uses all samples in the `q` buffer.
                        zCurrBuffer = self.zBuffer[k][idxq]
                    elif (Ndft - Bq) % Lbc == 0 and Bq < Ndft:
                        # Node `q` has not yet transmitted enough data to node
                        # `k`, but node `k` has already reached its first
                        # update instant. Interpretation: Node `q` samples
                        # slower than node `k`. 
                        nMissingBroadcasts = int(np.abs((Ndft - Bq) / Lbc))
                        print(f'[b- @ t={np.round(t, 3)}s] Buffer underflow at current node`s B_{self.neighbors[k][idxq]+1} buffer | -{nMissingBroadcasts} broadcast(s)')
                        # Raise negative flag
                        bufferFlags[idxq] = -1 * nMissingBroadcasts
                        zCurrBuffer = np.concatenate(
                            (np.zeros(Ndft - Bq), self.zBuffer[k][idxq]),
                            axis=0
                        )
                    elif (Ndft - Bq) % Lbc == 0 and Bq > Ndft:
                        # Node `q` has already transmitted too much data
                        # to node `k`. Interpretation: Node `q` samples faster
                        # than node `k`.
                        nExtraBroadcasts = int(np.abs((Ndft - Bq) / Lbc))
                        print(f'[b+ @ t={np.round(t, 3)}s] Buffer overflow at current node`s B_{self.neighbors[k][idxq]+1} buffer | +{nExtraBroadcasts} broadcasts(s)')
                        # Raise positive flag
                        bufferFlags[idxq] = +1 * nExtraBroadcasts
                        zCurrBuffer = self.zBuffer[k][idxq][-Ndft:]
                
                else:   # not the first DANSE iteration 
                    # -- we are expecting a normally full buffer,
                    # with a DANSE chunk size considering overlap.

                    # case 1: no mismatch between node `k` and node `q`.
                    if Bq == Ns:
                        pass
                    # case 2: negative mismatch
                    elif (Ns - Bq) % Lbc == 0 and Bq < Ns:
                        nMissingBroadcasts = int(np.abs((Ns - Bq) / Lbc))
                        print(f'[b- @ t={np.round(t, 3)}s] Buffer underflow at current node`s B_{self.neighbors[k][idxq]+1} buffer | -{nMissingBroadcasts} broadcast(s)')
                        # Raise negative flag
                        bufferFlags[idxq] = -1 * nMissingBroadcasts
                    # case 3: positive mismatch
                    elif (Ns - Bq) % Lbc == 0 and Bq > Ns:       
                        nExtraBroadcasts = int(np.abs((Ns - Bq) / Lbc))
                        print(f'[b+ @ t={np.round(t, 3)}s] Buffer overflow at current node`s B_{self.neighbors[k][idxq]+1} buffer | +{nExtraBroadcasts} broadcasts(s)')
                        # Raise positive flag
                        bufferFlags[idxq] = +1 * nExtraBroadcasts
                    else:
                        if (Ns - Bq) % Lbc != 0 and\
                            np.abs(self.i[k] - (self.nIter - 1)) < 10:
                            print('[b! @ t={np.round(t, 3)}s] This is the last iteration -- not enough samples anymore due to cumulated SROs effect, skip update.')
                            # Raise "end of signal" flag
                            bufferFlags[idxq] = np.NaN
                        else:
                            raise ValueError(f'Unexpected buffer size ({Bq} samples, with L={Lbc} and N={Ns}) for neighbor node q={self.neighbors[k][idxq]+1}.')
                    # Build current buffer
                    if Ndft - Bq > 0:
                        zCurrBuffer = np.concatenate(
                            (self.z[k][-(Ndft - Bq):, idxq],
                            self.zBuffer[k][idxq]),
                            axis=0
                        )
                    else:   # edge case: no overlap between consecutive frames
                        zCurrBuffer = self.zBuffer[k][idxq]

            # Stack compressed signals
            zk = np.concatenate((zk, zCurrBuffer[:, np.newaxis]), axis=1)

        # Update DANSE variables
        self.z[k] = zk
        self.bufferFlags[k][self.i[k], :] = bufferFlags

    
    def build_ytilde(self, yk, tCurr, fs, k, p: DANSEparameters):
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
        yLocalCurr : [N x Mk] np.ndarray (float)
            Time chunk of local sensor signals.
        """

        # Extract current local data chunk
        yLocalCurr, self.idxBegChunk, self.idxEndChunk =\
            base.local_chunk_for_update(yk, tCurr, fs, p)

        # Compute VAD
        VADinFrame = self.fullVAD[
            np.amax([self.idxBegChunk, 0]):self.idxEndChunk, k
            ]
        # If there is a majority of "VAD = 1" in the frame,
        # set the frame-wise VAD to 1.
        self.oVADframes[self.i[k]] =\
            sum(VADinFrame == 0) <= len(VADinFrame) // 2
        # Count number of spatial covariance matrices updates
        if self.oVADframes[self.i[k]]:
            self.numUpdatesRyy[k] += 1
        else:
            self.numUpdatesRnn[k] += 1

        # Build full available observation vector
        yTildeCurr = np.concatenate((yLocalCurr, self.z[k]), axis=1)
        self.yTilde[k][:, self.i[k], :] = yTildeCurr
        # Go to frequency domain
        yTildeHatCurr = 1 / p.normFactWOLA * np.fft.fft(
            self.yTilde[k][:, self.i[k], :] *\
                p.winWOLAanalysis[:, np.newaxis],
            p.DFTsize,
            axis=0
        )
        # Keep only positive frequencies
        self.yTildeHat[k][:, self.i[k], :] = yTildeHatCurr[:self.nPosFreqs, :]

        return yLocalCurr


    def compensate_sros(self, k, t, p: DANSEparameters):
        """
        Compensate for SROs based on estimates, accounting for full-sample 
        drift flags.

        Parameters
        ----------
        k : int
            Receiving node index.
        t : float
            Current time instant [s].
        p : DANSEparameters object
            DANSE parameters.

        Returns
        -------
        skipUpdate : bool
            If True, skip next filter update: not enough samples
            due to cumulated SRO effect.
        """
        # Init
        skipUpdate = False
        extraPhaseShiftFactor = np.zeros(self.dimYTilde[k])

        for q in range(len(self.neighbors[k])):
            if not np.isnan(self.bufferFlags[k][self.i[k], q]):
                extraPhaseShiftFactor[self.nLocalMic[k] + q] =\
                    self.bufferFlags[k][self.i[k], q] * p.broadcastLength
                # ↑↑↑ if `bufferFlags[k][i[k], q] == 0`,
                # `extraPhaseShiftFactor = 0` and no additional phase shift.
                if self.bufferFlags[k][self.i[k], q] != 0:
                    # keep flagging iterations in memory
                    self.flagIterations[k].append(self.i[k])
                    # keep flagging instants in memory
                    self.flagInstants[k].append(t)
            else:
                # From `process_incoming_signals_buffers`: 
                # "Not enough samples due to cumulated SROs effect, skip upd."
                skipUpdate = True
        # Save uncompensated \tilde{y} for coherence-drift-based SRO estimation
        self.yTildeHatUncomp[k][:, self.i[k], :] = copy.copy(
            self.yTildeHat[k][:, self.i[k], :]
        )
        self.yyHuncomp[k][self.i[k], :, :, :] = np.einsum(
            'ij,ik->ijk',
            self.yTildeHatUncomp[k][:, self.i[k], :],
            self.yTildeHatUncomp[k][:, self.i[k], :].conj()
        )

        # Compensate SROs
        if p.compensateSROs:
            # Complete phase shift factors
            self.phaseShiftFactors[k] += extraPhaseShiftFactor
            if k == 0:  # Save for plotting
                self.phaseShiftFactorThroughTime[self.i[k]:] =\
                    self.phaseShiftFactors[k][self.nLocalMic[k] + q]
            # Apply phase shift factors
            self.yTildeHat[k][:, self.i[k], :] *=\
                np.exp(-1 * 1j * 2 * np.pi / p.DFTsize *\
                    np.outer(
                        np.arange(self.nPosFreqs),
                        self.phaseShiftFactors[k]
                    ))

        return skipUpdate


    def spatial_covariance_matrix_update(self, k):
        """
        Performs the spatial covariance matrices updates.
        
        Parameters
        ----------
        k : int
            Node index.
        """
        # Useful renaming 
        y = self.yTildeHat[k][:, self.i[k], :]
        
        yyH = np.einsum('ij,ik->ijk', y, y.conj())

        if self.oVADframes[self.i[k]]:
            self.Ryytilde[k] = self.expAvgBeta[k] * self.Ryytilde[k] +\
                (1 - self.expAvgBeta[k]) * yyH  # update signal + noise matrix
        else:     
            self.Rnntilde[k] = self.expAvgBeta[k] * self.Rnntilde[k] +\
                (1 - self.expAvgBeta[k]) * yyH  # update noise-only matrix

        self.yyH[k][self.i[k], :, :, :] = yyH

    
    def perform_gevd_noforloop(self, k, rank=1, refSensorIdx=0):
        """
        GEVD computations for DANSE, `for`-loop free.
        
        Parameters
        ----------
        k : int
            Node index.
        rank : int
            GEVD rank approximation.
        refSensorIdx : int
            Index of the reference sensor (>=0).
        """
        # ------------ for-loop-free estimate ------------
        n = self.Ryytilde[k].shape[-1]
        nFreqs = self.Ryytilde[k].shape[0]
        # Reference sensor selection vector 
        Evect = np.zeros((n,))
        Evect[refSensorIdx] = 1

        sigma = np.zeros((nFreqs, n))
        Xmat = np.zeros((nFreqs, n, n), dtype=complex)

        # t0 = time.perf_counter()
        for kappa in range(nFreqs):
            # Perform generalized eigenvalue decomposition 
            # -- as of 2022/02/17: scipy.linalg.eigh()
            # seemingly cannot be jitted nor vectorized.
            sigmacurr, Xmatcurr = sla.eigh(
                self.Ryytilde[k][kappa, :, :],
                self.Rnntilde[k][kappa, :, :],
                check_finite=False,
                driver='gvd'
            )
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

        # Udpate filter
        self.wTilde[k][:, self.i[k] + 1, :] = w


    def perform_update_noforloop(self, k, refSensorIdx=0):
        """
        Regular DANSE update computations, `for`-loop free.
        No GEVD involved here.
        
        Parameters
        ----------
        k : int
            Node index.
        refSensorIdx : int
            Index of the reference sensor (>=0).
        """
        # Reference sensor selection vector
        Evect = np.zeros((self.Ryytilde[k].shape[-1],))
        Evect[refSensorIdx] = 1

        # Cross-correlation matrix update 
        ryd = np.matmul(self.Ryytilde[k] - self.Rnntilde[k], Evect)
        # Update node-specific parameters of node k
        Ryyinv = np.linalg.inv(self.Ryytilde[k])
        w = np.matmul(Ryyinv, ryd[:,:,np.newaxis])
        w = w[:, :, 0]  # get rid of singleton dimension

        # Update filter
        self.wTilde[k][:, self.i[k] + 1, :] = w


    def update_sro_estimates(self, k, p: DANSEparameters):
        """
        Update SRO estimates.
        
        Parameters
        ----------
        k : int
            Node index.
        p : DANSEparameters object
            Parameters.
        """
        # Useful variables (compact coding)
        nNeighs = len(self.neighbors[k])
        iter = self.i[k]
        bufferFlagPos = p.broadcastLength * np.sum(
            self.bufferFlags[k][:(iter + 1), :],
            axis=0
        )
        bufferFlagPri = p.broadcastLength * np.sum(
            self.bufferFlags[k][:(iter - p.cohDrift.segLength + 1), :],
            axis=0
        )
        
        # DANSE filter update indices
        # corresponding to "Filter-shift" SRO estimate updates.
        cohDriftSROupdateIndices = np.arange(
            start=p.cohDrift.startAfterNups + p.cohDrift.estEvery,
            stop=self.nIter,
            step=p.cohDrift.estEvery
        )
        
        # Init arrays
        sroOut = np.zeros(nNeighs)
        if p.estimateSROs == 'CohDrift':
            
            ld = p.cohDrift.segLength

            if iter in cohDriftSROupdateIndices:

                flagFirstSROEstimate = False
                if iter == np.amin(cohDriftSROupdateIndices):
                    # Let `cohdrift_sro_estimation()` know that
                    # this is the 1st SRO estimation round.
                    flagFirstSROEstimate = True

                # Residuals method
                for q in range(nNeighs):
                    # index of the compressed signal from node `q` inside `yyH`
                    idxq = self.nLocalMic + q     
                    if p.cohDrift.loop == 'closed':
                        # Use SRO-compensated correlation matrix entries
                        # (closed-loop SRO est. + comp.).

                        # A posteriori coherence
                        cohPosteriori = (self.yyH[iter, :, 0, idxq]
                            / np.sqrt(self.yyH[iter, :, 0, 0] *\
                                self.yyH[iter, :, idxq, idxq]))
                        # A priori coherence
                        cohPriori = (self.yyH[iter - ld, :, 0, idxq]
                            / np.sqrt(self.yyH[iter - ld, :, 0, 0] *\
                                self.yyH[iter - ld, :, idxq, idxq]))
                        
                        # Set buffer flags to 0
                        bufferFlagPri = np.zeros_like(bufferFlagPri)
                        bufferFlagPos = np.zeros_like(bufferFlagPos)

                    elif p.cohDrift.loop == 'open':
                        # Use SRO-_un_compensated correlation matrix entries
                        # (open-loop SRO est. + comp.).

                        # A posteriori coherence
                        cohPosteriori = (self.yyHuncomp[iter, :, 0, idxq]
                            / np.sqrt(self.yyHuncomp[iter, :, 0, 0] *\
                                self.yyHuncomp[iter, :, idxq, idxq]))
                        # A priori coherence
                        cohPriori = (self.yyHuncomp[iter - ld, :, 0, idxq]
                            / np.sqrt(self.yyHuncomp[iter - ld, :, 0, 0] *\
                                self.yyHuncomp[iter - ld, :, idxq, idxq]))

                    # Perform SRO estimation via coherence-drift method
                    sroRes, apr = sros.cohdrift_sro_estimation(
                        wPos=cohPosteriori,
                        wPri=cohPriori,
                        avgResProd=self.avgProdResiduals[:, q],
                        Ns=p.Ns,
                        ld=ld,
                        method=p.cohDrift.estimationMethod,
                        alpha=p.cohDrift.alpha,
                        flagFirstSROEstimate=flagFirstSROEstimate,
                        bufferFlagPri=bufferFlagPri[q],
                        bufferFlagPos=bufferFlagPos[q]
                    )
                
                    sroOut[q] = sroRes
                    self.avgProdResiduals[:, q] = apr

        elif p.estimateSROs == 'Oracle':
            # No data-based dynamic SRO estimation: use oracle knowledge
            sroOut = (self.SROsppm[self.neighbors[k]] - self.SROsppm[k]) * 1e-6

        # Save SRO (residuals)
        self.SROsResiduals[k][iter, :] = sroOut

    
    def build_phase_shifts_for_srocomp(self, k, p: DANSEparameters):
        """
        Computed appropriate phase shift factors for next SRO compensation.
        
        Parameters
        ----------
        k : int
            Node index.
        p : DANSEparameters object
            Parameters.
        """

        for q in range(len(self.neighbors[k])):
            if p.estimateSROs == 'CohDrift':
                if p.cohDrift.loop == 'closed':
                    # Increment estimate using SRO residual
                    self.SROsEstimates[k][self.i[k], q] +=\
                        self.SROsResiduals[k][self.i[k], q] /\
                        (1 + self.SROsResiduals[k][self.i[k], q]) *\
                        p.cohDrift.alphaEps
                elif p.cohDrift.loop == 'open':
                    # Use SRO "residual" as estimates
                    self.SROsEstimates[k][self.i[k], q] =\
                        self.SROsResiduals[k][self.i[k], q] /\
                        (1 + self.SROsResiduals[k][self.i[k], q])
            # Increment phase shift factor recursively.
            # (valid directly for oracle SRO "estimation")
            self.phaseShiftFactors[k][self.nLocalMic[k] + q] -=\
                self.SROsEstimates[k][self.i[k], q] * p.Ns 

    
    def get_desired_signal(self, k, p: DANSEparameters):
        """
        Compute chunk of desired signal from DANSE freq.-domain filters
        and freq.-domain observation vector y_tilde.

        Parameters
        ----------
        k : int
            Node index.
        p : DANSEparameters object
            Parameters.
        """
        
        # Useful renaming (compact code)
        w = self.wTilde[k][:, self.i[k] + 1, :]
        y = self.yTildeHat[k][:, self.i[k], :]
        win = p.winWOLAsynthesis
        dChunk = self.d[self.idxBegChunk:self.idxEndChunk, k]
        yTD = self.yTilde[k][:, self.i[k], :self.nLocalMic[k]]
            
        dhatCurr = None  # init

        if p.desSigProcessingType == 'wola':
            # Compute desired signal chunk estimate using WOLA
            dhatCurr = np.einsum('ij,ij->i', w.conj(), y)
            # Transform back to time domain (WOLA processing)
            dChunCurr = p.normFactWOLA * win * base.back_to_time_domain(dhatCurr, len(win))
            # Overlap and add construction of output time-domain signal
            if len(dChunk) < len(win):
                dChunk += np.real_if_close(dChunCurr[-len(dChunk):])
            else:
                dChunk += np.real_if_close(dChunCurr)

        elif p.desSigProcessingType == 'conv':
            # Compute desired signal chunk estimate using T(z) approximation
            wIR = base.dist_fct_approx(w, win, win, p.Ns)
            # Perform convolution
            yfiltLastSamples = np.zeros((p.Ns, yTD.shape[-1]))
            for m in range(yTD.shape[-1]):
                # Indices required from convolution output vvv
                idDesired = np.arange(start=len(wIR) - p.Ns, stop=len(wIR))
                tmp = base.extract_few_samples_from_convolution(
                    idDesired,
                    wIR[:, m],
                    yTD[:, m]
                )
                yfiltLastSamples[:, m] = tmp

            dChunk = np.sum(yfiltLastSamples, axis=1)

        # Update estimates
        self.dhat[:, self.i[k], k] = dhatCurr
        if p.desSigProcessingType == 'wola':
            self.d[self.idxBegChunk:self.idxEndChunk, k] = dChunk
        elif p.desSigProcessingType == 'conv':
            self.d[self.idxEndChunk - p.Ns:self.idxEndChunk, k] = dChunk


@dataclass
class DANSEoutputs:
    """
    Dataclass to assemble all useful outputs
    of the DANSE algorithm.
    """
    
    def fromVariables(self, dv: DANSEvariables):
        """
        Selects useful output values from `DANSEvariables` object
        after DANSE processing.
        """

        # Desired signal estimates
        self.TDdesiredSignals = dv.d
        self.STFTDdesiredSignals = dv.dhat
        # SROs
        self.SROsEstimates = dv.SROsEstimates
        self.SROsResiduals = dv.SROsResiduals
        # Filters
        self.filters = dv.wTilde

        return self


@dataclass
class DANSEeventInstant:
    t: float = 0.   # event time instant [s]
    nodes: np.ndarray = np.array([0])   # node(s) concerned
    type: list[str] = field(default_factory=list)   # event type

    def __post_init__(self):
        self.nEvents = len(self.nodes)


