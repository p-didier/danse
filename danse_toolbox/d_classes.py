import copy
import numpy as np
import scipy.linalg as sla
from siggen.classes import Node
from dataclasses import dataclass, field
import danse_toolbox.d_base as base
import danse_toolbox.d_sros as sros


@dataclass
class DANSEvariables(base.DANSEparameters):
    """
    Main DANSE class. Stores all relevant variables and core functions on 
    those variables.
    """
    def import_params(self, p: base.DANSEparameters):
        self.__dict__.update(p.__dict__)

    def init_from_wasn(self, wasn: list[Node]):
        """
        Initialize `DANSEvariables` object based on `wasn`
        list of `Node` objects.
        """
        nNodes = len(wasn)  # number of nodes in WASN
        nSensorsTotal = sum([wasn[k].nSensors for k in range(nNodes)])
        self.nPosFreqs = int(self.DFTsize // 2 + 1)  # number of >0 freqs.
        # Expected number of DANSE iterations (==  # of signal frames)
        self.nIter = int((wasn[0].data.shape[0] - self.DFTsize) / self.Ns) + 1

        avgProdResiduals = []   # average residuals product coming out of
                                # filter-shift processing (SRO estimation).
        bufferFlags = []
        dimYTilde = np.zeros(nNodes, dtype=int)   # dimension of \tilde{y}_k
        phaseShiftFactors = []
        Rnncentr = []   # autocorrelation matrix when VAD=0 [centralised]
        Ryycentr = []   # autocorrelation matrix when VAD=1 [centralised]
        Rnnlocal = []   # autocorrelation matrix when VAD=0 [local]
        Ryylocal = []   # autocorrelation matrix when VAD=1 [local]
        Rnntilde = []   # autocorrelation matrix when VAD=0 [DANSE]
        Ryytilde = []   # autocorrelation matrix when VAD=1 [DANSE]
        SROsEstimates = []  # SRO estimates per node (for each neighbor)
        SROsResiduals = []  # SRO residuals per node (for each neighbor)
        t = np.zeros((len(wasn[0].timeStamps), nNodes))  # time stamps
        wIR = []
        wCentr = []
        wLocal = []
        wTilde = []
        wTildeExt = []
        wTildeExtTarget = []
        yyH = []
        yyHuncomp = []
        yCentr = []
        yHatCentr = []
        yHatLocal = []
        yLocal = []
        yTilde = []
        yTildeHat = []
        yTildeHatUncomp = []
        z = []
        zBuffer = []
        zLocal = []

        def _init_complex_filter(size, refIdx=0):
            """Returns an initialized STFT-domain filter vector,
            i.e., a selector of the reference sensor at node 1."""
            wtmp = np.zeros(size, dtype=complex)
            if len(size) == 3:
                wtmp[:, :, refIdx] = 1
            elif len(size) == 2:
                wtmp[:, refIdx] = 1
            return wtmp

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
            # TODO: centralised and local covariance matrices should
            # contain the same _local_ part as the 'tilde' covariance
            # matrices... 
            sliceCentr = np.finfo(float).eps *\
                (np.random.random((nSensorsTotal, nSensorsTotal)) +\
                1j * np.random.random((nSensorsTotal, nSensorsTotal))) 
            Rnncentr.append(np.tile(sliceCentr, (self.nPosFreqs, 1, 1)))
            Ryycentr.append(np.tile(sliceCentr, (self.nPosFreqs, 1, 1)))
            sliceLocal = np.finfo(float).eps *\
                (np.random.random((wasn[k].nSensors, wasn[k].nSensors)) +\
                1j * np.random.random((wasn[k].nSensors, wasn[k].nSensors))) 
            Rnnlocal.append(np.tile(sliceLocal, (self.nPosFreqs, 1, 1)))
            Ryylocal.append(np.tile(sliceLocal, (self.nPosFreqs, 1, 1)))
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
            wTilde.append(_init_complex_filter(
                (self.nPosFreqs, self.nIter + 1, dimYTilde[k]),
                self.referenceSensor
            ))
            wTildeExt.append(_init_complex_filter(
                (self.nPosFreqs, wasn[k].nSensors),
                self.referenceSensor
            ))
            wTildeExtTarget.append(_init_complex_filter(
                (self.nPosFreqs, wasn[k].nSensors),
                self.referenceSensor
            ))
            wCentr.append(_init_complex_filter(
                (self.nPosFreqs, self.nIter + 1, nSensorsTotal),
                self.referenceSensor
            ))
            wLocal.append(_init_complex_filter(
                (self.nPosFreqs, self.nIter + 1, wasn[k].nSensors),
                self.referenceSensor
            ))
            #
            yCentr.append(np.zeros(
                (self.DFTsize, self.nIter, nSensorsTotal)))
            yLocal.append(np.zeros(
                (self.DFTsize, self.nIter, wasn[k].nSensors)))
            yHatCentr.append(np.zeros(
                (self.nPosFreqs, self.nIter, nSensorsTotal), dtype=complex))
            yHatLocal.append(np.zeros(
                (self.nPosFreqs, self.nIter, wasn[k].nSensors), dtype=complex))
            yTilde.append(np.zeros((self.DFTsize, self.nIter, dimYTilde[k])))
            yTildeHat.append(np.zeros(
                (self.nPosFreqs, self.nIter, dimYTilde[k]), dtype=complex))
            yTildeHatUncomp.append(np.zeros(
                (self.nPosFreqs, self.nIter, dimYTilde[k]), dtype=complex))
            yyH.append(np.zeros((self.nIter, self.nPosFreqs, dimYTilde[k],
                dimYTilde[k]), dtype=complex))
            yyHuncomp.append(np.zeros((self.nIter, self.nPosFreqs,
                dimYTilde[k], dimYTilde[k]), dtype=complex))
            #
            z.append(np.empty((self.DFTsize, 0), dtype=float))
            zBuffer.append([np.array([]) for _ in range(nNeighbors)])
            zLocal.append(np.array([]))

        # Create fields
        self.avgProdResiduals = avgProdResiduals
        self.bufferFlags = bufferFlags
        self.d = np.zeros(
            (wasn[self.referenceSensor].data.shape[0], nNodes)
        )
        self.dCentr = np.zeros(
            (wasn[self.referenceSensor].data.shape[0], nNodes)
        )
        self.dLocal = np.zeros(
            (wasn[self.referenceSensor].data.shape[0], nNodes)
        )
        self.i = np.zeros(nNodes, dtype=int)
        self.dimYTilde = dimYTilde
        self.dhat = np.zeros(
            (self.nPosFreqs, self.nIter, nNodes), dtype=complex
        )
        self.dHatCentr = np.zeros(
            (self.nPosFreqs, self.nIter, nNodes), dtype=complex
        )
        self.dHatLocal = np.zeros(
            (self.nPosFreqs, self.nIter, nNodes), dtype=complex
        )
        self.expAvgBeta = [node.beta for node in wasn]
        self.flagIterations = [[] for _ in range(nNodes)]
        self.flagInstants = [[] for _ in range(nNodes)]
        self.fullVAD = [node.vadCombined for node in wasn]
        self.idxBegChunk = None
        self.idxEndChunk = None
        self.lastExtFiltUp = np.zeros(nNodes)
        self.neighbors = [node.neighborsIdx for node in wasn]
        self.nCentrFilterUps = np.zeros(nNodes)
        self.nLocalFilterUps = np.zeros(nNodes)
        self.nInternalFilterUps = np.zeros(nNodes)
        self.nLocalMic = [node.data.shape[-1] for node in wasn]
        self.numUpdatesRyy = np.zeros(nNodes, dtype=int)
        self.numUpdatesRnn = np.zeros(nNodes, dtype=int)
        self.oVADframes = np.zeros(self.nIter)
        self.phaseShiftFactors = phaseShiftFactors
        self.phaseShiftFactorThroughTime = np.zeros((self.nIter))
        self.lastBroadcastInstant = np.zeros(nNodes)
        self.lastTDfilterUp = np.zeros(nNodes)
        self.Rnncentr = Rnncentr
        self.Ryycentr = Ryycentr
        self.Rnnlocal = Rnnlocal
        self.Ryylocal = Ryylocal
        self.Rnntilde = Rnntilde
        self.Ryytilde = Ryytilde
        self.SROsppm = np.array([node.sro for node in wasn])
        self.SROsEstimates = SROsEstimates
        self.SROsResiduals = SROsResiduals
        self.startUpdates = np.full(shape=(nNodes,), fill_value=False)
        self.startUpdatesCentr = np.full(shape=(nNodes,), fill_value=False)
        self.startUpdatesLocal = np.full(shape=(nNodes,), fill_value=False)
        self.timeInstants = t
        self.tStartForMetrics = np.full(shape=(nNodes,), fill_value=None)
        self.tStartForMetricsCentr = np.full(shape=(nNodes,), fill_value=None)
        self.tStartForMetricsLocal = np.full(shape=(nNodes,), fill_value=None)
        self.yin = [node.data for node in wasn]
        self.yyH = yyH
        self.yyHuncomp = yyHuncomp
        self.yCentr = yCentr
        self.yHatCentr = yHatCentr
        self.yLocal = yLocal
        self.yHatLocal = yHatLocal
        self.yTilde = yTilde
        self.yTildeHat = yTildeHat
        self.yTildeHatUncomp = yTildeHatUncomp
        self.wIR = wIR
        self.wTilde = wTilde
        self.wCentr = wCentr
        self.wLocal = wLocal
        self.wTildeExt = wTildeExt
        self.wTildeExtTarget = wTildeExtTarget
        self.z = z
        self.zBuffer = zBuffer
        self.zLocal = zLocal

        # For centralised and local estimates
        self.yinStacked = np.concatenate(tuple([x for x in self.yin]), axis=-1)

        return self

    def broadcast(self, tCurr, fs, k):
        """
        Parameters
        ----------
        tCurr : float
            Broadcast event global time instant [s].
        fs : float
            Node's sampling frequency [Hz].
        k : int
            Node index.
        """

        # Extract correct frame of local signals
        ykFrame = base.local_chunk_for_broadcast(self.yin[k], tCurr, fs, self.DFTsize)

        if len(ykFrame) < self.DFTsize:

            print('Cannot perform compression: not enough local signals samples.')

        elif self.broadcastType == 'wholeChunk':

            # Time-domain chunk-wise broadcasting
            _, self.zLocal[k] = base.danse_compression_whole_chunk(
                ykFrame,
                self.wTildeExt[k],
                self.winWOLAanalysis,
                self.winWOLAsynthesis,
                zqPrevious=self.zLocal[k]
            )  # local compressed signals (time-domain)

            # Fill buffers in
            self.zBuffer = base.fill_buffers_whole_chunk(
                k,
                self.neighbors,
                self.zBuffer,
                self.zLocal[k][:(self.DFTsize // 2)]
            ) 
        
        elif self.broadcastType == 'fewSamples':
            # Time-domain broadcasting, `L` samples at a time,
            # via linear-convolution approximation of WOLA filtering process

            # Only update filter every so often
            updateBroadcastFilter = False
            if np.abs(tCurr - self.lastTDfilterUp[k]) >= self.upTDfilterEvery:
                updateBroadcastFilter = True
                self.lastTDfilterUp[k] = tCurr

            # If "efficient" events for broadcast
            # (unnecessary broadcast instants were aggregated):
            if self.efficientSpSBC:
                # Count samples recorded since the last broadcast at node `k`
                # and consequently adapt the `L` "broadcast length" variable
                # used in `danse_compression_few_samples` and
                # `fill_buffers_td_few_samples`.
                nSamplesSinceLastBroadcast = ((self.timeInstants[:, k] >\
                    self.lastBroadcastInstant[k]) &\
                    (self.timeInstants[:, k] <= tCurr)).sum()
                self.lastBroadcastInstant[k] = tCurr
                currL = nSamplesSinceLastBroadcast
            else:
                currL = self.broadcastLength

            self.zLocal[k], self.wIR[k] = base.danse_compression_few_samples(
                ykFrame,
                self.wTildeExt[k],
                currL,
                self.wIR[k],
                self.winWOLAanalysis,
                self.winWOLAsynthesis,
                self.Ns,
                updateBroadcastFilter
            )  # local compressed signals

            self.zBuffer = base.fill_buffers_td_few_samples(
                k,
                self.neighbors,
                self.zBuffer,
                self.zLocal[k],
                currL
            )

        
    def update_and_estimate(self, tCurr, fs, k):
        """
        Update filter coefficient at current node
        and estimate corresponding desired signal frame.

        Parameters
        ----------
        tCurr : float
            Current time instant [s].
        fs : float
            Node `k`'s sampling frequency [Hz].
        k : int
            Receiving node index.
        """

        # Process buffers
        self.process_incoming_signals_buffers(k, tCurr)
        # Wipe local buffers
        self.zBuffer[k] = [np.array([])\
            for _ in range(len(self.neighbors[k]))]
        # Construct `\tilde{y}_k` in frequency domain and VAD at current frame
        self.build_ytilde(tCurr, fs, k)
        # Consider local / centralised estimation(s)
        if self.computeCentralised:
            self.build_ycentr(tCurr, fs, k)
        if self.computeLocal:  # extract local info from `\tilde{y}_k`
            self.yLocal[k][:, self.i[k], :] =\
                self.yTilde[k][:, self.i[k], :self.nSensorPerNode[k]]
            self.yHatLocal[k][:, self.i[k], :] =\
                self.yTildeHat[k][:, self.i[k], :self.nSensorPerNode[k]]
        # Account for buffer flags
        skipUpdate = self.compensate_sros(k, tCurr)
        # Ryy and Rnn updates (including centralised / local, if needed)
        self.spatial_covariance_matrix_update(k)
        # Check quality of covariance matrix estimates 
        self.check_covariance_matrices(k)

        if not skipUpdate:
            # If covariance matrices estimates are full-rank, update filters
            self.perform_update(k)
            # ^^^ depends on outcome of `check_covariance_matrices()`.
            # if self.tStartForMetrics[k] is None:
            self.evaluate_tstart_for_metrics_computation(k, tCurr)
        else:
            # Do not update the filter coefficients
            self.wTilde[k][:, self.i[k] + 1, :] =\
                self.wTilde[k][:, self.i[k], :]
            if skipUpdate:
                print(f'Node {k+1}: {self.i[k]+1}^th update skipped.')
        if self.bypassUpdates:
            print('!! User-forced bypass of filter coefficients updates !!')

        # Update external filters (for broadcasting)
        self.update_external_filters(k, tCurr)
        # Update SRO estimates
        self.update_sro_estimates(k)
        # Update phase shifts for SRO compensation
        if self.compensateSROs:
            self.build_phase_shifts_for_srocomp(k)
        # Compute desired signal chunk estimate
        self.get_desired_signal(k)
        # Update iteration index
        self.i[k] += 1


    def evaluate_tstart_for_metrics_computation(self, k, t):
        """
        Evaluates the start instants for metrics computations.

        Parameters
        ----------
        k : int
            Node index.
        t : float
            Current time instant [s].
        """ 
        
        def _eval(s, nUps):
            """Helper function."""
            tstart = None
            if nUps >= s.minFiltUpdatesForMetrics:
                if s.compensateSROs and s.estimateSROs == 'CohDrift':
                    # Make sure SRO compensation has started
                    if nUps > s.cohDrift.startAfterNups:
                        tstart = t
                else:
                    tstart = t
            return tstart
        
        if self.tStartForMetrics[k] is None:
            self.tStartForMetrics[k] = _eval(self, self.nInternalFilterUps[k])
        if self.computeCentralised and self.tStartForMetricsCentr[k] is None:
            self.tStartForMetricsCentr[k] = _eval(self, self.nCentrFilterUps[k])
        if self.computeLocal and self.tStartForMetricsLocal[k] is None:
            self.tStartForMetricsLocal[k] = _eval(self, self.nLocalFilterUps[k])


    def check_covariance_matrices(self, k):
        """
        Checks that the number of rank-1 covariance matrix estimate updates
        done in `spatial_covariance_matrix_update()` is at least equal to
        the dimension of the corresponding covariance matrix (ensuring full-
        rank property).

        Parameters
        ----------
        k : int
            Node index.
        """
        if not self.startUpdates[k]:
            if self.numUpdatesRyy[k] > self.Ryytilde[k].shape[-1] and \
                self.numUpdatesRnn[k] > self.Ryytilde[k].shape[-1]:
                self.startUpdates[k] = True
        if self.computeCentralised and not self.startUpdatesCentr[k]:
            if self.numUpdatesRyy[k] > self.Ryycentr[k].shape[-1] and \
                self.numUpdatesRnn[k] > self.Ryycentr[k].shape[-1]:
                self.startUpdatesCentr[k] = True
        if self.computeLocal and not self.startUpdatesLocal[k]:
            if self.numUpdatesRyy[k] > self.Ryylocal[k].shape[-1] and \
                self.numUpdatesRnn[k] > self.Ryylocal[k].shape[-1]:
                self.startUpdatesLocal[k] = True


    def build_ycentr(self, tCurr, fs, k):
        """
        Build STFT-domain centralised observation vector.
        """
        # Extract current local data chunk
        yCentrCurr, _, _ = base.local_chunk_for_update(
            self.yinStacked,
            tCurr,
            fs,
            bd=self.broadcastType,
            Ndft=self.DFTsize,
            Ns=self.Ns
        )
        self.yCentr[k][:, self.i[k], :] = yCentrCurr
        # Go to frequency domain
        yHatCentrCurr = 1 / self.normFactWOLA * np.fft.fft(
            self.yCentr[k][:, self.i[k], :] *\
                self.winWOLAanalysis[:, np.newaxis],
            self.DFTsize,
            axis=0
        )
        # Keep only positive frequencies
        self.yHatCentr[k][:, self.i[k], :] =\
            yHatCentrCurr[:self.nPosFreqs, :]


    def update_external_filters(self, k, t):
        """
        Update external filters for relaxed filter update.
        To be used when using simultaneous or asynchronous node-updating.
        When using sequential node-updating, do not differential between
        internal (`self.wTilde`) and external filters. 
        
        Parameters
        ----------
        k : int
            Receiving node index.
        t : float
            Current time instant [s].
        """

        # Simultaneous or asynchronous node-updating
        if self.nodeUpdating in ['sim', 'asy']:

            self.wTildeExt[k] = self.expAvgBeta[k] * self.wTildeExt[k] +\
                (1 - self.expAvgBeta[k]) *  self.wTildeExtTarget[k]
            # Update targets
            if t - self.lastExtFiltUp[k] >= self.timeBtwExternalFiltUpdates:
                self.wTildeExtTarget[k] = (1 - self.alphaExternalFilters) *\
                    self.wTildeExtTarget[k] + self.alphaExternalFilters *\
                    self.wTilde[k][:, self.i[k] + 1, :self.nLocalMic[k]]
                # Update last external filter update instant [s]
                self.lastExtFiltUp[k] = t
                if self.printout_externalFilterUpdate:    # inform user
                    print(f't={np.round(t, 3)}s -- UPDATING EXTERNAL FILTERS for node {k+1} (scheduled every [at least] {self.timeBtwExternalFiltUpdates}s)')

        # Sequential node-updating
        elif self.nodeUpdating == 'seq':

            self.wTildeExt[k] =\
                self.wTilde[k][:, self.i[k] + 1, :self.nLocalMic[k]]



    def process_incoming_signals_buffers(self, k, t):
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
        """

        # Useful renaming
        Ndft = self.DFTsize
        Ns = self.Ns
        Lbc = self.broadcastLength

        # Initialize compressed signal matrix
        # ($\mathbf{z}_{-k}$ in [1]'s notation)
        zk = np.empty((self.DFTsize, 0), dtype=float)

        # Initialise flags vector (overflow: >0; underflow: <0; or none: ==0)
        bufferFlags = np.zeros(len(self.neighbors[k]))

        for idxq in range(len(self.neighbors[k])):
            
            Bq = len(self.zBuffer[k][idxq])  # buffer size for neighbour `q`

            # Time-domain chunks broadcasting
            if self.broadcastType == 'wholeChunk':
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
                    
            elif self.broadcastType == 'fewSamples':

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

    
    def build_ytilde(self, tCurr, fs, k):
        """
        
        Parameters
        ----------
        tCurr : float
            Current time instant [s].
        fs : float
            Node `k`'s sampling frequency [Hz].
        k : int
            Receiving node index.
        dv : DANSEvariables object
            DANSE variables to be updated.
        """

        # Get data
        yk = self.yin[k]

        # Extract current local data chunk
        yLocalCurr, self.idxBegChunk, self.idxEndChunk =\
            base.local_chunk_for_update(
                yk,
                tCurr,
                fs,
                bd=self.broadcastType,
                Ndft=self.DFTsize,
                Ns=self.Ns
            )

        # Compute VAD
        VADinFrame = self.fullVAD[k][
            np.amax([self.idxBegChunk, 0]):self.idxEndChunk
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
        yTildeHatCurr = 1 / self.normFactWOLA * np.fft.fft(
            self.yTilde[k][:, self.i[k], :] *\
                self.winWOLAanalysis[:, np.newaxis],
            self.DFTsize,
            axis=0
        )
        # Keep only positive frequencies
        self.yTildeHat[k][:, self.i[k], :] = yTildeHatCurr[:self.nPosFreqs, :]


    def compensate_sros(self, k, t):
        """
        Compensate for SROs based on estimates, accounting for full-sample 
        drift flags.

        Parameters
        ----------
        k : int
            Receiving node index.
        t : float
            Current time instant [s].

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
                    self.bufferFlags[k][self.i[k], q] * self.broadcastLength
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
        if self.compensateSROs:
            # Complete phase shift factors
            self.phaseShiftFactors[k] += extraPhaseShiftFactor
            if k == 0:  # Save for plotting
                self.phaseShiftFactorThroughTime[self.i[k]:] =\
                    self.phaseShiftFactors[k][self.nLocalMic[k] + q]
            # Apply phase shift factors
            self.yTildeHat[k][:, self.i[k], :] *=\
                np.exp(-1 * 1j * 2 * np.pi / self.DFTsize *\
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

        def _upd(Ryy, Rnn, yyH,
            vad=self.oVADframes[self.i[k]], beta=self.expAvgBeta[k]
        ):
            """Quick helper function to perform exponential averaging."""
            if vad:
                Ryy = beta * Ryy + (1 - beta) * yyH
            else:
                Rnn = beta * Rnn + (1 - beta) * yyH
            return Ryy, Rnn

        # Useful renaming
        y = self.yTildeHat[k][:, self.i[k], :]
        yyH = np.einsum('ij,ik->ijk', y, y.conj())  # outer product
        self.Ryytilde[k], self.Rnntilde[k] = _upd(
            self.Ryytilde[k], self.Rnntilde[k], yyH
        )  # update
        self.yyH[k][self.i[k], :, :, :] = yyH

        
        # Consider centralised / local estimation(s)
        if self.computeLocal:
            y = self.yHatLocal[k][:, self.i[k], :]
            yyH = np.einsum('ij,ik->ijk', y, y.conj())
            self.Ryylocal[k], self.Rnnlocal[k] = _upd(
                self.Ryylocal[k], self.Rnnlocal[k], yyH
            )  # update local
        if self.computeCentralised:
            y = self.yHatCentr[k][:, self.i[k], :]
            yyH = np.einsum('ij,ik->ijk', y, y.conj())
            self.Ryycentr[k], self.Rnncentr[k] = _upd(
                self.Ryycentr[k], self.Rnncentr[k], yyH
            )  # update centralised



    def perform_update(self, k):
        """
        Filter update for DANSE, `for`-loop free.
        GEVD or no GEVD, depending on `self.performGEVD`.
        
        Parameters
        ----------
        k : int
            Node index.
        """

        def _update_w(Ryy, Rnn, refSensor):
            """Helper function for regular MWF-like
            DANSE filter update."""
            # Reference sensor selection vector
            Evect = np.zeros((Ryy.shape[-1],))
            Evect[refSensor] = 1

            # Cross-correlation matrix update 
            ryd = np.matmul(Ryy - Rnn, Evect)
            # Update node-specific parameters of node k
            Ryyinv = np.linalg.inv(Ryy)
            w = np.matmul(Ryyinv, ryd[:,:,np.newaxis])
            return w[:, :, 0]  # get rid of singleton dimension

        def _update_w_gevd(Ryy, Rnn, refSensor):
            """Helper function for GEVD-based MWF-like
            DANSE filter update."""
            n = Ryy.shape[-1]
            nFreqs = Ryy.shape[0]
            # Reference sensor selection vector 
            Evect = np.zeros((n,))
            Evect[refSensor] = 1

            sigma = np.zeros((nFreqs, n))
            Xmat = np.zeros((nFreqs, n, n), dtype=complex)

            # t0 = time.perf_counter()
            for kappa in range(nFreqs):
                # Perform generalized eigenvalue decomposition 
                # -- as of 2022/02/17: scipy.linalg.eigh()
                # seemingly cannot be jitted nor vectorized.
                sigmacurr, Xmatcurr = sla.eigh(
                    Ryy[kappa, :, :],
                    Rnn[kappa, :, :],
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
            for ii in range(self.GEVDrank):
                Dmat[:, ii, ii] = np.squeeze(1 - 1/sigma[:, ii])
            # LMMSE weights
            Qhermitian = np.transpose(Qmat.conj(), axes=[0,2,1])
            w = np.matmul(np.matmul(np.matmul(Xmat, Dmat), Qhermitian), Evect)
            return w

        # Select appropriate update function
        if self.performGEVD:
            update_fcn = _update_w_gevd
        else:
            update_fcn = _update_w

        if not self.bypassUpdates:
            if self.startUpdates[k]:
                # Update DANSE filter
                self.wTilde[k][:, self.i[k] + 1, :] = update_fcn(
                    self.Ryytilde[k],
                    self.Rnntilde[k],
                    self.referenceSensor
                )
                self.nInternalFilterUps[k] += 1  
            # Update centralised filter
            if self.computeCentralised and self.startUpdatesCentr[k]:
                self.wCentr[k][:, self.i[k] + 1, :] = update_fcn(
                    self.Ryycentr[k],
                    self.Rnncentr[k],
                    self.referenceSensor
                )
                self.nCentrFilterUps[k] += 1  
            # Update local filter
            if self.computeLocal and self.startUpdatesLocal[k]:
                self.wLocal[k][:, self.i[k] + 1, :] = update_fcn(
                    self.Ryylocal[k],
                    self.Rnnlocal[k],
                    self.referenceSensor
                )
                self.nLocalFilterUps[k] += 1  

    


    def update_sro_estimates(self, k):
        """
        Update SRO estimates.
        
        Parameters
        ----------
        k : int
            Node index.
        """
        # Useful variables (compact coding)
        nNeighs = len(self.neighbors[k])
        iter = self.i[k]
        bufferFlagPos = self.broadcastLength * np.sum(
            self.bufferFlags[k][:(iter + 1), :],
            axis=0
        )
        bufferFlagPri = self.broadcastLength * np.sum(
            self.bufferFlags[k][:(iter - self.cohDrift.segLength + 1), :],
            axis=0
        )
        
        # DANSE filter update indices
        # corresponding to "Filter-shift" SRO estimate updates.
        cohDriftSROupdateIndices = np.arange(
            start=self.cohDrift.startAfterNups + self.cohDrift.estEvery,
            stop=self.nIter,
            step=self.cohDrift.estEvery
        )
        
        # Init arrays
        sroOut = np.zeros(nNeighs)
        if self.estimateSROs == 'CohDrift':
            
            ld = self.cohDrift.segLength

            if iter in cohDriftSROupdateIndices:

                flagFirstSROEstimate = False
                if iter == np.amin(cohDriftSROupdateIndices):
                    # Let `cohdrift_sro_estimation()` know that
                    # this is the 1st SRO estimation round.
                    flagFirstSROEstimate = True

                # Residuals method
                for q in range(nNeighs):
                    # index of compressed signal from node `q` inside `yyH`
                    idxq = self.nLocalMic[k] + q     
                    if self.cohDrift.loop == 'closed':
                        # Use SRO-compensated correlation matrix entries
                        # (closed-loop SRO est. + comp.).

                        # A posteriori coherence
                        cohPosteriori = (self.yyH[k][iter, :, 0, idxq]
                            / np.sqrt(self.yyH[k][iter, :, 0, 0] *\
                                self.yyH[k][iter, :, idxq, idxq]))
                        # A priori coherence
                        cohPriori = (self.yyH[k][iter - ld, :, 0, idxq]
                            / np.sqrt(self.yyH[k][iter - ld, :, 0, 0] *\
                                self.yyH[k][iter - ld, :, idxq, idxq]))
                        
                        # Set buffer flags to 0
                        bufferFlagPri = np.zeros_like(bufferFlagPri)
                        bufferFlagPos = np.zeros_like(bufferFlagPos)

                    elif self.cohDrift.loop == 'open':
                        # Use SRO-_un_compensated correlation matrix entries
                        # (open-loop SRO est. + comp.).

                        # A posteriori coherence
                        cohPosteriori = (self.yyHuncomp[k][iter, :, 0, idxq]
                            / np.sqrt(self.yyHuncomp[k][iter, :, 0, 0] *\
                                self.yyHuncomp[k][iter, :, idxq, idxq]))
                        # A priori coherence
                        cohPriori = (self.yyHuncomp[k][iter - ld, :, 0, idxq]
                            / np.sqrt(self.yyHuncomp[k][iter - ld, :, 0, 0] *\
                                self.yyHuncomp[k][iter - ld, :, idxq, idxq]))

                    # Perform SRO estimation via coherence-drift method
                    sroRes, apr = sros.cohdrift_sro_estimation(
                        wPos=cohPosteriori,
                        wPri=cohPriori,
                        avgResProd=self.avgProdResiduals[k][:, q],
                        Ns=self.Ns,
                        ld=ld,
                        method=self.cohDrift.estimationMethod,
                        alpha=self.cohDrift.alpha,
                        flagFirstSROEstimate=flagFirstSROEstimate,
                        bufferFlagPri=bufferFlagPri[q],
                        bufferFlagPos=bufferFlagPos[q]
                    )
                
                    sroOut[q] = sroRes
                    self.avgProdResiduals[k][:, q] = apr

        elif self.estimateSROs == 'Oracle':
            # No data-based dynamic SRO estimation: use oracle knowledge
            sroOut = (self.SROsppm[self.neighbors[k]] -\
                self.SROsppm[k]) * 1e-6

        # Save SRO (residuals)
        self.SROsResiduals[k][iter, :] = sroOut

    
    def build_phase_shifts_for_srocomp(self, k):
        """
        Computed appropriate phase shift factors for next SRO compensation.
        
        Parameters
        ----------
        k : int
            Node index.
        """

        for q in range(len(self.neighbors[k])):
            if self.estimateSROs == 'CohDrift':
                if self.cohDrift.loop == 'closed':
                    # Increment estimate using SRO residual
                    self.SROsEstimates[k][self.i[k], q] +=\
                        self.SROsResiduals[k][self.i[k], q] /\
                        (1 + self.SROsResiduals[k][self.i[k], q]) *\
                        self.cohDrift.alphaEps
                elif self.cohDrift.loop == 'open':
                    # Use SRO "residual" as estimates
                    self.SROsEstimates[k][self.i[k], q] =\
                        self.SROsResiduals[k][self.i[k], q] /\
                        (1 + self.SROsResiduals[k][self.i[k], q])
            # Increment phase shift factor recursively.
            # (valid directly for oracle SRO "estimation")
            self.phaseShiftFactors[k][self.nLocalMic[k] + q] -=\
                self.SROsEstimates[k][self.i[k], q] * self.Ns 

    
    def get_desired_signal(self, k):
        """
        Compute chunk of desired signal from DANSE freq.-domain filters
        and freq.-domain observation vector y_tilde.

        Parameters
        ----------
        k : int
            Node index.
        """

        # Build desired signal estimate
        dChunk, dhatCurr = base.get_desired_sig_chunk(
            self.desSigProcessingType,
            w=self.wTilde[k][:, self.i[k] + 1, :],
            y=self.yTildeHat[k][:, self.i[k], :],
            win=self.winWOLAsynthesis,
            dChunk=self.d[self.idxBegChunk:self.idxEndChunk, k],
            yTD=self.yTilde[k][:, self.i[k], :self.nLocalMic[k]],
            normFactWOLA=self.normFactWOLA,
            Ns=self.Ns,
        )
        self.dhat[:, self.i[k], k] = dhatCurr  # STFT-domain
        # Time-domain
        if self.desSigProcessingType == 'wola':
            self.d[self.idxBegChunk:self.idxEndChunk, k] = dChunk
        elif self.desSigProcessingType == 'conv':
            self.d[self.idxEndChunk - self.Ns:self.idxEndChunk, k] = dChunk

        if self.computeCentralised:
            # Build centralised desired signal estimate
            dChunk, dhatCurr = base.get_desired_sig_chunk(
                self.desSigProcessingType,
                w=self.wCentr[k][:, self.i[k] + 1, :],
                y=self.yHatCentr[k][:, self.i[k], :],
                win=self.winWOLAsynthesis,
                dChunk=self.dCentr[self.idxBegChunk:self.idxEndChunk, k],
                yTD=self.yCentr[k][:, self.i[k], :self.nLocalMic[k]],
                normFactWOLA=self.normFactWOLA,
                Ns=self.Ns,
            )
            self.dHatCentr[:, self.i[k], k] = dhatCurr  # STFT-domain
            # Time-domain
            if self.desSigProcessingType == 'wola':
                self.dCentr[self.idxBegChunk:self.idxEndChunk, k] = dChunk
            elif self.desSigProcessingType == 'conv':
                self.dCentr[self.idxEndChunk -\
                    self.Ns:self.idxEndChunk, k] = dChunk
        
        if self.computeLocal:
            # Build local desired signal estimate
            dChunk, dhatCurr = base.get_desired_sig_chunk(
                self.desSigProcessingType,
                w=self.wLocal[k][:, self.i[k] + 1, :],
                y=self.yHatLocal[k][:, self.i[k], :],
                win=self.winWOLAsynthesis,
                dChunk=self.dLocal[self.idxBegChunk:self.idxEndChunk, k],
                yTD=self.yLocal[k][:, self.i[k], :self.nLocalMic[k]],
                normFactWOLA=self.normFactWOLA,
                Ns=self.Ns,
            )
            self.dHatLocal[:, self.i[k], k] = dhatCurr  # STFT-domain
            # Time-domain
            if self.desSigProcessingType == 'wola':
                self.dLocal[self.idxBegChunk:self.idxEndChunk, k] = dChunk
            elif self.desSigProcessingType == 'conv':
                self.dLocal[self.idxEndChunk -\
                    self.Ns:self.idxEndChunk, k] = dChunk

