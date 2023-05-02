from danse_toolbox.d_classes import *

@dataclass
class BatchDANSEvariables(base.DANSEparameters):
    """
    Main DANSE class for batch simulations.
    Stores all relevant variables and core functions on those variables.
    """
    def import_params(self, p: base.DANSEparameters):
        self.__dict__.update(p.__dict__)

    def init_from_wasn(self, wasn: list[Node]):
        """
        Initialize `BatchDANSEvariables` object based on `wasn`
        list of `Node` objects.
        """
        self.nNodes = len(wasn)  # number of nodes in WASN
        # Check for TI-DANSE case
        self.tidanseFlag = not check_if_fully_connected(wasn)
        
        t = np.zeros((len(wasn[0].timeStamps), self.nNodes))  # time stamps
        for k in range(self.nNodes):
            t[:, k] = wasn[k].timeStamps
        self.timeInstants = t

        self.yLocal = [node.data for node in wasn]  # local time-domain signals
        self.yLocal_s = [node.cleanspeech for node in wasn]  # speech only
        self.yLocal_n = [node.cleannoise for node in wasn]  # noise only
        if self.computeCentralised:
            self.yCentr = np.concatenate(
                tuple([y for y in self.yLocal]), axis=1
            )
            self.yCentr_s = np.concatenate(
                tuple([y for y in self.yLocal_s]), axis=1
            )
            self.yCentr_n = np.concatenate(
                tuple([y for y in self.yLocal_n]), axis=1
            )

        # Initialize filters for each node
        self.wTilde = []
        self.wCentr = []
        for k in range(self.nNodes):
            # DANSE filters
            filtInit = np.zeros((
                wasn[k].nSensors + len(wasn[k].neighborsIdx),
                self.maxBatchUpdates + 1
            ))
            filtInit[0, :] = 1
            self.wTilde.append(filtInit)
            # Centralized filters
            filtInit = np.zeros((
                sum([wasn[k].nSensors for k in range(self.nNodes)]),
                self.maxBatchUpdates + 1
            ))
            filtInit[0, :] = 1
            self.wCentr.append(filtInit)
        # Local filters
        self.wLocal = [
            self.wTilde[k][:self.nSensorPerNode[k], :] for k in range(self.nNodes)
        ]

        # Broadcast signals
        self.z = [s[:, 0] for s in self.yLocal]  # speech + noise
        self.z_s = [s[:, 0] for s in self.yLocal_s]  # speech
        self.z_n = [s[:, 0] for s in self.yLocal_n]  # noise

        # Desired signal estimates
        self.dHat = []
        for k in range(self.nNodes):
            self.dHat.append(
                np.zeros((self.yLocal[k].shape[0], self.maxBatchUpdates + 1))
            )
        self.dHatCentr = copy.deepcopy(self.dHat)
        self.dHatLocal = copy.deepcopy(self.dHat)

        # True desired signals
        self.d = [wasn[k].cleanspeechRefSensor for k in range(self.nNodes)]

        # DANSE update indices
        self.i = [0 for _ in range(self.nNodes)]

        self.centralisedComputed = [False for _ in range(self.nNodes)]  # flag for centralised filters
        self.localComputed = [False for _ in range(self.nNodes)]  # flag for local filters
        
        return self

    def batch_update_broadcast_signals(self):
        """
        Update the broadcast signals in batch mode, using the latest filters.
        """
        for k in range(self.nNodes):
            self.z[k] = np.dot(
                self.wTilde[k][:self.nSensorPerNode[k], self.i[k]],
                self.yLocal[k].T
            )
            self.z_s[k] = np.dot(
                self.wTilde[k][:self.nSensorPerNode[k], self.i[k]],
                self.yLocal_s[k].T
            )
            self.z_n[k] = np.dot(
                self.wTilde[k][:self.nSensorPerNode[k], self.i[k]],
                self.yLocal_n[k].T
            )
    
    def batch_update(self, k):
        """
        Update the DANSE variables in batch mode.
        """
        # Get y tilde (DANSE)
        self.yTilde = copy.deepcopy(self.yLocal)  # y tilde signals
        self.yTilde_s = copy.deepcopy(self.yLocal_s)  # y tilde signals (speech)
        self.yTilde_n = copy.deepcopy(self.yLocal_n)  # y tilde signals (noise)
        for kTmp in range(self.nNodes):
            if self.tidanseFlag:
                raise NotImplementedError
            else:
                for q in range(self.nNodes):
                    if q != kTmp:
                        self.yTilde[kTmp] = np.concatenate(
                            (self.yTilde[kTmp], self.z[q][:, np.newaxis]),
                            axis=1
                        )
                        self.yTilde_s[kTmp] = np.concatenate(
                            (self.yTilde_s[kTmp], self.z_s[q][:, np.newaxis]),
                            axis=1
                        )
                        self.yTilde_n[kTmp] = np.concatenate(
                            (self.yTilde_n[kTmp], self.z_n[q][:, np.newaxis]),
                            axis=1
                        )
        
        # Get spatial correlation matrices (batch mode)
        Ryy = np.dot(self.yTilde_s[k].T.conj(), self.yTilde_s[k])
        Rnn = np.dot(self.yTilde_n[k].T.conj(), self.yTilde_n[k])

        # Update DANSE filters
        self.wTilde[k][:, self.i[k] + 1] = self.compute_filters(
            Ryy, Rnn, self.referenceSensor
        )
        
        # Update centralized filters
        if not self.centralisedComputed[k] and self.computeCentralised:
            RyyCentr = np.dot(self.yCentr_s.T.conj(), self.yCentr_s)
            RnnCentr = np.dot(self.yCentr_n.T.conj(), self.yCentr_n)
            self.wCentr[k][:, self.i[k] + 1] = self.compute_filters(
                RyyCentr, RnnCentr, int(np.sum(self.nSensorPerNode[:k]))  # <-- !!
            )
            self.centralisedComputed[k] = True
        elif self.centralisedComputed[k] and self.computeCentralised:
            # No need to recompute the centralised filters if they have already
            # been computed once.
            self.wCentr[k][:, self.i[k] + 1] = self.wCentr[k][:, self.i[k]]

        # Update local filters
        if not self.localComputed[k] and self.computeLocal:
            RyyLocal = np.dot(self.yLocal_s[k].T.conj(), self.yLocal_s[k])
            RnnLocal = np.dot(self.yLocal_n[k].T.conj(), self.yLocal_n[k])
            self.wLocal[k][:, self.i[k] + 1] = self.compute_filters(
                RyyLocal, RnnLocal, self.referenceSensor
            )
            self.localComputed[k] = True
        elif self.localComputed[k] and self.computeLocal:
            # No need to recompute the local filters if they have already
            # been computed once.
            self.wLocal[k][:, self.i[k] + 1] = self.wLocal[k][:, self.i[k]]

    def compute_filters(self, Ryy, Rnn, referenceSensor):
        if self.performGEVD:
            raise NotImplementedError
        else:
            # Reference sensor selection vector
            Evect = np.zeros(Ryy.shape[-1])
            Evect[referenceSensor] = 1
            # Cross-correlation matrix update 
            ryd = np.matmul(Ryy - Rnn, Evect)
            # Update node-specific parameters of node k
            Ryyinv = np.linalg.inv(Ryy)
            w = np.matmul(Ryyinv, ryd[:, np.newaxis])
            return w[:, 0]  # get rid of singleton dimension
    
    def batch_estimate(self, k):
        """
        Estimate the desired signal in batch mode.
        """
        # Estimate the desired signal
        self.dHat[k][:, self.i[k] + 1] = np.dot(
            self.wTilde[k][:, self.i[k] + 1], self.yTilde[k].T
        )
        # Estimate the desired signal (centralised)
        if self.computeCentralised:
            self.dHatCentr[k][:, self.i[k] + 1] = np.dot(
                self.wCentr[k][:, self.i[k] + 1], self.yCentr.T
            )
        # Estimate the desired signal (local)
        if self.computeLocal:
            self.dHatLocal[k][:, self.i[k] + 1] = np.dot(
                self.wLocal[k][:, self.i[k] + 1], self.yLocal[k].T
            )   