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

        # Initialize filters for each node
        self.wTilde = []
        for k in range(self.nNodes):
            filtInit = np.zeros((
                wasn[k].nSensors + len(wasn[k].neighborsIdx),
                self.maxBatchUpdates + 1
            ))
            filtInit[0, :] = 1
            self.wTilde.append(filtInit)

        # Broadcast signals
        self.z = [s[:, 0] for s in self.yLocal]  # speech + noise
        self.z_s = [s[:, 0] for s in self.yLocal_s]  # speech
        self.z_n = [s[:, 0] for s in self.yLocal_n]  # noise

        self.d = [s[:, 0] for s in self.yLocal]  # desired signal estimates
        self.i = [0 for _ in range(self.nNodes)]  # DANSE update indices
        
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
        # Get y tilde
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

        if self.performGEVD:
            raise NotImplementedError
        else:
            # Reference sensor selection vector
            Evect = np.zeros(Ryy.shape[-1])
            Evect[self.referenceSensor] = 1
            # Cross-correlation matrix update 
            ryd = np.matmul(Ryy - Rnn, Evect)
            # Update node-specific parameters of node k
            Ryyinv = np.linalg.inv(Ryy)
            w = np.matmul(Ryyinv, ryd[:, np.newaxis])
            self.wTilde[k][:, self.i[k] + 1] = w[:, 0]  # get rid of singleton dimension
        
        # Update DANSE update index
        self.i[k] += 1
    
    def batch_estimate(self, k):
        """
        Estimate the desired signal in batch mode.
        """
        # Estimate the desired signal
        self.d[k] = np.dot(self.wTilde[k][:, self.i[k]], self.yTilde[k].T)