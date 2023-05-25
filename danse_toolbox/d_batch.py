from danse_toolbox.d_classes import *

@dataclass
class BatchDANSEvariables(DANSEvariables):
    """
    Main DANSE class for batch simulations.
    Stores all relevant variables and core functions on those variables.
    """
    def init(self):
        self.startUpdates = [True for _ in range(self.nNodes)]
        # Get MMSE costs for centralised and local estimates
        self.mmseCostLocal = [np.mean(
            np.abs(self.cleanSpeechSignalsAtNodes[k][:, self.referenceSensor] -\
                self.dLocal[:, k])**2
        ) for k in range(self.nNodes)]
        self.mmseCostCentr = [np.mean(
            np.abs(self.cleanSpeechSignalsAtNodes[k][:, self.referenceSensor] -\
                self.dCentr[:, k])**2
        ) for k in range(self.nNodes)]
        self.mmseCost = np.full((self.maxBatchUpdates, self.nNodes), None)  # <-- to be updated

    def get_centralized_and_local_estimates(self):
        """
        Get the centralized and local estimates of the desired signal.
        """

        # Select appropriate update function
        if self.performGEVD:
            filter_update_fcn = update_w_gevd
            rank = self.GEVDrank
        else:
            filter_update_fcn = update_w
            rank = 1  # <-- arbitrary, not used in this case

        if self.computeCentralised:
            for k in range(self.nNodes):
                self.wCentr[k][:, self.i[k] + 1, :] = filter_update_fcn(
                    self.Ryycentr[k],
                    self.Rnncentr[k],
                    refSensorIdx=int(
                        np.sum(self.nSensorPerNode[:k]) + self.referenceSensor
                    ),
                    rank=rank
                )
                # Estimate the centralised desired signal via linear combination
                self.dHatCentr[:, :, k] = np.einsum(
                    'ik,ijk->ij',
                    self.wCentr[k][:, self.i[k] + 1, :].conj(), 
                    self.yCentrBatch[:, :-1, :]  # <-- exclude last frame FIXME: why?
                )                
                self.dHatCentr_s[:, :, k] = np.einsum(
                    'ik,ijk->ij',
                    self.wCentr[k][:, self.i[k] + 1, :].conj(), 
                    self.yCentrBatch_s[:, :-1, :]  # <-- exclude last frame FIXME: why?
                )
                self.dHatCentr_n[:, :, k] = np.einsum(
                    'ik,ijk->ij',
                    self.wCentr[k][:, self.i[k] + 1, :].conj(), 
                    self.yCentrBatch_n[:, :-1, :]  # <-- exclude last frame FIXME: why?
                )
                # Convert back to time domain
                self.dCentr[:, k] =self.get_istft(self.dHatCentr[:, :, k], k)\
                    / np.sum(self.winWOLAanalysis)
                self.dCentr_s[:, k] = self.get_istft(self.dHatCentr_s[:, :, k], k)\
                    / np.sum(self.winWOLAanalysis)
                self.dCentr_n[:, k] = self.get_istft(self.dHatCentr_n[:, :, k], k)\
                    / np.sum(self.winWOLAanalysis)

        if self.computeLocal:
            for k in range(self.nNodes):
                self.wLocal[k][:, self.i[k] + 1, :] = filter_update_fcn(
                    self.Ryylocal[k],
                    self.Rnnlocal[k],
                    refSensorIdx=self.referenceSensor,
                    rank=rank
                )
                # Estimate the local desired signal via linear combination
                self.dHatLocal[:, :, k] = np.einsum(
                    'ik,ijk->ij',
                    self.wLocal[k][:, self.i[k] + 1, :].conj(), 
                    self.yinSTFT[k][:, :-1, :]  # <-- exclude last frame FIXME: why?
                )
                self.dHatLocal_s[:, :, k] = np.einsum(
                    'ik,ijk->ij',
                    self.wLocal[k][:, self.i[k] + 1, :].conj(), 
                    self.yinSTFT_s[k][:, :-1, :]  # <-- exclude last frame FIXME: why?
                )
                self.dHatLocal_n[:, :, k] = np.einsum(
                    'ik,ijk->ij',
                    self.wLocal[k][:, self.i[k] + 1, :].conj(), 
                    self.yinSTFT_n[k][:, :-1, :]  # <-- exclude last frame FIXME: why?
                )
                # Convert back to time domain
                self.dLocal[:, k] = self.get_istft(self.dHatLocal[:, :, k], k)\
                    / np.sum(self.winWOLAanalysis)
                self.dLocal_s[:, k] = self.get_istft(self.dHatLocal_s[:, :, k], k)\
                    / np.sum(self.winWOLAanalysis)
                self.dLocal_n[:, k] = self.get_istft(self.dHatLocal_n[:, :, k], k)\
                    / np.sum(self.winWOLAanalysis)


    def batch_update_danse_covmats(self, k):
        """
        Update the broadcast signals in batch mode, using the latest filters.
        """
        self.Ryytilde[k], self.Rnntilde[k] = update_covmats_batch(
            self.get_y_tilde_batch(k),
            self.oVADframes[k]
        )
    
    def batch_estimate(self, k):
        """
        Estimate the desired signal in batch mode.
        """
        # Get ytilde batch
        yTildeBatch, yTildeBatch_s, yTildeBatch_n = self.get_y_tilde_batch(
            k,
            computeSpeechAndNoiseOnly=True
        )

        # Estimate the desired signal via linear combination
        self.dhat[:, :, k] = np.einsum(
            'ik,ijk->ij',
            self.wTilde[k][:, self.i[k] + 1, :].conj(), 
            yTildeBatch[:, :-1, :]  # <-- exclude last frame FIXME: why?
        )
        self.dhat_s[:, :, k] = np.einsum(
            'ik,ijk->ij',
            self.wTilde[k][:, self.i[k] + 1, :].conj(), 
            yTildeBatch_s[:, :-1, :]  # <-- exclude last frame FIXME: why?
        )
        self.dhat_n[:, :, k] = np.einsum(
            'ik,ijk->ij',
            self.wTilde[k][:, self.i[k] + 1, :].conj(), 
            yTildeBatch_n[:, :-1, :]  # <-- exclude last frame FIXME: why?
        )
        # Convert back to time domain
        self.d[:, k] = self.get_istft(self.dhat[:, :, k], k)\
                    / np.sum(self.winWOLAanalysis)
        self.d_s[:, k] = self.get_istft(self.dhat_s[:, :, k], k)\
                    / np.sum(self.winWOLAanalysis)
        self.d_n[:, k] = self.get_istft(self.dhat_n[:, :, k], k)\
                    / np.sum(self.winWOLAanalysis)

    def get_istft(self, xSTFT, k):
        """
        Convert a STFT signal to time domain.

        Parameters
        ----------
        xSTFT : ndarray, shape (F, T)
            The STFT signal to convert.
        k : int
            The node index.
            
        Returns
        -------
        x : ndarray, shape (T,)
            The time domain signal.
        """
        # Convert back to time domain
        x = base.get_istft(
            x=xSTFT,
            fs=self.fs[k],
            win=self.winWOLAanalysis,
            ovlp=1 - self.Ns / self.DFTsize,
            boundary=None  # no padding to center frames at t=0s!
        )[0]
        # Pad with zeros to match length of original signal
        if len(x) < len(self.d[:, k]):
            x = np.pad(
                x,
                (0, len(self.d[:, k]) - len(x)),
                mode='constant'
            )
        return x
    
    def get_mmse_cost(self, k):
        """
        Compute the DANSE MMSE cost function for the given node.

        Parameters
        ----------
        k : int
            The node index.
        """
        # True desired signal
        targetSig = self.cleanSpeechSignalsAtNodes[k][:, self.referenceSensor]
        self.mmseCost[self.i[k], k] = np.mean(
            np.abs(targetSig - self.d[:, k])**2
        )