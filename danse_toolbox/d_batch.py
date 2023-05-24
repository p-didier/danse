from danse_toolbox.d_classes import *

@dataclass
class BatchDANSEvariables(DANSEvariables):
    """
    Main DANSE class for batch simulations.
    Stores all relevant variables and core functions on those variables.
    """
    # def init(self):
    #     self.dhatAll = [np.zeros_like(self.dhat) for _ in range(self.nIter)]

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
                # TODO: Convert back to time domain
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
                # TODO: Convert back to time domain


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
        yTildeBatch = self.get_y_tilde_batch(k)

        # Estimate the desired signal via linear combination
        self.dhat[:, :, k] = np.einsum(
            'ik,ijk->ij',
            self.wTilde[k][:, self.i[k] + 1, :].conj(), 
            yTildeBatch[:, :-1, :]  # <-- exclude last frame FIXME: why?
        )
        # TODO: Convert back to time domain