import numpy as np
from danse_toolbox.d_classes import *

from pathlib import Path, PurePath
import sys

# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}/danse/dxcpphat')
sys.path.append(f'{pathToRoot}/_third_parties')
from paderwasn.synchronization.time_shift_estimation import max_time_lag_search
# from paderwasn.paderwasn.synchronization.time_shift_estimation import max_time_lag_search
import sro_estimation as dxcp


def cohdrift_sro_estimation(
        wPos: np.ndarray,
        wPri: np.ndarray,
        avgResProd,
        Ns,
        ld,
        alpha=0.95,
        method='gs',
        flagFirstSROEstimate=False,
        bufferFlagPos=0,
        bufferFlagPri=0):
    """
    Estimates a(n) (residual) SRO using a coherence drift technique.
    
    Parameters
    ----------
    wPos : [N x 1] np.ndarray (complex)
        A posteriori (iteration `i + 1`) value for every frequency bin
    wPri : [N x 1] np.ndarray (complex)
        A priori (iteration `i`) value for every frequency bin
    avg_res_prod : [2*(N-1) x 1] np.ndarray (complex)
        Exponentially averaged complex conjugate product of `wPos` and `wPri`
    Ns : int
        Number of new samples at each new STFT frame, counting overlap (`Ns=N*(1-O)`, where `O` is the amount of overlap [/100%])
    ld : int
        Number of STFT frames separating `wPos` from `wPri`.
    alpha : float
        Exponential averaging constant (DWACD method: .95).
    method : str
        Method to use to retrieve SRO once the exponentially averaged product has been computed.
    flagFirstSROEstimate : bool
        If True, this is the first SRO estimation round --> do not apply exponential averaging.
    bufferFlagPos : int
        Cumulate buffer flags from initialization to current iteration, for current node pair.
    bufferFlagPri : int
        Cumulate buffer flags from initialization to "current iteration - `ld`", for current node pair.

    Returns
    -------
    sro_est : float
        Estimated residual SRO
        -- `nLocalSensors` first elements of output should be zero (no intra-node SROs)
    avg_res_prod_out : [2*(N-1) x 1] np.ndarray (complex)
        Exponentially averaged residuals (complex conjugate) product - post-processing.
    """

    # "Residuals" product
    res_prod = wPos * wPri.conj()
    # Prep for ISTFT (negative frequency bins too)
    res_prod = np.concatenate(
        [res_prod[:-1],
            np.conj(res_prod)[::-1][:-1]],
        -1
    )
    # Account for potential buffer flags (extra / missing sample)
    res_prod *= np.exp(1j * 2 * np.pi / len(res_prod) *\
        np.arange(len(res_prod)) * (bufferFlagPos - bufferFlagPri))

    # Update the average coherence product
    if flagFirstSROEstimate:
        avgResProd_out = res_prod     
        # ^^^ 1st SRO estimation, no exponential averaging (initialization)
    else:
        avgResProd_out = alpha * avgResProd + (1 - alpha) * res_prod 

    # Estimate SRO
    if method == 'gs':
        # --------- DWACD-inspired golden section search
        sro_est = - max_time_lag_search(avgResProd_out) / (ld * Ns)
    elif method == 'ls':
        # --------- Least-squares solution over frequency bins
        kappa = np.arange(0, len(wPri))    # freq. bins indices
        # b = 2 * np.pi * kappa * (ld * Ns) / len(kappa)
        b = np.pi * kappa * (ld * Ns) / (len(kappa) * 2)
        sro_est = - b.T @ np.angle(avgResProd_out[-len(kappa):]) / (b.T @ b)

    return sro_est, avgResProd_out


def dxcpphat_sro_estimation(
    fs,
    fsref,
    N,
    localSig: np.ndarray,
    neighboursSig: np.ndarray,
    refSensorIdx=0):
    """
    Estimates a(n) (residual) SRO using the DXCP-PhaT technique [1].

    References
    ----------
    - [1] A. Chinaev, P. Thüne, and G. Enzner (2021). Double-cross-correlation
    processing for blind sampling-rate and time-offset estimation. IEEE/ACM
    Transactions on Audio, Speech, and Language Processing, vol. 29,
    pp. 1881-1896.

    Parameters
    ----------
    fs : int or float
        Sampling frequency of current node. 
    TODO:

    Returns
    -------
    TODO:
    """

    DXCPPhatInstance = dxcp.DXCPPhaT(
        RefSampRate_fs_Hz=fsref,
        FrameSize_input=N
    )

    # For each neighbour, compute SRO
    sroEst = np.zeros(neighboursSig.shape[-1])
    for q in range(neighboursSig.shape[-1]):
        OutputDXCPPhaTcl = DXCPPhatInstance.process_data(
            x_12_ell=np.stack((localSig, neighboursSig[:, q]), axis=1)
        )
        sroEst[q] = OutputDXCPPhaTcl['SROppm_est_out']

        # FIXME: 16.01.2023: outputs ZEROS only??!

    return sroEst