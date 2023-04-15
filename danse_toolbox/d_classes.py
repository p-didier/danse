import copy
import shutil
import random
import warnings
import numpy as np
import scipy.linalg as sla
from siggen.classes import *
from dataclasses import dataclass
import danse_toolbox.d_base as base
import danse_toolbox.d_sros as sros


# @dataclass
# class CondNum:
#     """
#     Class that stores the condition number(s) of a matrix or list of matrices.
#     """
#     cn: float = np.nan  # condition number
#     t: float = np.nan   # time at which the condition number was computed

@dataclass
class ConditionNumbers:
    cn_RyyDANSE: list = field(default_factory=list)  # condition number of
        # the DANSE $\tilde{\mathbf{R}}_{\mathbf{y}_k\mathbf{y}_k}$ matrices.
    iter_cn_RyyDANSE: list = field(default_factory=list)  # iter. at which the
        # condition number of the DANSE
        # $\tilde{\mathbf{R}}_{\mathbf{y}_k\mathbf{y}_k}$ matrices was computed.
    cn_RyyLocal: list = field(default_factory=list)  # condition number of
        # the local $\mathbf{R}_{\mathbf{y}_k\mathbf{y}_k}$ matrices.
    iter_cn_RyyLocal: list = field(default_factory=list)  # iter. at which the
        # condition number of the local
        # $\mathbf{R}_{\mathbf{y}_k\mathbf{y}_k}$ matrices was computed.
    cn_RyyCentr: list = field(default_factory=list)  # condition number of
        # the centralised $\mathbf{R}_{\mathbf{y}\mathbf{y}}$ matrices.
    iter_cn_RyyCentr: list = field(default_factory=list)  # iter. at which the
        # condition number of the centralised
        # $\mathbf{R}_{\mathbf{y}\mathbf{y}}$ matrices was computed.

    def init(self, computeLocal, computeCentralised):
        """Initializes the condition number variables."""
        # Inform user about whether the condition numbers are computed
        # for the different matrices.
        self.cn_RyyDANSEcomputed = True  # always true
        if computeLocal:
            self.cn_RyyLocalComputed = True
        else:
            self.cn_RyyLocalComputed = False
        if computeCentralised:
            self.cn_RyyCentrComputed = True
        else:
            self.cn_RyyCentrComputed = False

    def get_new_cond_number(self, k, covMat, iter, type='DANSE'):
        """Computes and stores a new array of condition numbers
        of a matrix or list of matrices."""

        # Compute condition numbers
        cns = compute_condition_numbers(covMat, loopAxis=0)

        # Store condition numbers
        if type == 'DANSE':
            self.cn_RyyDANSE[k] = np.concatenate(
                (self.cn_RyyDANSE[k], cns[:, np.newaxis]),
                axis=1
            )
            self.iter_cn_RyyDANSE[k].append(iter)
        #
        elif type == 'local':
            self.cn_RyyLocal[k] = np.concatenate(
                (self.cn_RyyLocal[k], cns[:, np.newaxis]),
                axis=1
            )
            self.iter_cn_RyyLocal[k].append(iter)
        #
        elif type == 'centralised':
            self.cn_RyyCentr[k] = np.concatenate(
                (self.cn_RyyCentr[k], cns[:, np.newaxis]),
                axis=1
            )
            self.iter_cn_RyyCentr[k].append(iter)



def compute_condition_numbers(array, loopAxis=None):
    """Computes the condition number(s) of an array or list of arrays.
    Parameters
    ----------
    array : 2D or 3D np.ndarray or list[2D or 3D np.ndarray]
        Input matrices.
    loopAxis : int, optional
        Axis along which to loop in order to compute the condition numbers.
        Only used if `array` is a 3D array or list of 3D arrays.

    Returns
    ----------
    cond : float or np.ndarray[float]
        Condition number(s) of (matrices in) `array`.
    """

    def _compute_condition_number_ndarray(A, loopAx=None):
        """
        Helper function that computes the condition number of a 2D or 3D
        NumPy array `A` (along a given axis `ax` if `A` is 3D).
        """
        if isinstance(A, np.ndarray):
            if A.ndim == 2:
                cond = np.linalg.cond(A)
            elif A.ndim == 3:
                if loopAx is None:
                    warnings.warn('Axis along which to compute the condition number is not specified. Returning NaN for the condition number.')
                    return np.nan
                cond = np.zeros(A.shape[loopAx])
                for ii in range(A.shape[loopAx]):
                    if loopAx == 0:
                        cond[ii] = np.linalg.cond(A[ii, :, :])
                    elif loopAx == 1:
                        cond[ii] = np.linalg.cond(A[:, ii, :])
                    elif loopAx == 2:
                        cond[ii] = np.linalg.cond(A[:, :, ii])
            else:
                warnings.warn('Only 2D or 3D NumPy arrays are supported. Returning NaN for the condition number.')
                cond = np.nan
        else:
            raise TypeError('Input must be a NumPy array.')
        return cond

    # Compute condition number(s)
    if isinstance(array, np.ndarray):
        cond = _compute_condition_number_ndarray(array, loopAx=loopAxis)
    elif isinstance(array, list):
        cond = np.zeros((len(array), array[0].shape[loopAxis]))
        for ii in range(len(array)):
            cond[ii, :] = _compute_condition_number_ndarray(array[ii], loopAx=loopAxis)
    return cond


@dataclass
class ExportParameters:
    """Boolean decisions for what to export after running the DANSE algorithm."""
    filterNormsPlot: bool = True  # if True, filter norms plot is exported
    conditionNumberPlot: bool = True  # if True, condition number plot is exported
    convergencePlot: bool = True  # if True, convergence plot is exported
    wavFiles: bool = True  # if True, WAV files are exported
    acousticScenarioPlot: bool = True  # if True, acoustic scenario plot is exported
    sroEstimPerfPlot: bool = True  # if True, SRO estimation performance plot is exported
    metricsPlot: bool = True  # if True, metrics plot is exported
    waveformsAndSpectrograms: bool = True  # if True, waveforms and spectrograms are exported
    # vvv Files (not plots)
    danseOutputsFile: bool = True  # if True, DANSE outputs are exported as a pickle file
    parametersFile: bool = True  # if True, parameters are exported as a pickle file or a YAML file
    filterNorms: bool = True  # if True, filter norms are exported as a pickle file
    # vvv Global
    bypassAllExports: bool = False  # if True, all exports are bypassed
    exportFolder: str = ''  # folder to export outputs

@dataclass
class TestParameters:
    referenceSensor: int = 0    # Index of the reference sensor at each node
    #
    wasnParams: WASNparameters = WASNparameters(
        sigDur=5
    )
    danseParams: base.DANSEparameters = base.DANSEparameters()
    exportParams: ExportParameters = ExportParameters()
    #
    setThoseSensorsToNoise: list = field(default_factory=list)
    # ^^^ set those sensor indices to noise (render them useless).
    #
    seed: int = 12345
    snrYlimMax: float = None  # SNR ylim max (if None, use auto lim)
    loadedFromYaml: bool = False  # if True, the parameters were loaded from a YAML file
    originYaml: str = ''  # path to the YAML file from which the parameters were loaded

    def __post_init__(self):
        np.random.seed(self.seed)  # set random seed
        random.seed(self.seed)  # set random seed
        #
        self.testid = f'J{self.wasnParams.nNodes}Mk{list(self.wasnParams.nSensorPerNode)}WNn{self.wasnParams.nNoiseSources}Nd{self.wasnParams.nDesiredSources}T60_{int(self.wasnParams.t60)*1e3}ms'
        # Check consistency
        if self.danseParams.nodeUpdating == 'sym' and\
            any(self.wasnParams.SROperNode != 0):
            raise ValueError('Simultaneous node-updating impossible in the presence of SROs.')
        if not self.is_fully_connected_wasn() and\
            'topo-indep' not in self.danseParams.nodeUpdating:
            # Switch to topology-independent node-update system
            print(f'/!\ The WASN is not fully connected -- switching `danseParams.nodeUpdating` from "{self.danseParams.nodeUpdating}" to "topo-indep_{self.danseParams.nodeUpdating}".')
            self.danseParams.nodeUpdating = f'topo-indep_{self.danseParams.nodeUpdating}'
        if not self.exportParams.conditionNumberPlot:
            # If condition number plot is not exported, don't compute it
            self.danseParams.saveConditionNumber = False

    def save(self, exportType='pkl'):
        """Saves dataclass to Pickle archive."""
        met.save(self, self.exportFolder, exportType=exportType)

    def load(self, foldername, dataType='pkl'):
        """Loads dataclass to Pickle archive in folder `foldername`."""
        d = met.load(self, foldername, silent=True, dataType=dataType)
        d.__post_init__()  # re-initialize
        return d

    def is_fully_connected_wasn(self):
        return self.wasnParams.topologyParams.topologyType == 'fully-connected'
    
    def load_from_yaml(self, path):
        """Loads dataclass from YAML file."""
        self.loadedFromYaml = True  # flag to indicate that the object was loaded from YAML
        self.originYaml = path  # path to YAML file
        return met.load_from_yaml(path, self)
    
    def save_yaml(self):
        """Saves dataclass to YAML file."""
        if self.loadedFromYaml:
            # Copy YAML file to export folder
            shutil.copy(self.originYaml, self.exportParams.exportFolder)
            # Also save as readily readable .txt file
            met.save_as_txt(self, self.exportParams.exportFolder)
        else:
            raise ValueError('Cannot save YAML file: the parameters were not loaded from a YAML file.')


def check_if_fully_connected(wasn: list[Node]):
    """
    Returns True if the WASN is fully connected, False otherwise.
    """
    return np.array(
        [len(node.neighborsIdx) == len(wasn) - 1 for node in wasn]
    ).all()

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
        nSensorsTotal = sum([node.nSensors for node in wasn])
        self.nPosFreqs = int(self.DFTsize // 2 + 1)  # number of >0 freqs.
        # Expected number of DANSE iterations (==  # of signal frames)
        self.nIter = int((wasn[0].data.shape[0] - self.DFTsize) / self.Ns) + 1
        # Check for TI-DANSE case
        tidanseFlag = ~check_if_fully_connected(wasn)

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

        def _init_complex_filter(size, refIdx=0, initType='selectFirstSensor', fixedValue=0.):
            """Returns an initialized STFT-domain filter vector,
            i.e., a selector of the reference sensor at node 1."""
            if initType == 'selectFirstSensor':
                wInit = np.zeros(size, dtype=complex)
                if len(size) == 3:
                    wInit[:, :, refIdx] = 1
                elif len(size) == 2:
                    wInit[:, refIdx] = 1
            elif initType == 'random':
                rng = np.random.default_rng(self.seed)
                wInit = (rng.random(size) - 0.5) +\
                    1j * (rng.random(size) - 0.5)
            elif initType == 'fixedValue':
                wInit = np.full(size, fill_value=fixedValue, dtype=complex)
            elif initType == self.filterInitType:
                wInit = np.full(size, fill_value=fixedValue, dtype=complex)
                if len(size) == 3:
                    wInit[:, :, refIdx] = 1
                elif len(size) == 2:
                    wInit[:, refIdx] = 1
            return wInit

        # Initialize covariance matrices slices
        rng = np.random.default_rng(self.seed)

        def _init_covmats(dims: tuple) -> np.ndarray:
            """Helper function to initialize the covariance matrices."""
            # Generate a basis random array for random initializations
            randArray = 2 * rng.random(dims) - 1 +\
                1j * (2 * rng.random(dims) - 1)
            if self.covMatInitType == 'fully_random':
                # Scale the random array
                fullSlice = self.covMatRandomInitScaling * randArray
            elif self.covMatInitType == 'eye_and_random':
                if len(dims) == 2:  # for single freq-bin slice
                    eyePart = np.eye(dims[-1]) * self.covMatEyeInitScaling
                elif len(dims) == 3:  # for multiple bins
                    eyePart = np.tile(
                        np.eye(dims[-1]) * self.covMatEyeInitScaling,
                        (dims[0], 1, 1)
                    )
                # Scale the random array and add an identity matrix
                # to each slice.
                fullSlice = eyePart + self.covMatRandomInitScaling * randArray
            return fullSlice

        # Covariance matrices initialization
        if self.covMatInitType == 'batch_ground_truth':
            raise ValueError('[NYI]')
            # TODO: -- see journal week14_2023 TUE
            
            # Initialise the covariance matrices with their batch ground truth
            for k in range(nNodes):
                nHat = np.fft.fft(wasn[k].cleannoise, n=self.DFTsize, axis=0)
                yHat = np.fft.fft(wasn[k].data, n=self.DFTsize, axis=0)
                Rnntilde.append(np.einsum('ij,ik->ijk', nHat, nHat.conj()))
                Ryytilde.append(np.einsum('ij,ik->ijk', yHat, yHat.conj()))

                # Rnncentr.append(np.tile(sliceCentr, (self.nPosFreqs, 1, 1)))
                # Ryycentr.append(np.tile(sliceCentr, (self.nPosFreqs, 1, 1)))

                # Rnnlocal.append(np.tile(sliceLocal, (self.nPosFreqs, 1, 1)))
                # Ryylocal.append(np.tile(sliceLocal, (self.nPosFreqs, 1, 1)))
        elif self.covMatSameInitForAllNodes:
            if self.covMatSameInitForAllFreqs:
                fullSlice = _init_covmats(
                    (nSensorsTotal, nSensorsTotal)
                )
            else:
                fullSlice = _init_covmats(
                    (self.nPosFreqs, nSensorsTotal, nSensorsTotal)
                )
        
        for k in range(nNodes):
            nNeighbors = len(wasn[k].neighborsIdx)
            #
            avgProdResiduals.append(np.zeros(
                (self.DFTsize, nNeighbors),dtype=complex
                ))
            # init all buffer flags at 0 (assuming no over- or under-flow)
            bufferFlags.append(np.zeros((self.nIter, nNeighbors)))    
            #
            if tidanseFlag:
                dimYTilde[k] = wasn[k].nSensors + 1
            else:
                dimYTilde[k] = wasn[k].nSensors + nNeighbors
            # initiate phase shift factors as 0's (no phase shift)
            phaseShiftFactors.append(np.zeros(dimYTilde[k]))   
            #
            if not self.covMatSameInitForAllNodes:
                if self.covMatSameInitForAllFreqs:
                    fullSlice = _init_covmats(
                        (nSensorsTotal, nSensorsTotal)
                    )
                else:
                    fullSlice = _init_covmats(
                        (self.nPosFreqs, nSensorsTotal, nSensorsTotal)
                    )
            
            if self.covMatSameInitForAllFreqs:
                sliceTilde = fullSlice[:dimYTilde[k], :dimYTilde[k]]
                Rnntilde.append(np.tile(sliceTilde, (self.nPosFreqs, 1, 1)))
                Ryytilde.append(np.tile(sliceTilde, (self.nPosFreqs, 1, 1)))
                #
                sliceCentr = copy.deepcopy(fullSlice)
                Rnncentr.append(np.tile(sliceCentr, (self.nPosFreqs, 1, 1)))
                Ryycentr.append(np.tile(sliceCentr, (self.nPosFreqs, 1, 1)))
                #
                sliceLocal = fullSlice[:wasn[k].nSensors, :wasn[k].nSensors]
                Rnnlocal.append(np.tile(sliceLocal, (self.nPosFreqs, 1, 1)))
                Ryylocal.append(np.tile(sliceLocal, (self.nPosFreqs, 1, 1)))
            else:
                Rnntilde.append(fullSlice[:, :dimYTilde[k], :dimYTilde[k]])
                Ryytilde.append(fullSlice[:, :dimYTilde[k], :dimYTilde[k]])
                #
                Rnncentr.append(fullSlice)
                Ryycentr.append(fullSlice)
                #
                Rnnlocal.append(
                    fullSlice[:, :wasn[k].nSensors, :wasn[k].nSensors]
                )
                Ryylocal.append(
                    fullSlice[:, :wasn[k].nSensors, :wasn[k].nSensors]
                )
            
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
                self.referenceSensor,
                initType=self.filterInitType,
                fixedValue=self.filterInitFixedValue
            ))
            if tidanseFlag:
                wTildeExt.append(_init_complex_filter(
                    (self.nPosFreqs, dimYTilde[k]),
                    self.referenceSensor,
                    initType=self.filterInitType,
                    fixedValue=self.filterInitFixedValue
                ))
                wTildeExtTarget.append(_init_complex_filter(
                    (self.nPosFreqs, dimYTilde[k]),
                    self.referenceSensor,
                    initType=self.filterInitType,
                    fixedValue=self.filterInitFixedValue
                ))
            else:
                wTildeExt.append(_init_complex_filter(
                    (self.nPosFreqs, wasn[k].nSensors),
                    self.referenceSensor,
                    initType=self.filterInitType,
                    fixedValue=self.filterInitFixedValue
                ))
                wTildeExtTarget.append(_init_complex_filter(
                    (self.nPosFreqs, wasn[k].nSensors),
                    self.referenceSensor,
                    initType=self.filterInitType,
                    fixedValue=self.filterInitFixedValue
                ))
            wCentr.append(_init_complex_filter(
                (self.nPosFreqs, self.nIter + 1, nSensorsTotal),
                self.referenceSensor,
                initType=self.filterInitType,
                fixedValue=self.filterInitFixedValue
            ))
            wLocal.append(_init_complex_filter(
                (self.nPosFreqs, self.nIter + 1, wasn[k].nSensors),
                self.referenceSensor,
                initType=self.filterInitType,
                fixedValue=self.filterInitFixedValue
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
        self.cleanSpeechSignalsAtNodes = [node.cleanspeech for node in wasn]
        self.cleanNoiseSignalsAtNodes = [node.cleannoise for node in wasn]
        self.d = np.zeros(
            (wasn[self.referenceSensor].data.shape[0], nNodes)
        )
        self.d_s = np.zeros(
            (wasn[self.referenceSensor].data.shape[0], nNodes)
        )
        self.d_n = np.zeros(
            (wasn[self.referenceSensor].data.shape[0], nNodes)
        )
        self.dCentr = np.zeros(
            (wasn[self.referenceSensor].data.shape[0], nNodes)
        )
        self.dCentr_s = np.zeros(
            (wasn[self.referenceSensor].data.shape[0], nNodes)
        )
        self.dCentr_n = np.zeros(
            (wasn[self.referenceSensor].data.shape[0], nNodes)
        )
        self.dLocal = np.zeros(
            (wasn[self.referenceSensor].data.shape[0], nNodes)
        )
        self.dLocal_s = np.zeros(
            (wasn[self.referenceSensor].data.shape[0], nNodes)
        )
        self.dLocal_n = np.zeros(
            (wasn[self.referenceSensor].data.shape[0], nNodes)
        )
        self.i = np.zeros(nNodes, dtype=int)
        self.dimYTilde = dimYTilde
        self.dhat = np.zeros(
            (self.nPosFreqs, self.nIter, nNodes), dtype=complex
        )
        self.dhat_s = np.zeros(
            (self.nPosFreqs, self.nIter, nNodes), dtype=complex
        )
        self.dhat_n = np.zeros(
            (self.nPosFreqs, self.nIter, nNodes), dtype=complex
        )
        self.dHatCentr = np.zeros(
            (self.nPosFreqs, self.nIter, nNodes), dtype=complex
        )
        self.dHatCentr_s = np.zeros(
            (self.nPosFreqs, self.nIter, nNodes), dtype=complex
        )
        self.dHatCentr_n = np.zeros(
            (self.nPosFreqs, self.nIter, nNodes), dtype=complex
        )
        self.dHatLocal = np.zeros(
            (self.nPosFreqs, self.nIter, nNodes), dtype=complex
        )
        self.dHatLocal_s = np.zeros(
            (self.nPosFreqs, self.nIter, nNodes), dtype=complex
        )
        self.dHatLocal_n = np.zeros(
            (self.nPosFreqs, self.nIter, nNodes), dtype=complex
        )
        self.downstreamNeighbors = [node.downstreamNeighborsIdx\
                                    for node in wasn]
        self.expAvgBeta = [node.beta for node in wasn]
        self.firstDANSEupdateRefSensor = None  # init
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
        self.oVADframes = [np.zeros(self.nIter) for _ in range(nNodes)]
        self.centrVADframes = np.full(self.nIter, fill_value=None)
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
        self.upstreamNeighbors = [node.upstreamNeighborsIdx for node in wasn]
        self.yin = [node.data for node in wasn]
        self.yyH = yyH
        self.yyHuncomp = yyHuncomp
        self.yCentr = yCentr
        self.yCentr_s = copy.deepcopy(yCentr)
        self.yCentr_n = copy.deepcopy(yCentr)
        self.yHatCentr = yHatCentr
        self.yHatCentr_s = copy.deepcopy(yHatCentr)
        self.yHatCentr_n = copy.deepcopy(yHatCentr)
        self.yLocal = yLocal
        self.yLocal_s = copy.deepcopy(yLocal)
        self.yLocal_n = copy.deepcopy(yLocal)
        self.yHatLocal = yHatLocal
        self.yHatLocal_s = copy.deepcopy(yHatLocal)
        self.yHatLocal_n = copy.deepcopy(yHatLocal)
        self.yTilde = yTilde
        self.yTilde_s = copy.deepcopy(yTilde)
        self.yTilde_n = copy.deepcopy(yTilde)
        self.yTildeHat = yTildeHat
        self.yTildeHat_s = copy.deepcopy(yTildeHat)
        self.yTildeHat_n = copy.deepcopy(yTildeHat)
        self.yTildeHatUncomp = yTildeHatUncomp
        self.wIR = wIR
        self.wTilde = wTilde
        self.wCentr = wCentr
        self.wLocal = wLocal
        self.wTildeExt = wTildeExt
        self.wTildeExtTarget = wTildeExtTarget
        self.z = z
        self.z_s = copy.deepcopy(z)
        self.z_n = copy.deepcopy(z)
        self.zBuffer = zBuffer
        self.zBuffer_s = copy.deepcopy(zBuffer)
        self.zBuffer_n = copy.deepcopy(zBuffer)
        self.zLocal = zLocal
        self.zLocal_s = copy.deepcopy(zLocal)
        self.zLocal_n = copy.deepcopy(zLocal)

        # For centralised and local estimates
        self.yinStacked = np.concatenate(
            tuple([x for x in self.yin]),
            axis=-1
        )
        self.yinStacked_s = np.concatenate(
            tuple([x for x in self.cleanSpeechSignalsAtNodes]),
            axis=-1
        )
        self.yinStacked_n = np.concatenate(
            tuple([x for x in self.cleanNoiseSignalsAtNodes]),
            axis=-1
        )

        # For debugging purposes
        initCNlist = [np.empty((self.nPosFreqs, 0)) for _ in range(nNodes)]
        initIterCNlist = [[] for _ in range(nNodes)]
        self.condNumbers = ConditionNumbers(
            cn_RyyDANSE=copy.deepcopy(initCNlist),
            iter_cn_RyyDANSE=copy.deepcopy(initIterCNlist),
            cn_RyyLocal=copy.deepcopy(initCNlist),
            iter_cn_RyyLocal=copy.deepcopy(initIterCNlist),
            cn_RyyCentr=copy.deepcopy(initCNlist),
            iter_cn_RyyCentr=copy.deepcopy(initIterCNlist),
        )
        # Inform about which condition numbers are to be computed
        self.condNumbers.init(self.computeLocal, self.computeCentralised)
        # Information about the last saved condition number
        self.lastCondNumberSaveRyyTilde = [-1 for _ in range(nNodes)]
        self.lastCondNumberSaveRyyLocal = [-1 for _ in range(nNodes)]
        self.lastCondNumberSaveRyyCentr = [-1 for _ in range(nNodes)]

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
        ykFrame = base.local_chunk_for_broadcast(
            self.yin[k],
            tCurr,
            fs,
            self.DFTsize
        )
        ykFrame_s = base.local_chunk_for_broadcast(
            self.cleanSpeechSignalsAtNodes[k],
            tCurr,
            fs,
            self.DFTsize
        )   # - speech-only for SNR computation
        ykFrame_n = base.local_chunk_for_broadcast(
            self.cleanNoiseSignalsAtNodes[k],
            tCurr,
            fs,
            self.DFTsize
        )   # - noise-only for SNR computation

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
            _, self.zLocal_s[k] = base.danse_compression_whole_chunk(
                ykFrame_s,
                self.wTildeExt[k],
                self.winWOLAanalysis,
                self.winWOLAsynthesis,
                zqPrevious=self.zLocal_s[k]
            )  # - speech-only for SNR computation
            _, self.zLocal_n[k] = base.danse_compression_whole_chunk(
                ykFrame_n,
                self.wTildeExt[k],
                self.winWOLAanalysis,
                self.winWOLAsynthesis,
                zqPrevious=self.zLocal_n[k]
            )  # - noise-only for SNR computation

            # Fill buffers in
            self.zBuffer = base.fill_buffers_whole_chunk(
                k,
                self.neighbors,
                self.zBuffer,
                self.zLocal[k][:self.Ns]
            ) 
            self.zBuffer_s = base.fill_buffers_whole_chunk(
                k,
                self.neighbors,
                self.zBuffer_s,
                self.zLocal_s[k][:self.Ns]
            )  # - speech-only for SNR computation
            self.zBuffer_n = base.fill_buffers_whole_chunk(
                k,
                self.neighbors,
                self.zBuffer_n,
                self.zLocal_n[k][:self.Ns]
            )  # - noise-only for SNR computation

            stop = 1
        
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

            self.zLocal_s[k], _ = base.danse_compression_few_samples(
                ykFrame_s,
                self.wTildeExt[k],
                currL,
                self.wIR[k],
                self.winWOLAanalysis,
                self.winWOLAsynthesis,
                self.Ns,
                updateBroadcastFilter
            )  # - speech-only for SNR computation
            self.zLocal_n[k], _ = base.danse_compression_few_samples(
                ykFrame_n,
                self.wTildeExt[k],
                currL,
                self.wIR[k],
                self.winWOLAanalysis,
                self.winWOLAsynthesis,
                self.Ns,
                updateBroadcastFilter
            )  # - noise-only for SNR computation
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
            self.zBuffer_s = base.fill_buffers_td_few_samples(
                k,
                self.neighbors,
                self.zBuffer_s,
                self.zLocal_s[k],
                currL
            )  # - speech-only for SNR computation
            self.zBuffer_n = base.fill_buffers_td_few_samples(
                k,
                self.neighbors,
                self.zBuffer_n,
                self.zLocal_n[k],
                currL
            )  # - noise-only for SNR computation
        
    def update_and_estimate(self, tCurr, fs, k, bypassUpdateEventMat=False):
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
        bypassUpdateEventMat : bool
            If true, bypass filter udpate.
            (but still compute the desired signal estimate!)
        """

        if k == self.referenceSensor and self.nInternalFilterUps[k] == 0:
            # Save first update instant (for, e.g., SRO plot)
            self.firstDANSEupdateRefSensor = tCurr

        # Process buffers
        self.process_incoming_signals_buffers(k, tCurr)
        # Wipe local buffers
        self.zBuffer[k] = [np.array([])\
            for _ in range(len(self.neighbors[k]))]
        self.zBuffer_s[k] = [np.array([])\
            for _ in range(len(self.neighbors[k]))]  # - speech-only for SNR computation
        self.zBuffer_n[k] = [np.array([])\
            for _ in range(len(self.neighbors[k]))]  # - noise-only for SNR computation
        # Construct `\tilde{y}_k` in frequency domain and VAD at current frame
        self.build_ytilde(tCurr, fs, k)
        # Consider local / centralised estimation(s)
        if self.computeCentralised:
            self.build_ycentr(tCurr, fs, k)
        if self.computeLocal:  # extract local info from `\tilde{y}_k`
            self.yLocal[k][:, self.i[k], :] =\
                self.yTilde[k][:, self.i[k], :self.nSensorPerNode[k]]
            self.yLocal_s[k][:, self.i[k], :] =\
                self.yTilde_s[k][:, self.i[k], :self.nSensorPerNode[k]]
            self.yLocal_n[k][:, self.i[k], :] =\
                self.yTilde_n[k][:, self.i[k], :self.nSensorPerNode[k]]
            #
            self.yHatLocal[k][:, self.i[k], :] =\
                self.yTildeHat[k][:, self.i[k], :self.nSensorPerNode[k]]
            self.yHatLocal_s[k][:, self.i[k], :] =\
                self.yTildeHat_s[k][:, self.i[k], :self.nSensorPerNode[k]]
            self.yHatLocal_n[k][:, self.i[k], :] =\
                self.yTildeHat_n[k][:, self.i[k], :self.nSensorPerNode[k]]
        # Account for buffer flags
        skipUpdate = self.compensate_sros(k, tCurr)
        
        # Ryy and Rnn updates (including centralised / local, if needed)
        self.spatial_covariance_matrix_update(k)
        # Check quality of covariance matrix estimates 
        self.check_covariance_matrices(k)

        if not skipUpdate and not bypassUpdateEventMat:
            # If covariance matrices estimates are full-rank, update filters
            self.perform_update(k)
            # ^^^ depends on outcome of `check_covariance_matrices()`.
        else:
            # Do not update the filter coefficients
            self.wTilde[k][:, self.i[k] + 1, :] =\
                self.wTilde[k][:, self.i[k], :]
            if self.computeCentralised:
                self.wCentr[k][:, self.i[k] + 1, :] =\
                    self.wCentr[k][:, self.i[k], :]
            if self.computeLocal:
                self.wLocal[k][:, self.i[k] + 1, :] =\
                    self.wLocal[k][:, self.i[k], :]
            if skipUpdate:
                print(f'Node {k+1}: {self.i[k]+1}^th update skipped.')
        if self.bypassUpdates:
            print('!! User-forced bypass of filter coefficients updates !!')

        # Update external filters (for broadcasting)
        self.update_external_filters(k, tCurr)
        # Update SRO estimates
        self.update_sro_estimates(k, fs)
        # Update phase shifts for SRO compensation
        if self.compensateSROs:
            self.build_phase_shifts_for_srocomp(k)
        # Compute desired signal chunk estimate
        self.get_desired_signal(k)
        # Update iteration index
        self.i[k] += 1

    def compute_vad(self, k):
        """
        Computes the local VAD for the current frame at node `k`.
        
        Parameters
        ----------
        k : int
            Node index.
        """
        def _compute_vad_curr_nodes(k):
            """Helper function -- compute current-frame VAD at node `k`."""
            # Compute VAD
            VADinFrame = self.fullVAD[k][
                np.amax([self.idxBegChunk, 0]):self.idxEndChunk
            ]
            # If there is a majority of "VAD = 1" in the frame,
            # set the frame-wise VAD to 1.
            return sum(VADinFrame == 0) <= len(VADinFrame) // 2
            
        self.oVADframes[k][self.i[k]] = _compute_vad_curr_nodes(k)
        # Count number of spatial covariance matrices updates
        if self.oVADframes[k][self.i[k]]:
            self.numUpdatesRyy[k] += 1
        else:
            self.numUpdatesRnn[k] += 1

        # If needed, compute centralised VAD
        if self.computeCentralised and self.centrVADframes[self.i[k]] is None:
            self.centrVADframes[self.i[k]] = np.array([
                _compute_vad_curr_nodes(k) for k in range(self.nNodes)
            ]).all()


    def check_covariance_matrices(self, k):
        """
        Checks that the number of rank-1 covariance matrix estimate updates
        done in `spatial_covariance_matrix_update()` is at least equal to
        the dimension of the corresponding covariance matrix (ensuring full-
        rank property).
        >> Addition on 12.12.2022:
        -If GEVD is to be performed: check positive-definiteness of noise-only
        covariance matrices. 
        >> Addition on 13.12.2022:
        -If GEVD is to be performed: explicitly check full-rank property.

        Parameters
        ----------
        k : int
            Node index.
        """

        # def _is_pos_def(x):  # https://stackoverflow.com/a/16270026
        #     """Finds out whether matrix `x` is positive definite."""
        #     return np.all(np.linalg.eigvals(x) > 0)
        def _is_hermitian_and_posdef(x):
            """Finds out whether 3D complex matrix `x` is Hermitian along 
            the two last axes, as well as positive definite."""
            # Get rid of machine-precision residual imaginary parts
            x = np.real_if_close(x)
            # Assess Hermitian-ness
            b1 = np.allclose(np.transpose(x, axes=(0,2,1)).conj(), x)
            # Assess positive-definiteness
            b2 = True
            for ii in range(x.shape[0]):
                if any(np.linalg.eigvalsh(x[ii, :, :]) < 0):
                    b2 = False
                    break
            return b1 and b2
        
        def _check_validity(RnnMat, RyyMat):
            """Helper function for conciseness: combine validity checks."""
            def __full_rank_check(mat):
                """Helper subfunction: check full-rank property."""
                return (np.linalg.matrix_rank(mat) == mat.shape[-1]).all()
            check1 = _is_hermitian_and_posdef(RnnMat) 
            check2 = _is_hermitian_and_posdef(RyyMat) 
            check3 = __full_rank_check(RnnMat)
            check4 = __full_rank_check(RyyMat)
            return check1 and check2 and check3 and check4

        if not self.startUpdates[k]:
            if self.numUpdatesRyy[k] > self.Ryytilde[k].shape[-1] and \
                self.numUpdatesRnn[k] > self.Ryytilde[k].shape[-1]:
                if self.performGEVD:
                    if _check_validity(self.Rnntilde[k], self.Ryytilde[k]):
                        self.startUpdates[k] = True
                else:
                    self.startUpdates[k] = True

        # Centralised estimate
        if self.computeCentralised and not self.startUpdatesCentr[k]:
            if self.numUpdatesRyy[k] > self.Ryycentr[k].shape[-1] and \
                self.numUpdatesRnn[k] > self.Ryycentr[k].shape[-1]:
                if self.performGEVD:
                    if _check_validity(self.Rnncentr[k], self.Ryycentr[k]):
                        self.startUpdatesCentr[k] = True
                else:
                    self.startUpdatesCentr[k] = True

        # Local estimate
        if self.computeLocal and not self.startUpdatesLocal[k]:
            if self.numUpdatesRyy[k] > self.Ryylocal[k].shape[-1] and \
                self.numUpdatesRnn[k] > self.Ryylocal[k].shape[-1]:
                if self.performGEVD:
                    if _check_validity(self.Rnnlocal[k], self.Ryylocal[k]):
                        self.startUpdatesLocal[k] = True
                else:
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
        yCentrCurr_s, _, _ = base.local_chunk_for_update(
            self.yinStacked_s,
            tCurr,
            fs,
            bd=self.broadcastType,
            Ndft=self.DFTsize,
            Ns=self.Ns
        )
        yCentrCurr_n, _, _ = base.local_chunk_for_update(
            self.yinStacked_n,
            tCurr,
            fs,
            bd=self.broadcastType,
            Ndft=self.DFTsize,
            Ns=self.Ns
        )
        self.yCentr[k][:, self.i[k], :] = yCentrCurr
        self.yCentr_s[k][:, self.i[k], :] = yCentrCurr_s
        self.yCentr_n[k][:, self.i[k], :] = yCentrCurr_n

        # Go to frequency domain
        yHatCentrCurr = 1 / self.normFactWOLA * np.fft.fft(
            self.yCentr[k][:, self.i[k], :] *\
                self.winWOLAanalysis[:, np.newaxis],
            self.DFTsize,
            axis=0
        )
        yHatCentrCurr_s = 1 / self.normFactWOLA * np.fft.fft(
            self.yCentr_s[k][:, self.i[k], :] *\
                self.winWOLAanalysis[:, np.newaxis],
            self.DFTsize,
            axis=0
        )
        yHatCentrCurr_n = 1 / self.normFactWOLA * np.fft.fft(
            self.yCentr_n[k][:, self.i[k], :] *\
                self.winWOLAanalysis[:, np.newaxis],
            self.DFTsize,
            axis=0
        )
        # Keep only positive frequencies
        self.yHatCentr[k][:, self.i[k], :] = yHatCentrCurr[:self.nPosFreqs, :]
        self.yHatCentr_s[k][:, self.i[k], :] = yHatCentrCurr_s[:self.nPosFreqs, :]
        self.yHatCentr_n[k][:, self.i[k], :] = yHatCentrCurr_n[:self.nPosFreqs, :]


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

        if self.noExternalFilterRelaxation:
            # No relaxation (i.e., no "external" filters)
            self.wTildeExt[k] =\
                self.wTilde[k][:, self.i[k] + 1, :self.nLocalMic[k]]
        else:
            # Simultaneous or asynchronous node-updating
            if 'seq' not in self.nodeUpdating:
                # Relaxed external filter update
                self.wTildeExt[k] = self.expAvgBeta[k] * self.wTildeExt[k] +\
                    (1 - self.expAvgBeta[k]) *  self.wTildeExtTarget[k]
                # Update targets
                if t - self.lastExtFiltUp[k] >= self.timeBtwExternalFiltUpdates:
                    self.wTildeExtTarget[k] = (1 - self.alphaExternalFilters) *\
                        self.wTildeExtTarget[k] + self.alphaExternalFilters *\
                        self.wTilde[k][:, self.i[k] + 1, :self.nLocalMic[k]]
                    # Update last external filter update instant [s]
                    self.lastExtFiltUp[k] = t
                    if self.printoutsAndPlotting.printout_externalFilterUpdate:    # inform user
                        print(f't={np.round(t, 3)}s -- UPDATING EXTERNAL FILTERS for node {k+1} (scheduled every [at least] {self.timeBtwExternalFiltUpdates}s)')
            # Sequential node-updating
            else:
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
        zk_s = np.empty((self.DFTsize, 0), dtype=float)
        zk_n = np.empty((self.DFTsize, 0), dtype=float)

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
                        zCurrBuffer_s = np.concatenate((
                            np.zeros(Ndft - Bq),
                            self.zBuffer_s[k][idxq]
                        ))
                        zCurrBuffer_n = np.concatenate((
                            np.zeros(Ndft - Bq),
                            self.zBuffer_n[k][idxq]
                        ))
                    elif Bq == 0:
                        # Node `q` has not yet transmitted enough data to node
                        # `k`, but node `k` has already reached its first
                        # update instant. Interpretation: Node `q` samples
                        # slower than node `k`. 
                        print(f'[b- @ t={np.round(t, 3)}s] Buffer underflow at current node`s B_{self.neighbors[k][idxq]+1} buffer | -1 broadcast')
                        bufferFlags[idxq] = -1      # raise negative flag
                        zCurrBuffer = np.zeros(Ndft)
                        zCurrBuffer_s = np.zeros(Ndft)
                        zCurrBuffer_n = np.zeros(Ndft)
                else:
                    if Bq == Ns:
                        # All good, no under-/over-flows
                        if not np.any(self.z[k]):
                            # Not yet any previous buffer
                            # -- need to appstart zeros.
                            zCurrBuffer = np.concatenate((
                                np.zeros(Ndft - Bq), self.zBuffer[k][idxq]
                            ))
                            zCurrBuffer_s = np.concatenate((
                                np.zeros(Ndft - Bq), self.zBuffer_s[k][idxq]
                            ))
                            zCurrBuffer_n = np.concatenate((
                                np.zeros(Ndft - Bq), self.zBuffer_n[k][idxq]
                            ))
                        else:
                            # Concatenate last `Ns` samples of previous buffer
                            # with current buffer.
                            zCurrBuffer = np.concatenate((
                                self.z[k][-Ns:, idxq], self.zBuffer[k][idxq]
                            ))
                            zCurrBuffer_s = np.concatenate((
                                self.z_s[k][-Ns:, idxq], self.zBuffer_s[k][idxq]
                            ))
                            zCurrBuffer_n = np.concatenate((
                                self.z_n[k][-Ns:, idxq], self.zBuffer_n[k][idxq]
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
                        zCurrBuffer_s = self.zBuffer_s[k][idxq]
                        zCurrBuffer_n = self.zBuffer_n[k][idxq]
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
                        zCurrBuffer_s = np.concatenate(
                            (np.zeros(Ndft - Bq), self.zBuffer_s[k][idxq]),
                            axis=0
                        )
                        zCurrBuffer_n = np.concatenate(
                            (np.zeros(Ndft - Bq), self.zBuffer_n[k][idxq]),
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
                        zCurrBuffer_s = self.zBuffer_s[k][idxq][-Ndft:]
                        zCurrBuffer_n = self.zBuffer_n[k][idxq][-Ndft:]
                
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
                        zCurrBuffer_s = np.concatenate(
                            (self.z_s[k][-(Ndft - Bq):, idxq],
                            self.zBuffer_s[k][idxq]),
                            axis=0
                        )
                        zCurrBuffer_n = np.concatenate(
                            (self.z_n[k][-(Ndft - Bq):, idxq],
                            self.zBuffer_n[k][idxq]),
                            axis=0
                        )
                    else:   # edge case: no overlap between consecutive frames
                        zCurrBuffer = self.zBuffer[k][idxq]
                        zCurrBuffer_s = self.zBuffer_s[k][idxq]
                        zCurrBuffer_n = self.zBuffer_n[k][idxq]

            # Stack compressed signals
            zk = np.concatenate((zk, zCurrBuffer[:, np.newaxis]), axis=1)
            zk_s = np.concatenate((zk_s, zCurrBuffer_s[:, np.newaxis]), axis=1)
            zk_n = np.concatenate((zk_n, zCurrBuffer_n[:, np.newaxis]), axis=1)

        # Update DANSE variables
        self.z[k] = zk
        self.z_s[k] = zk_s
        self.z_n[k] = zk_n
        self.bufferFlags[k][self.i[k], :] = bufferFlags

    
    def build_ytilde(self, tCurr, fs, k):
        """
        Builds the full observation vector used in the DANSE filter update.

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
        # Extract current local data chunk
        yLocalCurr, self.idxBegChunk, self.idxEndChunk =\
            base.local_chunk_for_update(
                self.yin[k],
                tCurr,
                fs,
                bd=self.broadcastType,
                Ndft=self.DFTsize,
                Ns=self.Ns
            )
        yLocalCurr_s, _, _ = base.local_chunk_for_update(
            self.cleanSpeechSignalsAtNodes[k],
            tCurr,
            fs,
            bd=self.broadcastType,
            Ndft=self.DFTsize,
            Ns=self.Ns
        )
        yLocalCurr_n, _, _ = base.local_chunk_for_update(
            self.cleanNoiseSignalsAtNodes[k],
            tCurr,
            fs,
            bd=self.broadcastType,
            Ndft=self.DFTsize,
            Ns=self.Ns
        )

        # Compute VAD
        self.compute_vad(k)

        # Build full available observation vector
        yTildeCurr = np.concatenate((yLocalCurr, self.z[k]), axis=1)
        yTildeCurr_s = np.concatenate((yLocalCurr_s, self.z_s[k]), axis=1)
        yTildeCurr_n = np.concatenate((yLocalCurr_n, self.z_n[k]), axis=1)
        self.yTilde[k][:, self.i[k], :] = yTildeCurr
        self.yTilde_s[k][:, self.i[k], :] = yTildeCurr_s
        self.yTilde_n[k][:, self.i[k], :] = yTildeCurr_n
        # Go to frequency domain
        yTildeHatCurr = 1 / self.normFactWOLA * np.fft.fft(
            self.yTilde[k][:, self.i[k], :] *\
                self.winWOLAanalysis[:, np.newaxis],
            self.DFTsize,
            axis=0
        )
        yTildeHatCurr_s = 1 / self.normFactWOLA * np.fft.fft(
            self.yTilde_s[k][:, self.i[k], :] *\
                self.winWOLAanalysis[:, np.newaxis],
            self.DFTsize,
            axis=0
        )
        yTildeHatCurr_n = 1 / self.normFactWOLA * np.fft.fft(
            self.yTilde_n[k][:, self.i[k], :] *\
                self.winWOLAanalysis[:, np.newaxis],
            self.DFTsize,
            axis=0
        )
        # Keep only positive frequencies
        self.yTildeHat[k][:, self.i[k], :] = yTildeHatCurr[:self.nPosFreqs, :]
        self.yTildeHat_s[k][:, self.i[k], :] = yTildeHatCurr_s[:self.nPosFreqs, :]
        self.yTildeHat_n[k][:, self.i[k], :] = yTildeHatCurr_n[:self.nPosFreqs, :]


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

        def _update_covmats(Ryy, Rnn, yyH, vad, beta=self.expAvgBeta[k]):
            """Helper function to perform exponential averaging."""
            if vad:
                Ryy = beta * Ryy + (1 - beta) * yyH
            else:
                Rnn = beta * Rnn + (1 - beta) * yyH
            return Ryy, Rnn

        # Perform DANSE covariance matrices udpates
        y = self.yTildeHat[k][:, self.i[k], :]  # concise renaming
        yyH = np.einsum('ij,ik->ijk', y, y.conj())  # outer product
        self.Ryytilde[k], self.Rnntilde[k] = _update_covmats(
            self.Ryytilde[k],
            self.Rnntilde[k],
            yyH,
            vad=self.oVADframes[k][self.i[k]]
        )  # update
        if self.saveConditionNumber and\
            self.i[k] - self.lastCondNumberSaveRyyTilde[k] >=\
                self.saveConditionNumberEvery:  # save condition number
            # Inform user
            print('Computing condition numbers (Ryy matrices)...')
            self.condNumbers.get_new_cond_number(
                k,
                self.Ryytilde[k],
                iter=self.i[k],
                type='DANSE'
            )
            self.lastCondNumberSaveRyyTilde[k] = self.i[k]

        self.yyH[k][self.i[k], :, :, :] = yyH  # saving for SRO estimation
        
        # Consider local estimates
        if self.computeLocal:
            y = self.yHatLocal[k][:, self.i[k], :]  # concise renaming
            yyH = np.einsum('ij,ik->ijk', y, y.conj())
            self.Ryylocal[k], self.Rnnlocal[k] = _update_covmats(
                self.Ryylocal[k],
                self.Rnnlocal[k],
                yyH,
                vad=self.oVADframes[k][self.i[k]]
            )  # update local
            if self.saveConditionNumber and\
                self.i[k] - self.lastCondNumberSaveRyyLocal[k] >=\
                    self.saveConditionNumberEvery:  # save condition number
                self.condNumbers.get_new_cond_number(
                    k,
                    self.Ryylocal[k],
                    iter=self.i[k],
                    type='local'
                )
                self.lastCondNumberSaveRyyLocal[k] = self.i[k]
        
        # Consider centralised estimates
        if self.computeCentralised:
            y = self.yHatCentr[k][:, self.i[k], :]  # concise renaming
            yyH = np.einsum('ij,ik->ijk', y, y.conj())
            self.Ryycentr[k], self.Rnncentr[k] = _update_covmats(
                self.Ryycentr[k],
                self.Rnncentr[k],
                yyH,
                vad=self.centrVADframes[self.i[k]]
            )  # update centralised
            if self.saveConditionNumber and\
                self.i[k] - self.lastCondNumberSaveRyyCentr[k] >=\
                    self.saveConditionNumberEvery:  # save condition number
                self.condNumbers.get_new_cond_number(
                    k,
                    self.Ryycentr[k],
                    iter=self.i[k],
                    type='centralised'
                )
                self.lastCondNumberSaveRyyCentr[k] = self.i[k]

    def perform_update(self, k):
        """
        Filter update for DANSE, `for`-loop free.
        GEVD or no GEVD, depending on `self.performGEVD`.
        
        Parameters
        ----------
        k : int
            Node index.
        """
        def _update_w(Ryy, Rnn, refSensorIdx):
            """Helper function for regular MWF-like
            DANSE filter update."""
            # Reference sensor selection vector
            Evect = np.zeros(Ryy.shape[-1])
            Evect[refSensorIdx] = 1

            # Cross-correlation matrix update 
            ryd = np.matmul(Ryy - Rnn, Evect)
            # Update node-specific parameters of node k
            Ryyinv = np.linalg.inv(Ryy)
            w = np.matmul(Ryyinv, ryd[:, :, np.newaxis])
            return w[:, :, 0]  # get rid of singleton dimension

        def _update_w_gevd(Ryy, Rnn, refSensorIdx):
            """Helper function for GEVD-based MWF-like
            DANSE filter update."""
            n = Ryy.shape[-1]
            nFreqs = Ryy.shape[0]
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
            filter_update_fcn = _update_w_gevd
        else:
            filter_update_fcn = _update_w

        if not self.bypassUpdates:
            # Update DANSE filter
            if self.startUpdates[k]:
                self.wTilde[k][:, self.i[k] + 1, :] = filter_update_fcn(
                    self.Ryytilde[k],
                    self.Rnntilde[k],
                    refSensorIdx=self.referenceSensor
                )
                self.nInternalFilterUps[k] += 1  
            # Update centralised filter
            if self.computeCentralised and self.startUpdatesCentr[k]:
                self.wCentr[k][:, self.i[k] + 1, :] = filter_update_fcn(
                    self.Ryycentr[k],
                    self.Rnncentr[k],
                    refSensorIdx=int(
                        np.sum(self.nSensorPerNode[:k]) + self.referenceSensor
                    )
                )
                self.nCentrFilterUps[k] += 1  
            # Update local filter
            if self.computeLocal and self.startUpdatesLocal[k]:
                self.wLocal[k][:, self.i[k] + 1, :] = filter_update_fcn(
                    self.Ryylocal[k],
                    self.Rnnlocal[k],
                    refSensorIdx=self.referenceSensor
                )
                self.nLocalFilterUps[k] += 1

    def update_sro_estimates(self, k, fs):
        """
        Update SRO estimates.
        
        Parameters
        ----------
        k : int
            Node index.
        fs : int or float
            Sampling frequency of current node. 
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

        elif self.estimateSROs == 'DXCPPhaT':
            # DXCP-PhaT-based SRO estimation
            sroOut = sros.dxcpphat_sro_estimation(
                fs=fs,
                fsref=16e3,  # FIXME: HARD-CODED!!
                N=self.DFTsize,
                localSig=self.yTilde[k][
                    :,
                    self.i[k],
                    self.referenceSensor
                ],
                neighboursSig=self.yTilde[k][
                    :,
                    self.i[k],
                    self.nSensorPerNode[k]:
                ],
                refSensorIdx=self.referenceSensor,
            )

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
        dChunk_s, dhatCurr_s = base.get_desired_sig_chunk(
            self.desSigProcessingType,
            w=self.wTilde[k][:, self.i[k] + 1, :],
            y=self.yTildeHat_s[k][:, self.i[k], :],
            win=self.winWOLAsynthesis,
            dChunk=self.d_s[self.idxBegChunk:self.idxEndChunk, k],
            yTD=self.yTilde_s[k][:, self.i[k], :self.nLocalMic[k]],
            normFactWOLA=self.normFactWOLA,
            Ns=self.Ns,
        )
        dChunk_n, dhatCurr_n = base.get_desired_sig_chunk(
            self.desSigProcessingType,
            w=self.wTilde[k][:, self.i[k] + 1, :],
            y=self.yTildeHat_n[k][:, self.i[k], :],
            win=self.winWOLAsynthesis,
            dChunk=self.d_n[self.idxBegChunk:self.idxEndChunk, k],
            yTD=self.yTilde_n[k][:, self.i[k], :self.nLocalMic[k]],
            normFactWOLA=self.normFactWOLA,
            Ns=self.Ns,
        )
        self.dhat[:, self.i[k], k] = dhatCurr  # STFT-domain
        self.dhat_s[:, self.i[k], k] = dhatCurr_s  # STFT-domain
        self.dhat_n[:, self.i[k], k] = dhatCurr_n  # STFT-domain
        # Time-domain
        if self.desSigProcessingType == 'wola':
            self.d[self.idxBegChunk:self.idxEndChunk, k] = dChunk
            self.d_s[self.idxBegChunk:self.idxEndChunk, k] = dChunk_s
            self.d_n[self.idxBegChunk:self.idxEndChunk, k] = dChunk_n
        elif self.desSigProcessingType == 'conv':
            self.d[self.idxEndChunk - self.Ns:self.idxEndChunk, k] = dChunk
            self.d_s[self.idxEndChunk - self.Ns:self.idxEndChunk, k] = dChunk_s
            self.d_n[self.idxEndChunk - self.Ns:self.idxEndChunk, k] = dChunk_n

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
            dChunk_s, dhatCurr_s = base.get_desired_sig_chunk(
                self.desSigProcessingType,
                w=self.wCentr[k][:, self.i[k] + 1, :],
                y=self.yHatCentr_s[k][:, self.i[k], :],
                win=self.winWOLAsynthesis,
                dChunk=self.dCentr_s[self.idxBegChunk:self.idxEndChunk, k],
                yTD=self.yCentr_s[k][:, self.i[k], :self.nLocalMic[k]],
                normFactWOLA=self.normFactWOLA,
                Ns=self.Ns,
            )
            dChunk_n, dhatCurr_n = base.get_desired_sig_chunk(
                self.desSigProcessingType,
                w=self.wCentr[k][:, self.i[k] + 1, :],
                y=self.yHatCentr_n[k][:, self.i[k], :],
                win=self.winWOLAsynthesis,
                dChunk=self.dCentr_n[self.idxBegChunk:self.idxEndChunk, k],
                yTD=self.yCentr_n[k][:, self.i[k], :self.nLocalMic[k]],
                normFactWOLA=self.normFactWOLA,
                Ns=self.Ns,
            )
            self.dHatCentr[:, self.i[k], k] = dhatCurr  # STFT-domain
            self.dHatCentr_s[:, self.i[k], k] = dhatCurr_s  # STFT-domain
            self.dHatCentr_n[:, self.i[k], k] = dhatCurr_n  # STFT-domain
            # Time-domain
            if self.desSigProcessingType == 'wola':
                self.dCentr[self.idxBegChunk:self.idxEndChunk, k] = dChunk
                self.dCentr_s[self.idxBegChunk:self.idxEndChunk, k] = dChunk_s
                self.dCentr_n[self.idxBegChunk:self.idxEndChunk, k] = dChunk_n
            elif self.desSigProcessingType == 'conv':
                self.dCentr[self.idxEndChunk -\
                    self.Ns:self.idxEndChunk, k] = dChunk
                self.dCentr_s[self.idxEndChunk -\
                    self.Ns:self.idxEndChunk, k] = dChunk_s
                self.dCentr_n[self.idxEndChunk -\
                    self.Ns:self.idxEndChunk, k] = dChunk_n
        
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
            dChunk_s, dhatCurr_s = base.get_desired_sig_chunk(
                self.desSigProcessingType,
                w=self.wLocal[k][:, self.i[k] + 1, :],
                y=self.yHatLocal_s[k][:, self.i[k], :],
                win=self.winWOLAsynthesis,
                dChunk=self.dLocal_s[self.idxBegChunk:self.idxEndChunk, k],
                yTD=self.yLocal_s[k][:, self.i[k], :self.nLocalMic[k]],
                normFactWOLA=self.normFactWOLA,
                Ns=self.Ns,
            )
            dChunk_n, dhatCurr_n = base.get_desired_sig_chunk(
                self.desSigProcessingType,
                w=self.wLocal[k][:, self.i[k] + 1, :],
                y=self.yHatLocal_n[k][:, self.i[k], :],
                win=self.winWOLAsynthesis,
                dChunk=self.dLocal_n[self.idxBegChunk:self.idxEndChunk, k],
                yTD=self.yLocal_n[k][:, self.i[k], :self.nLocalMic[k]],
                normFactWOLA=self.normFactWOLA,
                Ns=self.Ns,
            )
            self.dHatLocal[:, self.i[k], k] = dhatCurr  # STFT-domain
            self.dHatLocal_s[:, self.i[k], k] = dhatCurr_s  # STFT-domain
            self.dHatLocal_n[:, self.i[k], k] = dhatCurr_n  # STFT-domain
            # Time-domain
            if self.desSigProcessingType == 'wola':
                self.dLocal[self.idxBegChunk:self.idxEndChunk, k] = dChunk
                self.dLocal_s[self.idxBegChunk:self.idxEndChunk, k] = dChunk_s
                self.dLocal_n[self.idxBegChunk:self.idxEndChunk, k] = dChunk_n
            elif self.desSigProcessingType == 'conv':
                self.dLocal[self.idxEndChunk -\
                    self.Ns:self.idxEndChunk, k] = dChunk
                self.dLocal_s[self.idxEndChunk -\
                    self.Ns:self.idxEndChunk, k] = dChunk_s
                self.dLocal_n[self.idxEndChunk -\
                    self.Ns:self.idxEndChunk, k] = dChunk_n


class TIDANSEvariables(DANSEvariables):
    """
    References
    ----------
    [1] P. Didier, T. Van Waterschoot, S. Doclo and M. Moonem, "Sampling Rate
    Offset Estimation and Compensation for Distributed Adaptive Node-Specific
    Signal Estimation in Wireless Acoustic Sensor Networks," in IEEE Open
    Journal of Signal Processing, doi: 10.1109/OJSP.2023.3243851.

    [2] J. Szurley, A. Bertrand and M. Moonen, "Topology-Independent
    Distributed Adaptive Node-Specific Signal Estimation in Wireless
    Sensor Networks," in IEEE Transactions on Signal and Information
    Processing over Networks, vol. 3, no. 1, pp. 130-144, March 2017,
    doi: 10.1109/TSIPN.2016.2623095.
    """

    def __init__(self, p: base.DANSEparameters, wasn: list[Node]):
        self.import_params(p)           # import base DANSE parameters
        self.init_from_wasn(wasn)       # import WASN parameters
        self.init_for_adhoc_topology()  # TI-DANSE-specific variables
    
    def init_for_adhoc_topology(self):
        """Parameters required for TI-DANSE in ad-hoc WASNs."""
        # `zLocalPrime`: fused local signals (!= partial in-network sum!).
        #       == $\dot{z}_kl[n]'$, following notation from [1].
        self.zLocalPrime = [np.zeros(self.DFTsize) for _ in range(self.nNodes)]
        self.zLocalPrime_s = [np.zeros(self.DFTsize) for _ in range(self.nNodes)]
        self.zLocalPrime_n = [np.zeros(self.DFTsize) for _ in range(self.nNodes)]
        # `zLocalPrimeBuffer`: fused local signals after WOLA overlap
        # (last Ns samples are influence by the window).
        self.zLocalPrimeBuffer = [np.array([]) for _ in range(self.nNodes)]
        self.zLocalPrimeBuffer_s = [np.array([]) for _ in range(self.nNodes)]
        self.zLocalPrimeBuffer_n = [np.array([]) for _ in range(self.nNodes)]
        # `eta`: in-network sum.
        #       == $\dot{\eta}[n]$ extrapolating notation from [1] & [2].
        self.eta = [np.array([]) for _ in range(self.nNodes)]
        self.eta_s = [np.array([]) for _ in range(self.nNodes)]
        self.eta_n = [np.array([]) for _ in range(self.nNodes)]
        # `etaMk`: in-network sum minus local fused signal.
        #       == $\dot{\eta}_{-k}[n]$ extrapolating notation from [1] & [2].
        self.etaMk = [np.array([]) for _ in range(self.nNodes)]
        self.etaMk_s = [np.array([]) for _ in range(self.nNodes)]
        self.etaMk_n = [np.array([]) for _ in range(self.nNodes)]
        #
        self.treeFormationCounter = 0  # counting the number of tree-formations
        self.currentWasnTreeObj = None  # current tree WASN object

    def update_up_downstream_neighbors(
            self,
            newWasnObj: WASN,
            plotit: bool=False,
            ax=None,
            scatterSize=300
        ):
        """
        Updates the neighbors lists based on the new WASN (after a new tree
        formation -- i.e., for a new update in sequential node-updating
        TI-DANSE).
        """
        self.upstreamNeighbors =\
            [node.upstreamNeighborsIdx for node in newWasnObj.wasn]
        self.downstreamNeighbors =\
            [node.downstreamNeighborsIdx for node in newWasnObj.wasn]
        self.treeFormationCounter += 1  # update counter too
        self.currentWasnTreeObj = newWasnObj
        if plotit:
            ax.clear()
            newWasnObj.plot_me(ax, scatterSize=scatterSize)


    def ti_fusion(self, k, tCurr, fs):
        """
        Fuses local sensor observations into a signal $z'_k$.
        
        Parameters
        ----------
        k : int
            Node index.
        tCurr : float
            Current time instant [s].
        fs : float or int
            Current node's sampling frequency [Hz].
        """
        
        # Extract correct frame of local signals
        ykFrame = base.local_chunk_for_broadcast(
            self.yin[k],
            tCurr,
            fs,
            self.DFTsize
        )
        ykFrame_s = base.local_chunk_for_broadcast(
            self.cleanSpeechSignalsAtNodes[k],
            tCurr,
            fs,
            self.DFTsize
        )  # [useful for SNR computation]
        ykFrame_n = base.local_chunk_for_broadcast(
            self.cleanNoiseSignalsAtNodes[k],
            tCurr,
            fs,
            self.DFTsize
        )  # [useful for SNR computation]

        if len(ykFrame) < self.DFTsize:
            print('Cannot perform compression: not enough local samples.')

        elif self.broadcastType == 'wholeChunk':
            # Time-domain chunk-wise compression
            _, self.zLocalPrimeBuffer[k] = self.ti_compression_whole_chunk(
                k,
                ykFrame,
                zForSynthesis=self.zLocalPrimeBuffer[k]
            )
            _, self.zLocalPrimeBuffer_s[k] = self.ti_compression_whole_chunk(
                k,
                ykFrame_s,
                zForSynthesis=self.zLocalPrimeBuffer_s[k]
            )
            _, self.zLocalPrimeBuffer_n[k] = self.ti_compression_whole_chunk(
                k,
                ykFrame_n,
                zForSynthesis=self.zLocalPrimeBuffer_n[k]
            )
            
            # _, self.zLocalPrimeBuffer[k] = base.danse_compression_whole_chunk(
            #     ykFrame,
            #     self.wTildeExt[k][:, :ykFrame.shape[-1]],
            #     h=self.winWOLAanalysis,
            #     f=self.winWOLAsynthesis,
            #     zqPrevious=self.zLocalPrimeBuffer[k]
            # )
        else:
            raise ValueError('[NYI]')  # TODO:
    
    def ti_compute_partial_sum(self, k):
        """
        Computes the partial in-network sum at the current node.

        Parameters
        ----------
        k : int
            Node index.
        """
        # Process WOLA buffer: use the valid part of the overlap-added signal
        # # TODO: this is for whole-chunk broadcasting only
        # if self.i[k] == 0:
        #     self.zLocalPrime[k] = copy.deepcopy(self.zLocalPrimeBuffer[k])
        # else:

        self.zLocalPrime[k] = np.concatenate((
            self.zLocalPrime[k][-(self.DFTsize - self.Ns):],
            self.zLocalPrimeBuffer[k][:self.Ns]
        ))
        self.zLocalPrime_s[k] = np.concatenate((
            self.zLocalPrime_s[k][-(self.DFTsize - self.Ns):],
            self.zLocalPrimeBuffer_s[k][:self.Ns]
        ))
        self.zLocalPrime_n[k] = np.concatenate((
            self.zLocalPrime_n[k][-(self.DFTsize - self.Ns):],
            self.zLocalPrimeBuffer_n[k][:self.Ns]
        ))

        # Compute partial sum
        self.zLocal[k] = copy.deepcopy(self.zLocalPrime[k])
        self.zLocal_s[k] = copy.deepcopy(self.zLocalPrime_s[k])
        self.zLocal_n[k] = copy.deepcopy(self.zLocalPrime_n[k])
        for l in self.upstreamNeighbors[k]:
            if len(self.zLocal[l] > 0):  # do not consider empty buffers
                self.zLocal[k] += self.zLocal[l]
                self.zLocal_s[k] += self.zLocal_s[l]
                self.zLocal_n[k] += self.zLocal_n[l]

        # At the root, the sum is not partial, it is complete
        if len(self.downstreamNeighbors[k]) == 0:
            self.eta[k] = copy.deepcopy(self.zLocal[k])
            self.eta_s[k] = copy.deepcopy(self.zLocal_s[k])
            self.eta_n[k] = copy.deepcopy(self.zLocal_n[k])
            self.etaMk[k] = self.eta[k] - self.zLocalPrime[k]
            self.etaMk_s[k] = self.eta_s[k] - self.zLocalPrime_s[k]
            self.etaMk_n[k] = self.eta_n[k] - self.zLocalPrime_n[k]

    def ti_relay_innetwork_sum_upstream(self, k):
        """
        Relay in-network sum coming from root to upstream neighbors.

        Parameters
        ----------
        k : int
            Node index.
        """
        for l in self.upstreamNeighbors[k]:
            self.eta[l] = copy.deepcopy(self.eta[k])  # relay
            self.eta_s[l] = copy.deepcopy(self.eta_s[k])  # relay
            self.eta_n[l] = copy.deepcopy(self.eta_n[k])  # relay
            self.etaMk[l] = self.eta[l] - self.zLocalPrime[l]  # $\eta_{-k}$
            self.etaMk_s[l] = self.eta_s[l] - self.zLocalPrime_s[l]  # $\eta_{-k}$
            self.etaMk_n[l] = self.eta_n[l] - self.zLocalPrime_n[l]  # $\eta_{-k}$

    def ti_update_and_estimate(self, k, tCurr, fs, bypassUpdateEventMat=False):
        """
        Performs an update of the DANSE filter coefficients at node `k` and
        estimates the desired signal(s).

        Parameters
        ----------
        k : int
            Node index.
        tCurr : float
            Current time instant [s].
        fs : float or int
            Current node's sampling frequency [Hz].
        bypassUpdateEventMat : bool
            If true, bypass filter udpate.
            (but still compute the desired signal estimate!)
        """
        
        if self.firstDANSEupdateRefSensor is None and\
            self.nInternalFilterUps[k] == 0:
            # Save first update instant (for, e.g., SRO plot)
            self.firstDANSEupdateRefSensor = tCurr

        # vvvvvvvvvvvv TODO? vvvvvvvvvvvv
        # # Process buffers
        # self.process_incoming_signals_buffers(k, tCurr)
        # # Wipe local buffers
        # self.zBuffer[k] = [np.array([])\
        #     for _ in range(len(self.neighbors[k]))]
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # Construct `\tilde{y}_k` in frequency domain
        self.ti_build_ytilde(k, tCurr, fs)
        # Compute current VAD
        self.compute_vad(k)
        
        # Consider local / centralised estimation(s)
        if self.computeCentralised:
            self.build_ycentr(tCurr, fs, k)
        if self.computeLocal:  # extract local info from `\tilde{y}_k`
            self.yLocal[k][:, self.i[k], :] =\
                self.yTilde[k][:, self.i[k], :self.nSensorPerNode[k]]
            self.yLocal_s[k][:, self.i[k], :] =\
                self.yTilde_s[k][:, self.i[k], :self.nSensorPerNode[k]]
            self.yLocal_n[k][:, self.i[k], :] =\
                self.yTilde_n[k][:, self.i[k], :self.nSensorPerNode[k]]
            #
            self.yHatLocal[k][:, self.i[k], :] =\
                self.yTildeHat[k][:, self.i[k], :self.nSensorPerNode[k]]
            self.yHatLocal_s[k][:, self.i[k], :] =\
                self.yTildeHat_s[k][:, self.i[k], :self.nSensorPerNode[k]]
            self.yHatLocal_n[k][:, self.i[k], :] =\
                self.yTildeHat_n[k][:, self.i[k], :self.nSensorPerNode[k]]

        # vvvvvvvvvvvv TODO: vvvvvvvvvvvv
        # # Account for buffer flags
        # skipUpdate = self.compensate_sros(k, tCurr)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # Ryy and Rnn updates (including centralised / local, if needed)
        self.spatial_covariance_matrix_update(k)
        # Check quality of covariance matrix estimates 
        self.check_covariance_matrices(k)

        # if not skipUpdate:
        # If covariance matrices estimates are full-rank, update filters
        if not bypassUpdateEventMat:
            self.perform_update(k)
            # ^^^ depends on outcome of `check_covariance_matrices()`.
        else:
            # Do not update the filter coefficients
            self.wTilde[k][:, self.i[k] + 1, :] =\
                self.wTilde[k][:, self.i[k], :]
            if self.computeCentralised:
                self.wCentr[k][:, self.i[k] + 1, :] =\
                    self.wCentr[k][:, self.i[k], :]
            if self.computeLocal:
                self.wLocal[k][:, self.i[k] + 1, :] =\
                    self.wLocal[k][:, self.i[k], :]
            # if skipUpdate:
            #     print(f'Node {k+1}: {self.i[k]+1}^th update skipped.')
        if self.bypassUpdates:
            print('!! User-forced bypass of filter coefficients updates !!')

        # Update external filters (for broadcasting)
        self.ti_update_external_filters(k, tCurr)
        # self.update_external_filters(k, tCurr)

        # vvvvvvvvvvvv TODO: vvvvvvvvvvvv
        # # Update SRO estimates
        # self.update_sro_estimates(k, fs)
        # # Update phase shifts for SRO compensation
        # if self.compensateSROs:
        #     self.build_phase_shifts_for_srocomp(k)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # Compute desired signal chunk estimate
        self.get_desired_signal(k)

        # Update iteration index
        self.i[k] += 1

    def ti_build_ytilde(self, k, tCurr, fs):        
        """
        Builds the observation vector used for the TI-DANSE filter update.
        
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
        # Extract current local data chunk
        yLocalCurr, self.idxBegChunk, self.idxEndChunk =\
            base.local_chunk_for_update(
                self.yin[k],
                tCurr,
                fs,
                bd=self.broadcastType,
                Ndft=self.DFTsize,
                Ns=self.Ns
            )
        yLocalCurr_s, _, _ =\
            base.local_chunk_for_update(
                self.cleanSpeechSignalsAtNodes[k],
                tCurr,
                fs,
                bd=self.broadcastType,
                Ndft=self.DFTsize,
                Ns=self.Ns
            )
        yLocalCurr_n, _, _ =\
            base.local_chunk_for_update(
                self.cleanNoiseSignalsAtNodes[k],
                tCurr,
                fs,
                bd=self.broadcastType,
                Ndft=self.DFTsize,
                Ns=self.Ns
            )

        # Build full available observation vector
        yTildeCurr = np.concatenate(
            (yLocalCurr, self.etaMk[k][:, np.newaxis]),
            axis=1
        )
        yTildeCurr_s = np.concatenate(
            (yLocalCurr_s, self.etaMk_s[k][:, np.newaxis]),
            axis=1
        )
        yTildeCurr_n = np.concatenate(
            (yLocalCurr_n, self.etaMk_n[k][:, np.newaxis]),
            axis=1
        )
        self.yTilde[k][:, self.i[k], :] = yTildeCurr
        self.yTilde_s[k][:, self.i[k], :] = yTildeCurr_s
        self.yTilde_n[k][:, self.i[k], :] = yTildeCurr_n
        # Go to frequency domain
        yTildeHatCurr = 1 / self.normFactWOLA * np.fft.fft(
            self.yTilde[k][:, self.i[k], :] *\
                self.winWOLAanalysis[:, np.newaxis],
            self.DFTsize,
            axis=0
        )
        yTildeHatCurr_s = 1 / self.normFactWOLA * np.fft.fft(
            self.yTilde_s[k][:, self.i[k], :] *\
                self.winWOLAanalysis[:, np.newaxis],
            self.DFTsize,
            axis=0
        )
        yTildeHatCurr_n = 1 / self.normFactWOLA * np.fft.fft(
            self.yTilde_n[k][:, self.i[k], :] *\
                self.winWOLAanalysis[:, np.newaxis],
            self.DFTsize,
            axis=0
        )
        # Keep only positive frequencies
        self.yTildeHat[k][:, self.i[k], :] = yTildeHatCurr[:self.nPosFreqs, :]
        self.yTildeHat_s[k][:, self.i[k], :] = yTildeHatCurr_s[:self.nPosFreqs, :]
        self.yTildeHat_n[k][:, self.i[k], :] = yTildeHatCurr_n[:self.nPosFreqs, :]

    def ti_compression_whole_chunk(self, k, yq: np.ndarray, zForSynthesis: np.ndarray):
        """
        Performs local signals compression in the frequency domain
        for TI-DANSE.

        Parameters
        ----------
        k : int
            Node index.
        yq : [N x nSensors] np.ndarray (float)
            Local sensor signals.
        zForSynthesis : np.ndarray[float]
            Previous local buffer chunk for overlap-add synthesis.

        Returns
        -------
        zqHat : [N/2 x 1] np.ndarray (complex)
            Frequency-domain compressed signal for current frame.
        zq : [N x 1] np.ndarray (float)
            Time-domain latest WOLA chunk of compressed signal (after OLA).
        """

        # Transfer local observations to frequency domain
        DFTorder = yq.shape[0]  # DFT order

        yqHat = np.fft.fft(
            yq * self.winWOLAanalysis[:, np.newaxis], DFTorder, axis=0
        )
        # Keep only positive frequencies
        yqHat = yqHat[:int(DFTorder/2 + 1), :]
        # Compute compression vector
        # allIdx = np.arange(self.wTildeExt[k].shape[-1])
        if not self.startUpdates[k]:
            p = self.wTildeExt[k][:, :yq.shape[-1]]
        else:
            # If the external filters have started updating, we must 
            # transform by the inverse of the part of the estimator
            # corresponding to the in-network sum.
            p = self.wTildeExt[k][:, :yq.shape[-1]] / self.wTildeExt[k][:, -1:]
            # p = self.wTildeExt[k][:, :yq.shape[-1]]
        # Apply linear combination to form compressed signal.
        zqHat = np.einsum('ij,ij->i', p.conj(), yqHat)

        # WOLA synthesis stage
        if zForSynthesis is not None:
            # IDFT
            zqCurr = base.back_to_time_domain(zqHat, DFTorder, axis=0)
            zqCurr = np.real_if_close(zqCurr)
            zqCurr *= self.winWOLAsynthesis    # multiply by synthesis window

            if not np.any(zForSynthesis):
                # No previous frame, keep current frame
                zq = zqCurr
            else:
                # Overlap-add
                zq = np.zeros(DFTorder)
                # TODO: consider case with multiple neighbor nodes
                # (`len(zqPrevious) > 1`).
                zq[:(DFTorder // 2)] = zForSynthesis[-(DFTorder // 2):]
                zq += zqCurr
        else:
            zq = None
        
        return zqHat, zq
    
    def ti_update_external_filters(self, k, t):
        """
        Update external filters for relaxed filter update.
        To be used when using simultaneous or asynchronous node-updating.
        When using sequential node-updating, do not differential between
        internal (`self.wTilde`) and external filters. 
        For TI-DANSE.
        
        Parameters
        ----------
        k : int
            Receiving node index.
        t : float
            Current time instant [s].
        """

        if self.noExternalFilterRelaxation:
            # No relaxation (i.e., no "external" filters)
            self.wTildeExt[k] = self.wTilde[k][:, self.i[k] + 1, :]
        else:
            # Simultaneous or asynchronous node-updating
            if 'seq' not in self.nodeUpdating:
                # Relaxed external filter update
                self.wTildeExt[k] = self.expAvgBeta[k] * self.wTildeExt[k] +\
                    (1 - self.expAvgBeta[k]) *  self.wTildeExtTarget[k]
                # Update targets
                if t - self.lastExtFiltUp[k] >= self.timeBtwExternalFiltUpdates:
                    self.wTildeExtTarget[k] = (1 - self.alphaExternalFilters) *\
                        self.wTildeExtTarget[k] + self.alphaExternalFilters *\
                        self.wTilde[k][:, self.i[k] + 1, :]
                    # Update last external filter update instant [s]
                    self.lastExtFiltUp[k] = t
                    if self.printoutsAndPlotting.printout_externalFilterUpdate:    # inform user
                        print(f't={np.round(t, 3)}s -- UPDATING EXTERNAL FILTERS for node {k+1} (scheduled every [at least] {self.timeBtwExternalFiltUpdates}s)')
            # Sequential node-updating
            else:
                self.wTildeExt[k] = self.wTilde[k][:, self.i[k] + 1, :]