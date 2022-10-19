import numpy as np
from dataclasses import dataclass, field
from danse.siggen.classes import *

@dataclass
class Hyperparameters:
    # Efficiency
    efficientSpSBC: bool = True    # if True, perform efficient sample-per-sample broadcast
                                    # by adapting the DANSE-events creation mechanism.
    # Printouts
    printout_eventsParser: bool = True      # controls printouts in `events_parser()` function
    printout_eventsParserNoBC: bool = False     # if True, do not print out the broadcast events in the event parser
    # Other
    bypassFilterUpdates: bool = False   # if True, do not update filters

@dataclass
class DANSEparameters(Hyperparameters):
    referenceSensor: int = 0    # index of reference sensor at each node
    DFTsize: int = 1024    # DFT size
    WOLAovlp: float = .5   # WOLA window overlap [*100%]
    broadcastLength: int = 1    # number of samples to be broadcasted at a time
    broadcastType: str = 'wholeChunk_td'    # type of broadcast
        # -- 'wholeChunk_td': broadcast whole chunks of compressed signals in the time-domain,
        # -- 'wholeChunk_fd': broadcast whole chunks of compressed signals in the WOLA-domain,
        # -- 'fewSamples_td': linear-convolution approximation of WOLA compression process, broadcast L ≪ Ns samples at a time.
    winWOLAanalysis: np.ndarray = np.sqrt(np.hanning(DFTsize))      # WOLA analysis window
    winWOLAsynthesis: np.ndarray = np.sqrt(np.hanning(DFTsize))     # WOLA synthesis window
    # T(z)-approximation | Sample-wise broadcasts
    updateTDfilterEvery: float = 1.        # [s] duration of pause between two consecutive time-domain filter updates.
    # SROs
    compensateSROs: bool = False    # if True, compensate for SROs
    # General
    performGEVD: bool = True    # if True, perform GEVD
    GEVDrank: int = 1           # GEVD rank
    t_expAvg50p: float = 2.     # [s] Time in the past at which the value is weighted by 50% via exponential averaging

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
        self.nPosFreqs = int(self.DFTsize // 2 + 1)  # number of frequency lines (only positive frequencies)
        self.numIterations = int((wasn[0].data.shape[0] - self.DFTsize) / self.Ns) + 1

        bufferFlags = []
        dimYTilde = np.zeros(nNodes, dtype=int)   # dimension of \tilde{y}_k (== M_k + |\mathcal{Q}_k|)
        phaseShiftFactors = []
        Rnntilde = []   # autocorrelation matrix when VAD=0 - using full-observations vectors (also data coming from neighbors)
        Ryytilde = []   # autocorrelation matrix when VAD=1 - using full-observations vectors (also data coming from neighbors)
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
            bufferFlags.append(np.zeros((self.numIterations, nNeighbors)))    # init all buffer flags at 0 (assuming no over- or under-flow)
            #
            dimYTilde[k] = wasn[k].nSensors + nNeighbors
            #
            phaseShiftFactors.append(np.zeros(dimYTilde[k]))   # initiate phase shift factors as 0's (no phase shift)
            #
            sliceTilde = np.finfo(float).eps *\
                (np.random.random((dimYTilde[k], dimYTilde[k])) +\
                    1j * np.random.random((dimYTilde[k], dimYTilde[k]))) 
            Rnntilde.append(np.tile(sliceTilde, (self.nPosFreqs, 1, 1)))                    # noise only
            Ryytilde.append(np.tile(sliceTilde, (self.nPosFreqs, 1, 1)))                    # speech + noise
            #
            t[:, k] = wasn[k].timeStamps
            #
            wtmp = np.zeros((2 * self.DFTsize - 1, wasn[k].nSensors))
            wtmp[self.DFTsize, 0] = 1   # initialize time-domain (real-valued) filter as Dirac for first sensor signal
            wIR.append(wtmp)
            wtmp = np.zeros((self.nPosFreqs, self.numIterations + 1, dimYTilde[k]), dtype=complex)
            wtmp[:, :, 0] = 1   # initialize filter as a selector of the unaltered first sensor signal
            wTilde.append(wtmp)
            wtmp = np.zeros((self.nPosFreqs, wasn[k].nSensors), dtype=complex)
            wtmp[:, 0] = 1   # initialize filter as a selector of the unaltered first sensor signal
            wTildeExt.append(wtmp)
            wTildeExtTarget.append(wtmp)
            #
            yyH.append(np.zeros((self.numIterations, self.DFTsize, dimYTilde[k], dimYTilde[k]), dtype=complex))
            yyHuncomp.append(np.zeros((self.numIterations, self.DFTsize, dimYTilde[k], dimYTilde[k]), dtype=complex))
            yTilde.append(np.zeros((self.DFTsize, self.numIterations, dimYTilde[k])))
            yTildeHat.append(np.zeros((self.nPosFreqs, self.numIterations, dimYTilde[k]), dtype=complex))
            yTildeHatUncomp.append(np.zeros((self.nPosFreqs, self.numIterations, dimYTilde[k]), dtype=complex))
            #
            if 'td' in self.broadcastType:
                z.append(np.empty((self.DFTsize, 0), dtype=float))
            elif self.broadcastType == 'wholeChunk_fd':
                z.append(np.empty((self.nPosFreqs, 0), dtype=complex))
            zBuffer.append([np.array([]) for _ in range(nNeighbors)])
            zLocal.append(np.array([]))

        # Create fields
        self.bufferFlags = bufferFlags
        self.DANSEiter = np.zeros(nNodes, dtype=int)
        self.dimYTilde = dimYTilde
        self.expAvgBeta = [ii.beta for ii in wasn]
        self.flagIterations = [[] for _ in range(nNodes)]
        self.flagInstants = [[] for _ in range(nNodes)]
        self.neighbors = [ii.neighborsIdx for ii in wasn]
        self.nInternalFilterUpdates = np.zeros(nNodes)
        self.numUpdatesRyy = np.zeros(nNodes, dtype=int)
        self.numUpdatesRnn = np.zeros(nNodes, dtype=int)
        self.oVADframes = np.zeros(self.numIterations)
        self.phaseShiftFactors = phaseShiftFactors
        self.phaseShiftFactorThroughTime = np.zeros((self.numIterations))
        self.previousTDfilterUpdate = np.zeros(nNodes)
        self.Rnntilde = Rnntilde
        self.Ryytilde = Ryytilde
        self.startUpdates = np.full(shape=(nNodes,), fill_value=False)
        self.timeInstants = t
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
            fullVAD[:, k] = wasn[k].vad[:, 0]  # <-- multiple desired sources case not considered TODO:
        self.fullVAD = fullVAD

        return self


@dataclass
class DANSEoutputs:
    pass


@dataclass
class DANSEeventInstant:
    t: float = 0.   # event time instant [s]
    nodes: np.ndarray = np.array([0])   # node(s) concerned
    type: list[str] = field(default_factory=list)   # event type

    def __post_init__(self):
        self.nEvents = len(self.nodes)