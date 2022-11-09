from dataclasses import dataclass, field
import numpy as np
import itertools

@dataclass
class AcousticScenarioParameters:
    rd: np.ndarray = np.array([5, 5, 5])  # room dimensions [m]
    fs: float = 16000.     # base sampling frequency [Hz]
    t60: float = 0.        # reverberation time [s]
    minDistToWalls: float = 0.33   # minimum distance between elements and room walls [m]
    #    
    referenceSensor: int = 0    # Index of the reference sensor at each node
    interSensorDist: float = 0.1   # distance separating microphones
    arrayGeometry: str = 'grid3d'   # microphone array geometry (only used if numSensorPerNode > 1)
    # ^^^ possible values: 'linear', 'radius', 'grid3d'
    #
    lenRIR: int = 2**10    # length of RIRs [samples]
    sigDur: float = 5.     # signals duration [s]
    #
    nDesiredSources: int = 1   # number of desired sources
    nNoiseSources: int = 1     # number of undesired (noise) sources
    desiredSignalFile: list[str] = field(default_factory=list)  # list of paths to desired signal file(s)
    noiseSignalFile: list[str] = field(default_factory=list)    # list of paths to noise signal file(s)
    baseSNR: int = 5                        # [dB] SNR between dry desired signals and dry noise
    #
    VADenergyFactor: float = 4000           # VAD energy factor (VAD threshold = max(energy signal)/VADenergyFactor)
    VADwinLength: float = 40e-3             # [s] VAD window length
    #
    nNodes: int = 1        # number of nodes in scenario
    nSensorPerNode: list[int] = field(default_factory=list)    # number of sensors per node


@dataclass
class WASNparameters(AcousticScenarioParameters):
    SROperNode: np.ndarray = np.array([0])
    topologyType: str = 'fully-connected'       # type of WASN topology
                # ^^^ valid values: "fully-connected"; TODO: add some
    
    def __post_init__(self):
        # Dimensionality checks
        if len(self.SROperNode) != self.nNodes:
            if all(self.SROperNode == self.SROperNode[0]):
                print(f'Automatically setting all SROs to the only value provided ({self.SROperNode[0]} PPM).')
                self.SROperNode = np.full(self.nNodes, fill_value=self.SROperNode[0])
            else:
                raise ValueError(f'The number of SRO values ({len(self.SROperNode)}) does not correspond to the number of nodes in the WASN ({self.nNodes}).')
        # Explicitly derive sensor-to-node indices
        self.sensorToNodeIndices = np.array(list(itertools.chain(*[[ii] * self.nSensorPerNode[ii]\
            for ii in range(len(self.nSensorPerNode))])))  # itertools trick from https://stackoverflow.com/a/953097
        # Sampling rate per node
        self.fsPerNode = self.fs * (1 + self.SROperNode * 1e-6)

@dataclass
class Node:
    nSensors: int = 1
    sro: float = 0.
    fs: float = 16000.
    cleanspeech: np.ndarray = np.array([])  # mic. signals if no noise present
    data: np.ndarray = np.array([])  # mic. signals
    enhancedData: np.ndarray = np.array([]) # signals after enhancement
    enhancedData_c: np.ndarray = np.array([]) # after CENTRALISED enhancement
    enhancedData_l: np.ndarray = np.array([]) # after LOCAL enhancement
    timeStamps: np.ndarray = np.array([])
    neighborsIdx: list[int] = field(default_factory=list)
    vad: np.ndarray = np.array([])
    beta: float = 1.    # exponential averaging (forgetting factor)
                        # for Ryy and Rnn updates at this node

    def __post_init__(self):
        # Combined VAD
        self.vadCombined = np.array(
            [1 if sum(self.vad[ii, :]) > 0 else 0\
                for ii in range(self.vad.shape[0])]
        )
        # Combined wet clean speeches
        self.cleanspeechCombined = np.sum(self.cleanspeech, axis=1)
