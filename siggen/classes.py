from dataclasses import dataclass, field
import numpy as np
import itertools

@dataclass
class AcousticScenarioParameters:
    rd : np.ndarray = np.array([5, 5, 5])  # room dimensions [m]
    fs : float = 16000.     # base sampling frequency [Hz]
    t60 : float = 0.        # reverberation time [s]
    minDistToWalls : float = 0.33   # minimum distance between elements and room walls [m]
    #
    interSensorDist : float = 0.1   # distance separating microphones
    arrayGeometry: str = 'grid3d'   # microphone array geometry (only used if numSensorPerNode > 1)
    # ^^^ possible values: 'linear', 'radius', 'grid3d'
    #
    lenRIR : int = 2**10    # length of RIRs [samples]
    sigDur : float = 5.     # signals duration [s]
    #
    nDesiredSources : int = 1   # number of desired sources
    nNoiseSources : int = 1     # number of undesired (noise) sources
    desiredSignalFile: list[str] = field(default_factory=list)  # list of paths to desired signal file(s)
    noiseSignalFile: list[str] = field(default_factory=list)    # list of paths to noise signal file(s)
    baseSNR: int = 0                        # [dB] SNR between dry desired signals and dry noise
    #
    VADenergyFactor: float = 4000           # VAD energy factor (VAD threshold = max(energy signal)/VADenergyFactor)
    VADwinLength: float = 40e-3             # [s] VAD window length
    #
    nNodes : int = 1        # number of nodes in scenario
    nSensorPerNode : list[int] = field(default_factory=list)    # number of sensors per node
    seed : int = 12345      # random generators seed

    def __post_init__(self):
        self.sensorToNodeIndices = np.array(list(itertools.chain(*[[ii] * self.nSensorPerNode[ii]\
            for ii in range(len(self.nSensorPerNode))])))  # itertools trick from https://stackoverflow.com/a/953097
        # Generate a reference for this experiment
        self.id = f'J{self.nNodes}Mk{list(self.nSensorPerNode)}Nn{self.nNoiseSources}Nd{self.nDesiredSources}T60_{int(self.t60)*1e3}ms'
