
import sys
from pathlib import Path, PurePath
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}')

PATHTOROOT = pathToRoot
SIGNALSPATH = f'{PATHTOROOT}/02_data/00_raw_signals'

import numpy as np
from danse.siggen.classes import AcousticScenarioParameters
import danse.siggen.utils as ut
from dataclasses import dataclass, field

@dataclass
class TestParameters:
    pathToASC : str = 'C:/Users/pdidier/Dropbox/PC/Documents/sounds-phd/02_data/01_acoustic_scenarios/tests/J2Mk[1_1]_Ns1_Nn1/AS2_anechoic'
    desiredSignalFile: list[str] = field(default_factory=list)            # list of paths to desired signal file(s)
    noiseSignalFile: list[str] = field(default_factory=list)              # list of paths to noise signal file(s)
    VADenergyFactor: float = 4000           # VAD energy factor (VAD threshold = max(energy signal)/VADenergyFactor)
    VADwinLength: float = 40e-3             # [s] VAD window length
    selfnoiseSNR: int = -50                 # [dB] microphone self-noise SNR
    referenceSensor: int = 0                # Index of the reference sensor at each node
    asc: AcousticScenarioParameters = AcousticScenarioParameters()

def main():

    room = ut.build_room(p)

    ut.plot_mic_sigs(room)

    stop = 1


if __name__ == '__main__':
    sys.exit(main())
