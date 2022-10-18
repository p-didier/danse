
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

p = AcousticScenarioParameters(
    rd=np.array([5, 5, 5]),
    fs=16000,
    t60=0.2,
    nNodes=2,
    nSensorPerNode=[2, 1],
    desiredSignalFile=[f'{SIGNALSPATH}/01_speech/{file}' for file in ['speech1.wav', 'speech2.wav']],
    noiseSignalFile=[f'{SIGNALSPATH}/02_noise/{file}' for file in ['whitenoise_signal_1.wav', 'whitenoise_signal_2.wav']],
)

def main():

    room, vad = ut.build_room(p)

    ut.plot_mic_sigs(room, vad)

    stop = 1


if __name__ == '__main__':
    sys.exit(main())