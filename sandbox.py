from dataclasses import dataclass, field
import sys
import matplotlib.pyplot as plt
from testutils.siggen import *
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}')

PATHTOROOT = pathToRoot
SIGNALSPATH = f'{PATHTOROOT}/02_data/00_raw_signals'

@dataclass
class TestParameters:
    pathToASC : str = 'C:/Users/pdidier/Dropbox/PC/Documents/sounds-phd/02_data/01_acoustic_scenarios/tests/J2Mk[1_1]_Ns1_Nn1/AS2_anechoic'
    desiredSignalFile: list[str] = field(default_factory=list)            # list of paths to desired signal file(s)
    noiseSignalFile: list[str] = field(default_factory=list)              # list of paths to noise signal file(s)
    fs : float = 16000.  # sampling frequency [Hz]
    sigDur : float = 1.  # signals duration [s]
    VADenergyFactor: float = 4000           # VAD energy factor (VAD threshold = max(energy signal)/VADenergyFactor)
    VADwinLength: float = 40e-3             # [s] VAD window length
    baseSNR: int = 0                        # [dB] SNR between dry desired signals and dry noise
    selfnoiseSNR: int = -50                 # [dB] microphone self-noise SNR
    randSeed: int = 12345                   # random generator(s) seed
    referenceSensor: int = 0                # Index of the reference sensor at each node


def main():

    params = TestParameters(
        sigDur=10,
        desiredSignalFile=[f'{SIGNALSPATH}/01_speech/{file}' for file in ['speech1.wav', 'speech2.wav']],
        noiseSignalFile=[f'{SIGNALSPATH}/02_noise/{file}' for file in ['whitenoise_signal_1.wav', 'whitenoise_signal_2.wav']],
    )

    run(params)

    stop = 1


def run(params: TestParameters):

    # Load acoustic scenario
    asc = load_asc(params.pathToASC)

    # Create signals
    sigs = generate_signals(asc)

    stop = 1


if __name__ == '__main__':
    sys.exit(main())