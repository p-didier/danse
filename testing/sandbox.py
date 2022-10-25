
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
import random
from danse.siggen.classes import *
import danse.siggen.utils as sig_ut
import danse.danse_toolbox.d_base as base
import danse.danse_toolbox.d_core as core
from danse.danse_toolbox.d_classes import *
import danse.danse_toolbox.d_post as postproc
from dataclasses import dataclass

@dataclass
class TestParameters:
    # TODO: vvv self-noise
    selfnoiseSNR: int = -50 # [dB] microphone self-noise SNR
    #
    referenceSensor: int = 0    # Index of the reference sensor at each node
    #
    wasn: WASNparameters = WASNparameters(
        sigDur=5
    )
    danseParams: DANSEparameters = DANSEparameters()
    #
    seed: int = 12345

    def __post_init__(self):
        np.random.seed(self.seed)  # set random seed
        random.seed(self.seed)  # set random seed
        #
        self.testid = f'J{self.wasn.nNodes}Mk{list(self.wasn.nSensorPerNode)}\
            Nn{self.wasn.nNoiseSources}Nd{self.wasn.nDesiredSources}T60_\
            {int(self.wasn.t60)*1e3}ms'
        # Check consistency
        if self.danseParams.nodeUpdating == 'sym' and\
            any(self.wasn.SROperNode != 0):
            raise ValueError('Simultaneous node-updating impossible in\
                the presence of SROs.')


def main():

    p = TestParameters(
        selfnoiseSNR=-99,  # TODO:
        wasn=WASNparameters(
            rd=np.array([5, 5, 5]),
            fs=16000,
            t60=0.2,
            nNodes=2,
            nSensorPerNode=[1,1],
            desiredSignalFile=[f'{SIGNALSPATH}/01_speech/{file}'\
                for file in [
                    'speech1.wav',
                    'speech2.wav'
                ]],
            noiseSignalFile=[f'{SIGNALSPATH}/02_noise/{file}'\
                for file in [
                    'whitenoise_signal_1.wav',
                    'whitenoise_signal_2.wav'
                ]],
            # SROperNode=np.array([0, 50])
            SROperNode=np.array([0, 0])
        ),
        danseParams=DANSEparameters(
            referenceSensor=0,
            DFTsize=1024,
            WOLAovlp=.5,
            nodeUpdating='seq',
            # nodeUpdating='asy',
            # broadcastType='fewSamples',
            broadcastType='wholeChunk',
            estimateSROs='CohDrift',
            # compensateSROs=True,
            compensateSROs=False,
            cohDrift=CohDriftParameters(
                loop='open',
                alpha=0.99
            )
        )
    )

    room, vad, wetSpeechAtRefSensor = sig_ut.build_room(p.wasn)
    # sig_ut.plot_mic_sigs(room, vad)  # <-- plot signals

    # Build WASN (asynchronicities, topology)
    wasn = sig_ut.build_wasn(room, vad, wetSpeechAtRefSensor, p.wasn)

    # DANSE
    out = danse_it_up(wasn, p)
    stop = 1


def danse_it_up(wasn: list[Node], p: TestParameters):
    """
    Container function for prepping signals and launching the DANSE algorithm.
    """

    # Prep for FFTs (zero-pad)
    for k in range(p.wasn.nNodes):  # for each node
        wasn[k].data, wasn[k].timeStamps, _ = base.prep_sigs_for_FFT(
            y=wasn[k].data,
            N=p.danseParams.DFTsize,
            Ns=p.danseParams.Ns,
            t=wasn[k].timeStamps
        )

        # Derive exponential averaging factor for `Ryy` and `Rnn` updates
        wasn[k].beta = np.exp(np.log(0.5) / \
            (p.danseParams.t_expAvg50p * wasn[k].fs / p.danseParams.Ns))

    # Launch DANSE
    out = core.danse(wasn, p.danseParams)

    # Visualize results
    # postproc.plot_sros(out)
    postproc.plot_des_sig_est(out)
    stop = 1


if __name__ == '__main__':
    sys.exit(main())
