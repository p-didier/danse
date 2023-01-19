
import sys
import random
import numpy as np
import pyroomacoustics as pra
from pathlib import Path
from siggen.classes import *
import siggen.utils as sig_ut
import danse_toolbox.d_core as core
from danse_toolbox.d_classes import *
import danse_toolbox.d_post as pp
import danse_toolbox.dataclass_methods as met
from danse_toolbox.d_base import DANSEparameters, CohDriftParameters

SIGNALSPATH = f'{Path(__file__).parent}/testing/sigs'

@dataclass
class TestParameters:
    selfnoiseSNR: int = -50 # [dB] microphone self-noise SNR
    #
    referenceSensor: int = 0    # Index of the reference sensor at each node
    #
    wasnParams: WASNparameters = WASNparameters(
        sigDur=5
    )
    danseParams: DANSEparameters = DANSEparameters()
    # exportFolder: str = f'{Path(__file__).parent}/out/dxcptests'  # folder to export outputs
    exportFolder: str = f'{Path(__file__).parent}/out/20230119_convergencePlotTest'  # folder to export outputs
    #
    seed: int = 12345

    def __post_init__(self):
        np.random.seed(self.seed)  # set random seed
        random.seed(self.seed)  # set random seed
        #
        self.testid = f'J{self.wasnParams.nNodes}Mk{list(self.wasnParams.nSensorPerNode)}WNn{self.wasnParams.nNoiseSources}Nd{self.wasnParams.nDesiredSources}T60_{int(self.wasnParams.t60)*1e3}ms'
        # Check consistency
        if self.danseParams.nodeUpdating == 'sym' and\
            any(self.wasnParams.SROperNode != 0):
            raise ValueError('Simultaneous node-updating impossible in the presence of SROs.')

    def save(self, exportType='pkl'):
        """Saves dataclass to Pickle archive."""
        met.save(self, self.exportFolder, exportType=exportType)

    def load(self, foldername, dataType='pkl'):
        """Loads dataclass to Pickle archive in folder `foldername`."""
        return met.load(self, foldername, silent=True, dataType=dataType)

p = TestParameters(
    wasnParams=WASNparameters(
        sigDur=15,
        rd=np.array([5, 5, 5]),
        fs=16000,
        t60=0.2,
        # nNodes=2,
        nNodes=4,
        # selfnoiseSNR=np.inf,  # if `== np.inf` --> no self-noise at all
        selfnoiseSNR=99,
        nSensorPerNode=[1, 3, 2, 5],
        # nSensorPerNode=[1, 1],
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
        SROperNode=np.array([0, 0]),
        # loadFrom=None,
        # loadFrom='C:/Users/pdidier/Dropbox/PC/Documents/sounds-phd/02_data/01_acoustic_scenarios/for_submissions/icassp2023/J4Mk[1_3_2_5]_Ns1_Nn2/AS18_RT150ms'
    ),
    danseParams=DANSEparameters(
        DFTsize=1024,
        WOLAovlp=.5,
        # nodeUpdating='seq',
        nodeUpdating='asy',
        # broadcastType='fewSamples',
        broadcastType='wholeChunk',
        # estimateSROs='CohDrift',
        estimateSROs='DXCPPhaT',
        compensateSROs=True,
        # compensateSROs=False,
        cohDrift=CohDriftParameters(
            loop='open',
            alpha=0.99
        ),
        computeCentralised=True,
        computeLocal=True,
    )
)
p.danseParams.get_wasn_info(p.wasnParams)  # complete parameters

def main(p: TestParameters):

    # Build room
    room, vad, wetSpeechAtRefSensor = sig_ut.build_room(p.wasnParams)

    # Build WASN (asynchronicities, topology)
    wasn = sig_ut.build_wasn(room, vad, wetSpeechAtRefSensor, p.wasnParams)

    # DANSE
    out, wasnUpdated = danse_it_up(wasn, p)

    # Visualize results
    out = postprocess(out, wasnUpdated, room, p)

    # Save `DANSEoutputs` object after metrics computation in `postprocess()`
    out.save(foldername=p.exportFolder, light=True)
    p.save()    # save `TestParameters` object


def danse_it_up(
    wasn: list[Node],
    p: TestParameters
    ) -> tuple[pp.DANSEoutputs, list[Node]]:
    """
    Container function for prepping signals and launching the DANSE algorithm.
    """

    for k in range(p.wasnParams.nNodes):  # for each node
        # Derive exponential averaging factor for `Ryy` and `Rnn` updates
        wasn[k].beta = np.exp(np.log(0.5) / \
            (p.danseParams.t_expAvg50p * wasn[k].fs / p.danseParams.Ns))

    # Launch DANSE
    out, wasnUpdated = core.danse(wasn, p.danseParams)

    return out, wasnUpdated


def postprocess(out: pp.DANSEoutputs,
        wasn: list[Node],
        room: pra.room.ShoeBox,
        p: TestParameters) -> pp.DANSEoutputs:
    """
    Defines the post-processing steps to be undertaken after a DANSE run.
    Using the `danse.danse_toolbox.d_post` [abbrev. `pp`] functions.

    Parameters
    ----------
    out : `danse.danse_toolbox.d_post.DANSEoutputs` object
        DANSE outputs (signals, etc.)
    wasn : list of `Node` objects
        WASN under consideration, after DANSE processing.
    room : `pyroomacoustics.room.ShoeBox` object
        Acoustic scenario under consideration.
    p : `TestParameters` object
        Test parameters.
    """

    # Default booleans
    runit = True   # by default, run
    # Check whether export folder exists
    if Path(p.exportFolder).is_dir():
        # Check whether the folder contains something
        if Path(p.exportFolder).stat().st_size > 0:
            inp = input(f'The folder\n"{p.exportFolder}"\ncontains data. Overwrite? [y/[n]]:  ')
            if inp not in ['y', 'Y']:
                runit = False   # don't run
                print('Aborting export.')
    else:
        print(f'Create export folder "{p.exportFolder}".')
        Path(p.exportFolder).mkdir()

    if runit:
        # Export convergence plot
        # out.plot_convergence(wasn)
        out.plot_convergence(p.exportFolder)

        # Export .wav files
        out.export_sounds(wasn, p.exportFolder)

        # Plot (+ export) acoustic scenario (WASN)
        pp.plot_asc(room, p.wasnParams, p.exportFolder)

        # Plot SRO estimation performance
        fig = out.plot_sro_perf(
            Ns=p.danseParams.Ns,
            fs=p.wasnParams.fs,
            xaxistype='both'  # "both" == iterations [-] _and_ instants [s]
        )
        fig.savefig(f'{p.exportFolder}/sroEvolution.png')
        fig.savefig(f'{p.exportFolder}/sroEvolution.pdf')

        # Plot performance metrics (+ export)
        out.plot_perf(wasn, p.exportFolder)

        # Plot signals at specific nodes (+ export)
        out.plot_sigs(wasn, p.exportFolder)

    return out


if __name__ == '__main__':
    sys.exit(main(p=p))
