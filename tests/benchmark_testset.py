# Set of benchmarking tests for assessing performance of DANSE toolbox at any
# point in its implementation.
#
# ~P. Didier -- 01.12.2022

import sys, os
import random
import itertools
from pathlib import Path
from siggen.classes import *
import pyroomacoustics as pra
import siggen.utils as sig_ut
import danse_toolbox.d_post as pp
from danse_toolbox.d_base import *
import danse_toolbox.d_core as core
from danse_toolbox.d_classes import *
import danse_toolbox.dataclass_methods as met

SIGNALSPATH = f'{Path(__file__).parent}/testing/sigs'

# List of booleans parameters to consider.
# 01.12.2022: all booleans commutations are considered.
# - 'seq': if True, sequential node-updating. Else, asynchronous.
# - 'rev': if True, reverberant room. Else, anechoic.
# - 'SROs': if True, asynchronous WASN. Else, synchronous.
# - 'estcomp': if True, estimate and compensate for SROs. Else, do not.
# - 'basic': if True, consider a 2-nodes WASN with 1 sensor each.
#    Else, consider a 'realistic' WASN with more nodes, multiple sensors each.
PARAMETERS = ['seq', 'rev', 'SROs', 'estcomp', 'basic']

@dataclass
class TestParameters:
    selfnoiseSNR: int = -50 # [dB] microphone self-noise SNR
    referenceSensor: int = 0    # Index of the reference sensor at each node
    wasn: WASNparameters = WASNparameters(sigDur=15)
    danseParams: DANSEparameters = DANSEparameters()
    exportFolder: str = f'{Path(__file__).parent}/out'  # folder to export outputs
    seed: int = 12345

    def __post_init__(self):
        np.random.seed(self.seed)  # set random seed
        random.seed(self.seed)  # set random seed
        # Check consistency
        if self.danseParams.nodeUpdating == 'sym' and\
            any(self.wasn.SROperNode != 0):
            raise ValueError('Simultaneous node-updating impossible in the presence of SROs.')

    def save(self, exportType='pkl'):
        """Saves dataclass to Pickle archive."""
        met.save(self, self.exportFolder, exportType=exportType)

    def load(self, foldername, dataType='pkl'):
        """Loads dataclass to Pickle archive in folder `foldername`."""
        return met.load(self, foldername, silent=True, dataType=dataType)

BASEPARAMS = TestParameters(
    wasn=WASNparameters(
        sigDur=15,
        rd=np.array([5, 5, 5]),
        fs=8000,
        t60=0.0,
        nNodes=2,
        selfnoiseSNR=99,
        nSensorPerNode=[1, 1],
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
        SROperNode=[0, 0]
    ),
    danseParams=DANSEparameters(
        DFTsize=1024,
        WOLAovlp=.5,
        nodeUpdating='asy',
        broadcastType='wholeChunk',
        estimateSROs='CohDrift',
        compensateSROs=False,
        cohDrift=CohDriftParameters(
            loop='open',
            alpha=0.99
        ),
        computeCentralised=True,
        computeLocal=True,
    )
)
BASEPARAMS.danseParams.get_wasn_info(BASEPARAMS.wasn)  # complete parameters


def main():
    """Main function."""

    # Generate test battery
    bools = list(itertools.product([False, True], repeat=len(PARAMETERS)))
    tests = []
    for ii in range(len(bools)):
        paramBools = dict([(PARAMETERS[jj], bools[ii][jj])\
                for jj in range(len(PARAMETERS))])
        # Avoid unnecessary tests (e.g., SRO est./comp. when no SRO present)
        if not paramBools['SROs'] and paramBools['estcomp']:
            pass
        else:
            tests.append(paramBools)

    # Run tests
    for ii in range(len(tests)):
        print(f'######### Running test {ii+1}/{len(tests)} ({tests[ii]})... #########')
        run(tests[ii])
    print(f'######### All testing done. #########')

    stop = 1


def run(test: dict):
    """Builds test parameters and runs the test."""

    # Prepare parameters
    p = BASEPARAMS
    if test['seq']:
        p.danseParams.nodeUpdating = 'seq'  # sequential node-updating
    else:
        p.danseParams.nodeUpdating = 'asy'  # asynchronous node-updating

    if test['rev']:
        p.wasn.t60 = 0.2
    else:
        p.wasn.t60 = 0.0

    if test['basic']:
        p.wasn.nNodes = 2
        p.wasn.nSensorPerNode = [1,1]
    else:
        p.wasn.nNodes = 4
        p.wasn.nSensorPerNode = [1,3,2,5]

    if test['SROs']:
        if test['basic']:  # only two nodes
            p.wasn.SROperNode = [0, 100]
        else:   # more nodes
            p.wasn.SROperNode = [0, 50, -50, 100]
    else:
        p.wasn.SROperNode = [0] * p.wasn.nNodes

    if test['estcomp']:
        p.danseParams.compensateSROs = True
        p.danseParams.broadcastType = 'fewSamples'
        p.danseParams.broadcastLength = 1
    else:
        p.danseParams.compensateSROs = False
        p.danseParams.broadcastType = 'wholeChunk'
        p.danseParams.broadcastLength = p.danseParams.DFTsize // 2
    
    # Finalize parameters update
    p.wasn.__post_init__()
    p.danseParams.get_wasn_info(p.wasn)

    # Build export folder name
    foldername = '_'.join([f'{ii}{int(test[ii])}' for ii in list(test.keys())])
    p.exportFolder = f'{Path(__file__).parent}/out/benchmark/{foldername}'

    # Launch test
    out = launch(p)
    # Save `DANSEoutputs` object after metrics computation in `postprocess()`
    out.save(foldername=p.exportFolder, light=True)
    p.save()    # save `TestParameters` object

    return out


def launch(p: TestParameters):
    """Launches a test, given parameters."""
    # Build room
    room, vad, wetSpeechAtRefSensor = sig_ut.build_room(p.wasn)
    # Build WASN (asynchronicities, topology)
    wasn = sig_ut.build_wasn(room, vad, wetSpeechAtRefSensor, p.wasn)
    # DANSE
    out, wasnUpdated = danse_it_up(wasn, p)
    # Visualize results
    out = postprocess(out, wasnUpdated, room, p)
    return out


def danse_it_up(
    wasn: list[Node],
    p: TestParameters
    ) -> tuple[pp.DANSEoutputs, list[Node]]:
    """
    Container function for prepping signals and launching the DANSE algorithm.
    """
    for k in range(p.wasn.nNodes):  # for each node
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
        # if Path(p.exportFolder).stat().st_size > 0:
        #     inp = input(f'The folder\n"{p.exportFolder}"\ncontains data. Overwrite? [y/[n]]:  ')
        #     if inp not in ['y', 'Y']:
        #         runit = False   # don't run
        #         print('Aborting export.') # TODO: TMP TMP TMP 
        runit = False
    else:
        print(f'Create export folder "{p.exportFolder}".')
        os.makedirs(p.exportFolder)  # better than Path().mkdir()
        # because allows subfolders in folders
        # (https://stackoverflow.com/a/6692700)

    if runit:
        # Export .wav files
        out.export_sounds(wasn, p.exportFolder)
        # Plot (+ export) acoustic scenario (WASN)
        pp.plot_asc(room, p.wasn, p.exportFolder)
        # Plot performance metrics (+ export)
        out.plot_perf(wasn, p.exportFolder)
        # Plot signals at specific nodes (+ export)
        out.plot_sigs(wasn, p.exportFolder)

    return out


if __name__ == '__main__':
    sys.exit(main())
