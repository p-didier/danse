
import sys
import numpy as np
from pathlib import Path
from siggen.classes import *
import siggen.utils as sig_ut
import pyroomacoustics as pra
import danse_toolbox.d_post as pp
import danse_toolbox.d_core as core
from danse_toolbox.d_classes import *
from danse_toolbox.d_utils import wipe_folder
from danse_toolbox.d_base import DANSEparameters, CohDriftParameters

SIGNALSPATH = f'{Path(__file__).parent}/testing/sigs'
SEED = 12347

p = TestParameters(
    exportFolder = f'{Path(__file__).parent}/out/20230323_tests',
    seed=SEED,
    wasnParams=WASNparameters(
        # generateRandomWASNwithSeed=420,
        topologyParams=TopologyParameters(  # topology-related parameters
            # topologyType='ad-hoc',
            # topologyType='fully-connected',
            topologyType='user-defined',
            commDistance=4.,  # [m]
            seed=SEED,
            # plotTopo=True,
            userDefinedTopo=np.array([
                [1, 1, 0],  # Node 1
                [1, 1, 1],  # Node 2
                [0, 1, 1],  # Node 3
            ]),
            # userDefinedTopo=np.ones((3, 3)),
            # userDefinedTopo=np.ones((4, 4)),  # 20.02.2023: replicating ICASSP paper's WASN structure
            # userDefinedTopo=np.array([
            #     [1, 1],  # Node 1
            #     [1, 1],  # Node 2
            # ])
        ),
        sigDur=15,
        rd=np.array([5, 5, 5]),
        fs=16000,
        t60=0.0,
        interSensorDist=0.2,
        # nNodes=2,
        nNodes=3,
        # nNodes=4,
        # nSensorPerNode=[1, 1],
        nSensorPerNode=[1, 1, 1],
        # nSensorPerNode=[1, 3, 2],
        # nSensorPerNode=[1, 3, 2, 5],
        # nSensorPerNode=[1, 1, 1, 1],
        # selfnoiseSNR=np.inf,  # if `== np.inf` --> no self-noise at all
        selfnoiseSNR=99,
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
        # SROperNode=np.array([0, 200, -200, 400]),
        # SROperNode=np.array([0, 50, -50, 100]),
        # SROperNode=np.array([0, 20, -20, 40]),
        # SROperNode=np.array([0, 0, 0, 0]),
        # SROperNode=np.array([0, 50]),
        # SROperNode=np.array([0, 0]),
        SROperNode=np.array([0]),
    ),
    danseParams=DANSEparameters(
        DFTsize=1024,
        WOLAovlp=.5,  # [*100 -> %]
        # nodeUpdating='seq',
        nodeUpdating='asy',
        # broadcastType='fewSamples',
        broadcastType='wholeChunk',
        estimateSROs='CohDrift',
        # estimateSROs='DXCPPhaT',
        # compensateSROs=True,
        compensateSROs=False,
        cohDrift=CohDriftParameters(
            loop='open',
            alpha=0.95
        ),
        # vvvvvvvv FOR TI-DANSE TESTING vvvvvvvv
        computeCentralised=True,
        computeLocal=True,
        # noExternalFilterRelaxation=True,
        noExternalFilterRelaxation=False,
        performGEVD=False,
        # performGEVD=True,
        # bypassUpdates=True  # /!\
        t_expAvg50p=2,
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
)
p.danseParams.get_wasn_info(p.wasnParams)  # complete parameters

def main(p: TestParameters):

    # Build room
    room, vad, wetSpeechAtRefSensor, wetNoiseAtRefSensor =\
        sig_ut.build_room(p.wasnParams)

    # Build WASN (asynchronicities, topology)
    wasnObj = sig_ut.build_wasn(
        room,
        vad,
        wetSpeechAtRefSensor,
        wetNoiseAtRefSensor,
        p.wasnParams
    )

    # pp.plot_asc(
    #     room,
    #     p.wasnParams,
    #     p.exportFolder,
    #     wasnObj.adjacencyMatrix,
    #     [node.nodeType for node in wasnObj.wasn]
    # )
    # DANSE
    out, wasnObjUpdated = danse_it_up(wasnObj, p)

    # Visualize results
    out = postprocess(out, wasnObjUpdated, room, p)
# 
    # Save `DANSEoutputs` object after metrics computation in `postprocess()`
    out.save(foldername=p.exportFolder, light=True)
    p.save()    # save `TestParameters` object


def danse_it_up(
    wasnObj: WASN,
    p: TestParameters
    ) -> tuple[pp.DANSEoutputs, WASN]:
    """
    Container function for prepping signals and launching the DANSE algorithm.
    """

    for k in range(p.wasnParams.nNodes):  # for each node
        # Derive exponential averaging factor for `Ryy` and `Rnn` updates
        wasnObj.wasn[k].beta = np.exp(np.log(0.5) / \
            (p.danseParams.t_expAvg50p * wasnObj.wasn[k].fs / p.danseParams.Ns))

    # Launch DANSE
    if p.is_fully_connected_wasn():
        # Fully connected WASN case
        out, wasnUpdated = core.danse(wasnObj, p.danseParams)
    else:
        # Ad-hoc WASN topology case
        out, wasnUpdated = core.tidanse(wasnObj, p.danseParams)

    return out, wasnUpdated


def postprocess(
    out: pp.DANSEoutputs,
    wasnObj: WASN,
    room: pra.room.ShoeBox,
    p: TestParameters
    ) -> pp.DANSEoutputs:
    """
    Defines the post-processing steps to be undertaken after a DANSE run.
    Using the `danse.danse_toolbox.d_post` [abbrev. `pp`] functions.

    Parameters
    ----------
    out : `danse.danse_toolbox.d_post.DANSEoutputs` object
        DANSE outputs (signals, etc.)
    wasnObj : `WASN` object
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
                print('Aborting figures and sounds export.')
            else:
                print('Wiping folder before new figures and sounds exports.')
                wipe_folder(p.exportFolder)
    else:
        print(f'Create export folder "{p.exportFolder}".')
        Path(p.exportFolder).mkdir()

    if runit:
        # Export convergence plot
        # out.plot_convergence(wasn)
        # out.plot_convergence(p.exportFolder)

        # Export .wav files
        out.export_sounds(wasnObj.wasn, p.exportFolder)

        # Plot (+ export) acoustic scenario (WASN)
        pp.plot_asc(
            asc=room,
            p=p.wasnParams,
            folder=p.exportFolder,
            usedAdjacencyMatrix=wasnObj.adjacencyMatrix,
            nodeTypes=[node.nodeType for node in wasnObj.wasn],
            originalAdjacencyMatrix=p.wasnParams.topologyParams.userDefinedTopo
        )

        # Plot SRO estimation performance
        fig = out.plot_sro_perf(
            Ns=p.danseParams.Ns,
            fs=p.wasnParams.fs,
            xaxistype='both'  # "both" == iterations [-] _and_ instants [s]
        )
        fig.savefig(f'{p.exportFolder}/sroEvolution.png')
        fig.savefig(f'{p.exportFolder}/sroEvolution.pdf')

        # Plot performance metrics (+ export)
        out.plot_perf(wasnObj.wasn, p.exportFolder)

        # Plot signals at specific nodes (+ export)
        out.plot_sigs(wasnObj.wasn, p.exportFolder)

    return out


if __name__ == '__main__':
    sys.exit(main(p=p))
