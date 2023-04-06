
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
from danse_toolbox.d_base import DANSEparameters, CohDriftParameters, PrintoutsAndPlotting

SIGNALS_PATH = f'{Path(__file__).parent.parent}/tests/sigs'
SEED = 12347
BYPASS_DYNAMIC_PLOTS = True  # if True, bypass all runtime (dynamic) plotting 

def main(p: TestParameters=None, plotASCearly=False) -> pp.DANSEoutputs:
    """Main function.
    
    Parameters:
    -----------
    p: TestParameters
        Test parameters.

    Returns:
    --------
    outPostProc: DANSEoutputs
        Output object containing all necessary info from DANSE run.
    """
    if p is None:
        p = TestParameters(
            bypassExport=False,
            exportFolder=f'{Path(__file__).parent.parent}/out/20230406_tests/testsync_tau30s',
            seed=SEED,
            wasnParams=WASNparameters(
                # layoutType='random_spinning_top',
                layoutType='vert_spinning_top',
                # layoutType='random',
                VADenergyDecrease_dB=35,  # [dB]
                # generateRandomWASNwithSeed=420,
                topologyParams=TopologyParameters(  # topology-related parameters
                    # topologyType='ad-hoc',
                    topologyType='fully-connected',
                    # topologyType='user-defined',
                    commDistance=4.,  # [m]
                    seed=SEED,
                    # plotTopo=True,
                    userDefinedTopo=np.array([
                        [1, 1, 0],  # Node 1
                        [1, 1, 1],  # Node 2
                        [0, 1, 1],  # Node 3
                    ]),
                    # userDefinedTopo=np.array([
                    #     [1, 1, 0, 0],  # Node 1
                    #     [1, 1, 1, 0],  # Node 2
                    #     [0, 1, 1, 1],  # Node 3
                    #     [0, 0, 1, 1],  # Node 4
                    # ]),
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
                # t60=0.0,
                t60=0.2,
                interSensorDist=0.2,
                # interSensorDist=0.1,
                # nNodes=2,
                nNodes=3,
                # nNodes=4,
                # nSensorPerNode=[1, 1],
                # nSensorPerNode=[1, 1, 1],
                # nSensorPerNode=[1, 2, 3, 4],
                # nSensorPerNode=[1, 1, 1, 1],
                nSensorPerNode=[1, 2, 3],
                # nSensorPerNode=[1, 2, 3, 2],
                # nSensorPerNode=[1, 3, 2],
                # nSensorPerNode=[1, 3, 2, 5],
                # nSensorPerNode=[1, 1, 1, 1],
                # selfnoiseSNR=np.inf,  # if `== np.inf` --> no self-noise at all
                selfnoiseSNR=99,
                desiredSignalFile=[f'{SIGNALS_PATH}/01_speech/{file}'\
                    for file in [
                        'speech1.wav',
                        'speech2.wav'
                    ]],
                # noiseSignalFile=[f'{SIGNALSPATH}/02_noise/{file}'\
                #     for file in [
                #         'whitenoise_signal_1.wav',
                #         'whitenoise_signal_2.wav'
                #     ]],
                noiseSignalFile=[f'{SIGNALS_PATH}/02_noise/{file}'\
                    for file in [
                        'ssn/ssn_speech1.wav',
                        'ssn/ssn_speech2.wav'
                    ]],
                # SROperNode=np.array([0, 200, -200, 400]),
                # SROperNode=np.array([0, 50, -50, 100]),
                # SROperNode=np.array([0, 20, -20, 40]),
                # SROperNode=np.array([0, 0, 0, 0]),
                # SROperNode=np.array([0, 50, 100]),
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
                #
                # vvvvvvvvv SRO-RELATED vvvvvvvvv
                estimateSROs='CohDrift',
                # estimateSROs='DXCPPhaT',
                # compensateSROs=True,
                compensateSROs=False,
                cohDrift=CohDriftParameters(
                    loop='open',
                    alpha=0.95
                ),
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                #
                filterInitType='selectFirstSensor_andFixedValue',
                filterInitFixedValue=1,
                # filterInitFixedValue=0.01,
                #
                # vvvvvvvv FOR TI-DANSE TESTING vvvvvvvv
                computeCentralised=True,
                computeLocal=True,
                # noExternalFilterRelaxation=True,
                noExternalFilterRelaxation=False,
                performGEVD=False,
                # performGEVD=True,
                # bypassUpdates=True  # /!\
                # t_expAvg50p=10,
                t_expAvg50p=30,
                # t_expAvg50p=3,
                # t_expAvg50p=1,
                timeBtwExternalFiltUpdates=1,
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                printoutsAndPlotting=PrintoutsAndPlotting(
                    showWASNs=False if BYPASS_DYNAMIC_PLOTS else True,
                    onlySNRandESTOIinPlots=True
                ),
                # vvvv Covariance matrices initialization parameters
                covMatInitType='fully_random',
                # covMatInitType='eye_and_random',
                # covMatInitType='batch_ground_truth',
                # covMatEyeInitScaling=0.0001,
                # covMatRandomInitScaling=0.00001,
                # covMatSameInitForAllFreqs=True,
                # covMatSameInitForAllNodes=False
            )
        )
        p.danseParams.get_wasn_info(p.wasnParams)  # complete parameters

    # Build room
    room, vad, wetSpeeches, wetNoises = sig_ut.build_room(p.wasnParams)

    # Build WASN (asynchronicities, topology)
    wasnObj = sig_ut.build_wasn(
        room,
        vad,
        wetSpeeches,
        wetNoises,
        p.wasnParams,
        p.danseParams.minNoSpeechDurEndUterrance
    )

    if plotASCearly:
        pp.plot_asc(
            room,
            p.wasnParams,
            p.exportFolder,
            wasnObj.adjacencyMatrix,
            [node.nodeType for node in wasnObj.wasn],
            plot3Dview=True
        )
    # DANSE
    out, wasnObjUpdated = danse_it_up(wasnObj, p)

    # Post-process results (save, export, plot...)
    outPostProc = postprocess(out, wasnObjUpdated, room, p)

    return outPostProc


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

    Returns
    -------
    out : `danse.danse_toolbox.d_post.DANSEoutputs` object
        DANSE outputs (signals, etc.), after post-processing.
    """

    # Default booleans
    runit = True   # by default, run
    if p.bypassExport:
        print('Not exporting figures and sounds export (`bypassExport` is True).')
    else:
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
            # Create dir. with missing parents directories.
            # https://stackoverflow.com/a/50110841
            Path(p.exportFolder).mkdir(parents=True)

    if runit:
        if not p.bypassExport:
            # Export convergence plot
            out.plot_convergence(p.exportFolder)

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

        # Compute performance metrics (+ export if needed)
        if p.bypassExport:
            exportFolder = None
        else:
            exportFolder = p.exportFolder
        out.plot_perf(
            wasnObj.wasn,
            exportFolder,
            p.danseParams.printoutsAndPlotting.onlySNRandESTOIinPlots,
            snrYlimMax=p.snrYlimMax,
        )

        if not p.bypassExport:
            # Plot signals at specific nodes (+ export)
            out.plot_sigs(wasnObj.wasn, p.exportFolder)

            # Save `DANSEoutputs` object after metrics computation
            out.save(foldername=p.exportFolder, light=True)
            p.save()    # save `TestParameters` object

    return out


if __name__ == '__main__':
    sys.exit(main(p=p))
