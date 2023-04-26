
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

PATH_TO_CONFIG_FILE = f'{Path(__file__).parent.parent}/config_files/sandbox_config.yaml'
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
        # Load parameters from config file
        print('Loading parameters...')
        p = TestParameters().load_from_yaml(PATH_TO_CONFIG_FILE)
        p.danseParams.get_wasn_info(p.wasnParams)  # complete parameters
        print('Parameters loaded.')
    else:
        # Check that parameters are complete
        if p.danseParams.wasnInfoInitiated is False:
            p.danseParams.get_wasn_info(p.wasnParams)

    # Build room
    print('Building room...')
    room, vad, wetSpeeches, wetNoises = sig_ut.build_room(p.wasnParams)
    print('Room built.')

    # Build WASN (asynchronicities, topology)
    print('Building WASN...')
    wasnObj = sig_ut.build_wasn(
        room,
        vad,
        wetSpeeches,
        wetNoises,
        p.wasnParams,
        p.danseParams.startComputeMetricsAt,
        p.danseParams.minNoSpeechDurEndUtterance,
        p.setThoseSensorsToNoise
    )
    print('WASN built.')

    if plotASCearly:
        pp.plot_asc(
            room,
            p.wasnParams,
            p.exportParams.exportFolder,
            wasnObj.adjacencyMatrix,
            [node.nodeType for node in wasnObj.wasn],
            plot3Dview=True
        )

    # Parameters check for DANSE
    p, wasnObj = core.check_params(p, wasnObj)
    
    # DANSE
    print(f'Running DANSE... (verbose: {p.danseParams.printoutsAndPlotting.verbose}, GEVD: {p.danseParams.performGEVD})')
    out, wasnObjUpdated = danse_it_up(wasnObj, p)
    print('DANSE run complete.')

    # Post-process results (save, export, plot...)
    print('Post-processing...')
    outPostProc = postprocess(out, wasnObjUpdated, room, p)
    print('Done.')

    return outPostProc


def danse_it_up(
    wasnObj: WASN,
    p: TestParameters
    ) -> tuple[pp.DANSEoutputs, WASN]:
    """
    Container function for prepping signals and launching the DANSE algorithm.
    """
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
    if p.exportParams.bypassAllExports:
        print('Not exporting figures and sounds export (`bypassExport` is True).')
    else:
        # Check whether export folder exists
        if Path(p.exportParams.exportFolder).is_dir():
            # Check whether the folder contains something
            if Path(p.exportParams.exportFolder).stat().st_size > 0:
                inp = input(f'The folder\n"{p.exportParams.exportFolder}"\ncontains data. Overwrite? [y/[n]]:  ')
                if inp not in ['y', 'Y']:
                    runit = False   # don't run
                    print('Aborting figures and sounds export.')
                else:
                    print('Wiping folder before new figures and sounds exports.')
                    wipe_folder(p.exportParams.exportFolder)
        else:
            print(f'Create export folder "{p.exportParams.exportFolder}".')
            # Create dir. with missing parents directories.
            # https://stackoverflow.com/a/50110841
            Path(p.exportParams.exportFolder).mkdir(parents=True)

    if runit:
        if not p.exportParams.bypassAllExports:
            # Export filter coefficients norm plot
            if p.exportParams.filterNormsPlot:
                out.plot_filter_norms(
                    p.exportParams.exportFolder,
                    exportNormsAsPickle=p.exportParams.filterNorms  # boolean to export filter norms as pickle  
                )

            # Export condition number plot
            if p.exportParams.conditionNumberPlot:
                out.plot_cond(p.exportParams.exportFolder)

            # Export convergence plot
            if p.exportParams.convergencePlot:
                out.plot_convergence(p.exportParams.exportFolder)

            # Export .wav files
            if p.exportParams.wavFiles:
                out.export_sounds(wasnObj.wasn, p.exportParams.exportFolder)

            # Plot (+ export) acoustic scenario (WASN)
            if p.exportParams.acousticScenarioPlot:
                pp.plot_asc(
                    asc=room,
                    p=p.wasnParams,
                    folder=p.exportParams.exportFolder,
                    usedAdjacencyMatrix=wasnObj.adjacencyMatrix,
                    nodeTypes=[node.nodeType for node in wasnObj.wasn],
                    originalAdjacencyMatrix=p.wasnParams.topologyParams.userDefinedTopo
                )

            # Plot SRO estimation performance
            if p.exportParams.sroEstimPerfPlot:
                fig = out.plot_sro_perf(
                    Ns=p.danseParams.Ns,
                    fs=p.wasnParams.fs,
                    xaxistype='both'  # "both" == iterations [-] _and_ instants [s]
                )
                fig.savefig(f'{p.exportParams.exportFolder}/sroEvolution.png')
                fig.savefig(f'{p.exportParams.exportFolder}/sroEvolution.pdf')

        # Compute performance metrics (+ export if needed)
        if p.exportParams.metricsPlot:
            if p.exportParams.bypassAllExports:
                exportFolder = None
            else:
                exportFolder = p.exportParams.exportFolder
                out.plot_perf(
                    wasnObj.wasn,
                    exportFolder,
                    p.danseParams.printoutsAndPlotting.onlySNRandESTOIinPlots,
                    snrYlimMax=p.snrYlimMax,
                )

        if not p.exportParams.bypassAllExports:
            # Plot signals at specific nodes (+ export)
            if p.exportParams.waveformsAndSpectrograms:
                out.plot_sigs(wasnObj.wasn, p.exportParams.exportFolder)

            # Save `DANSEoutputs` object after metrics computation
            if p.exportParams.danseOutputsFile:
                out.save(foldername=p.exportParams.exportFolder, light=True)
            # Save `TestParameters` object
            if p.exportParams.parametersFile:
                if p.loadedFromYaml:
                    p.save_yaml()   # save `TestParameters` object as YAML file
                else:
                    p.save()    # save `TestParameters` object as Pickle archive

    return out


if __name__ == '__main__':
    sys.exit(main())
