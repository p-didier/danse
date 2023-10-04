
import sys
from pathlib import Path
from siggen.classes import *
import siggen.utils as sig_ut
import pyroomacoustics as pra
import danse_toolbox.d_post as pp
import danse_toolbox.d_core as core
import danse_toolbox.d_classes as cl

PATH_TO_CONFIG_FILE = f'{Path(__file__).parent.parent}/config_files/sandbox_config.yaml'
BYPASS_DYNAMIC_PLOTS = True  # if True, bypass all runtime (dynamic) plotting 

def main(
        p: cl.TestParameters=None,
        cfgFilename: str=''
    ) -> pp.DANSEoutputs:
    """Main function.
    
    Parameters:
    -----------
    p: TestParameters
        Test parameters.
    plotASCearly: bool
        If True, plot the ASC before running DANSE.
    cfgFilename: str
        Path to the config file to use. If empty, use the default one.

    Returns:
    --------
    outPostProc: DANSEoutputs
        Output object containing all necessary info from DANSE run.
    """

    if p is None:
        # Load parameters from config file
        print('Loading parameters...')
        pathToCfg = cfgFilename if cfgFilename else PATH_TO_CONFIG_FILE
        p = cl.TestParameters().load_from_yaml(pathToCfg)
        p.danseParams.get_wasn_info(p.wasnParams)  # complete parameters
        print('Parameters loaded.')
    else:
        # Check that parameters are complete
        if p.danseParams.wasnInfoInitiated is False:
            p.danseParams.get_wasn_info(p.wasnParams)

    # Check export folder and whether we can run the test
    runit = p.exportParams.check_export_folder()

    if runit:
        # Build room
        print('Building scenario...')
        room, vad, wetSpeeches, wetNoises = sig_ut.build_scenario(p.wasnParams)
        # Complete parameters (useful in case of YAML-loaded layout)
        p.danseParams.get_wasn_info(p.wasnParams)
        print('Scenario built.')

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

        if 0:
            pp.plot_asc(
                room,
                p.wasnParams,
                p.exportParams.exportFolder,
                wasnObj.adjacencyMatrix,
                [node.nodeType for node in wasnObj.wasn],
                plot3Dview=True
            )
            plt.show()
        # Parameters check and pre-DANSE computations
        p, wasnObj = core.prep_for_danse(p, wasnObj)
        
        # DANSE
        out, wasnObjUpdated = danse_it_up(wasnObj, p)

        # Post-process results (save, export, plot...)
        print('Post-processing...')
        outPostProc = postprocess(out, wasnObjUpdated, p, room)
        print('Done.')
    else:
        print('Aborting DANSE run.')
        outPostProc = None

    return outPostProc


def danse_it_up(
        wasnObj: WASN,
        p: cl.TestParameters
    ) -> tuple[pp.DANSEoutputs, WASN]:
    """Container function for launching the- correct version of
    the DANSE algorithm."""
    args = (wasnObj, p.danseParams)
    # Select appropriate function
    if p.is_fully_connected_wasn():  # Fully connected WASN case
        print(f'Running {p.danseParams.simType} DANSE... (verbose: {p.danseParams.printoutsAndPlotting.verbose}, GEVD: {p.danseParams.performGEVD})')
        if p.danseParams.simType == 'batch':  # true batch mode
            raise NotImplementedError('Batch mode not implemented / tested yet.')
            danse_function = core.danse_batch
        else:
            danse_function = core.danse
    else:  # Ad-hoc WASN topology case
        print(f'Running {p.danseParams.simType} TI-DANSE... (verbose: {p.danseParams.printoutsAndPlotting.verbose}, GEVD: {p.danseParams.performGEVD})')
        if p.danseParams.simType == 'batch':  # true batch mode
            raise NotImplementedError('Batch mode not implemented / tested yet.')
            danse_function = core.tidanse_batch
        else:
            danse_function = core.tidanse
    # Launch DANSE
    dv, wasnObj = danse_function(*args)
    # Compute signals for SNR computation
    sigsSnr = cl.generate_signals_for_snr_computation(p.danseParams, dv, wasnObj, danse_function)
    
    # Format the output for post-processing
    out, wasnUpdated = core.format_output(
        p.danseParams,
        dv,
        wasnObj,
        sigsSnr=sigsSnr
    )
    print('DANSE run complete.')

    # If asked, compute best possible performance (centralized, no SROs, batch)
    if p.exportParams.bestPerfReference:
        print('Computing best possible performance...')
        outBP = core.get_best_perf(*args)
        out.include_best_perf_data(outBP)
        print('Best possible performance computed.')

    return out, wasnUpdated

def postprocess(
        out: pp.DANSEoutputs,
        wasnObj: WASN,
        p: cl.TestParameters,
        room: pra.room.ShoeBox=None,
        bypassGlobalPickleExport: bool=False
    ) -> pp.DANSEoutputs:
    """Defines the post-processing steps to be undertaken after a DANSE run.
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
    bypassGlobalPickleExport : bool
        If True, bypass the global Pickle export of aN
        `OutputsForPostProcessing` object.

    Returns
    -------
    out : `danse.danse_toolbox.d_post.DANSEoutputs` object
        DANSE outputs (signals, etc.), after post-processing.
    """
    if not bypassGlobalPickleExport:
        # Export all required data as global Pickle archive
        print('Exporting all data for further subsequent post-processing...')
        forPP = cl.OutputsForPostProcessing(out, wasnObj, p)
        forPP.save(p.exportParams.exportFolder)
    pp.export_danse_outputs(out, wasnObj, p, room)

    return out


if __name__ == '__main__':
    sys.exit(main())
