
import sys
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

def main(
        p: TestParameters=None,
        plotASCearly=False,
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
        p = TestParameters().load_from_yaml(pathToCfg)
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

    # Parameters check and pre-DANSE computations
    p, wasnObj = core.prep_for_danse(p, wasnObj)
    
    # DANSE
    out, wasnObjUpdated = danse_it_up(wasnObj, p)

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
    Container function for launching the- correct version of the DANSE algorithm.
    """
    # Select appropriate function
    args = (wasnObj, p.danseParams)
    if p.is_fully_connected_wasn():  # Fully connected WASN case
        print(f'Running {p.danseParams.simType} DANSE... (verbose: {p.danseParams.printoutsAndPlotting.verbose}, GEVD: {p.danseParams.performGEVD})')
        if p.danseParams.simType == 'batch':  # true batch mode
            danse_function = core.danse_batch
        else:
            danse_function = core.danse
    else:  # Ad-hoc WASN topology case
        print(f'Running {p.danseParams.simType} TI-DANSE... (verbose: {p.danseParams.printoutsAndPlotting.verbose}, GEVD: {p.danseParams.performGEVD})')
        
        if p.danseParams.simType == 'batch':  # true batch mode
            danse_function = core.tidanse_batch
        else:
            danse_function = core.tidanse
    # Launch DANSE
    out, wasnUpdated = danse_function(*args)
    print('DANSE run complete.')

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
                inp = input(f'The folder\n"{p.exportParams.exportFolder}"\ncontains data. Overwrite? [y/n]:  ')
                while inp not in ['y', 'n']:
                    inp = input(f'Invalid input "{inp}". Please answer "y" for "yes" or "n" for "no":  ')
                if inp == 'n':
                    runit = False   # don't run
                    print('Aborting figures and sounds export.')
                elif inp == 'y':
                    print('Wiping folder before new figures and sounds exports.')
                    wipe_folder(p.exportParams.exportFolder)
        else:
            print(f'Create export folder "{p.exportParams.exportFolder}".')
            # Create dir. with missing parents directories.
            # https://stackoverflow.com/a/50110841
            Path(p.exportParams.exportFolder).mkdir(parents=True)

    if runit:
        pp.export_danse_outputs(out, wasnObj, room, p)

    return out


if __name__ == '__main__':
    sys.exit(main())
