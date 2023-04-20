# Purpose of script:
# Check what happens when setting one or more microphones to random noise
# (useless microphones).
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Date: 14.04.2023.

import re
import sys
import pickle, gzip
import itertools
from pathlib import Path
from danse_toolbox.d_classes import *
from .sandbox import main as main_sandbox

YAML_FILE = f'{Path(__file__).parent.parent}/config_files/useless_microphones.yaml'
BYPASS_ALREADY_RUN = True  # if True, bypass tests that have already been run
# TEST_TYPE = 'render_mics_useless'
TEST_TYPE = 'add_useless_mics'
    # ^^^ 'render_mics_useless': render some mics useless.
    # ^^^ 'add_useless_mics': add some useless mics.
N_ADDED_USELESS_MICS = np.arange(0, 4)  # numbers of useless mics to add
    # ^^^ only used if TEST_TYPE == 'add_useless_mics'

def main():
    """Main function (called by default when running script)."""
    
    # Load test parameters from YAML file
    testParams = TestParameters().load_from_yaml(YAML_FILE)

    # Run tests
    res = run_tests(testParams)


def run_tests(p: TestParameters):
    """
    Run tests.

    Parameters:
    -----------
    p: TestParameters
        Test parameters.

    Returns:
    --------
    out: list
    """

    # Get test ref from test type
    testRef = ''.join([s[0] for s in re.split('_', TEST_TYPE)]) 

    # Choose loop variable
    if TEST_TYPE == 'render_mics_useless':
        # Extract number of sensors
        nSensors = len(p.wasnParams.sensorToNodeIndices)
        # Derive all combinations of useless sensors
        combs = list(itertools.product([False, True], repeat=nSensors))
        # Do not consider case where all sensors are noise
        loopVariable = combs[:-1]
        # All sensor indices
        allSensorIndices = np.arange(nSensors)
    elif TEST_TYPE == 'add_useless_mics':
        loopVariable = N_ADDED_USELESS_MICS

    # Run tests
    out = []
    for ii, currVar in enumerate(loopVariable):

        if TEST_TYPE == 'render_mics_useless':
            # Render a particular combination of sensors useless
            currComb = allSensorIndices[np.array(currVar)]
            # Set current run reference
            runRef = f'{testRef}{ii + 1}_{currComb}'
        elif TEST_TYPE == 'add_useless_mics':
            # Set current run reference
            runRef = f'{testRef}{ii + 1}_{currVar}'
        
        # Check if test has already been run
        outputArchiveExportPath = f'{p.exportParams.exportFolder}/out_{runRef}.pkl.gz'
        if BYPASS_ALREADY_RUN and Path(outputArchiveExportPath).exists():
            print(f'>>> Test {runRef} already run. Bypassing...')
            continue

        # Inform user
        print(f'>>> Running test "{runRef}"...')

        # Adapt test parameters and make sure that the RNG seed is reinitialized
        pCurr = copy.deepcopy(p)
        if TEST_TYPE == 'render_mics_useless':
            # Set sensors to noise
            pCurr.setThoseSensorsToNoise = currComb
        elif TEST_TYPE == 'add_useless_mics':
            # Add useless microphones
            pCurr.wasnParams.addedNoiseSignalsPerNode = np.full(
                p.wasnParams.nNodes,
                fill_value=currVar  # add `nUselessMics` useless mics per node
            )
        # Update WASN parameters
        pCurr.wasnParams.__post_init__()
        pCurr.danseParams.get_wasn_info(pCurr.wasnParams)
        # Adapt export folder
        pCurr.exportParams.exportFolder = f'{p.exportParams.exportFolder}/{runRef}'
        # Reinitialize RNG seed
        pCurr.__post_init__()

        # Run main and append
        outCurr = main_sandbox(pCurr)

        # Dump to pickle file
        dump_to_pickle_archive(
            [outCurr.filters, outCurr.filtersCentr],
            outputArchiveExportPath
        )

        out.append(outCurr)

    return out


def dump_to_pickle_archive(data, path: str):
    """Dump object to pickle Gzip archive."""
    with gzip.open(path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    sys.exit(main())