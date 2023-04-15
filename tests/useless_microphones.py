# Purpose of script:
# Check what happens when setting one or more microphones to random noise
# (useless microphones).
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Date: 14.04.2023.

import sys
import pickle, gzip
import itertools
from pathlib import Path
from danse_toolbox.d_classes import *
from .sandbox import main as main_sandbox

YAML_FILE = f'{Path(__file__).parent.parent}/config_files/useless_microphones.yaml'
BYPASS_ALREADY_RUN = True  # if True, bypass tests that have already been run

def main():
    """Main function (called by default when running script)."""
    
    # Load test parameters from YAML file
    testParams = TestParameters().load_from_yaml(YAML_FILE)

    # Run tests
    res = run_tests(testParams)

    stop = 1


def run_tests(p: TestParameters):
    """Run tests."""
    # Extract number of sensors
    nSensors = len(p.wasnParams.sensorToNodeIndices)

    # Derive all combinations of useless sensors
    combs = list(itertools.product([False, True], repeat=nSensors))
    # Do not consider case where all sensors are noise
    combs = combs[:-1]

    # All sensor indices
    allSensorIndices = np.arange(nSensors)
    out = []
    for ii, comb in enumerate(combs):
        # Render a particular combination of sensors useless
        currComb = allSensorIndices[np.array(comb)]
        # Set current run reference
        runRef = f'comb_{ii+1}_{currComb}'
        # Check if test has already been run
        outputArchiveExportPath = f'{p.exportParams.exportFolder}/out_{runRef}.pkl.gz'
        if BYPASS_ALREADY_RUN and Path(outputArchiveExportPath).exists():
            print(f'>>> Test {runRef} already run. Bypassing...')
            continue

        # Inform user
        print(f'>>> Running test {runRef}...')
        
        # Adapt test parameters and make sure that the RNG seed is reinitialized
        pCurr = copy.deepcopy(p)
        pCurr.setThoseSensorsToNoise = currComb
        pCurr.__post_init__()
        # Complete parameters
        pCurr.danseParams.get_wasn_info(pCurr.wasnParams)
        # Adapt export folder
        pCurr.exportParams.exportFolder = f'{p.exportParams.exportFolder}/{runRef}'
        
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