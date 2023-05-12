# Purpose of script:
# Benchmark tests for the TI-GEVD-DANSE online implementation.
# >> Journal entry reference: 2023, week19, MON. 
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import tests.sandbox as sandbox
from danse_toolbox.d_classes import TestParameters

REPO_ROOT_PATH = f'{Path(__file__).parent.parent}'  # Path to the root of the repository
OUTPUT_FOLDER = f'{REPO_ROOT_PATH}/out/20230508_tests/tigevddanse'  # Path to the output folder
BASE_CONFIG_FILE_PATH = f'{REPO_ROOT_PATH}/config_files/sandbox_config_tigevddanse_week19_2023.yaml'  # Path to the base config file

SKIP_IF_ALREADY_RUN = True  # if True, skip tests that have already been run

# Test parameters
TEST_PARAMS = {
    't60': [0.0, 0.2],  # reveberation time [s]
    'Mk': [[1, 1, 1], [1, 2, 3]],   # number of sensor per node
    'Mk_label': ['ss', 'ms'],       # label for number of sensor per node
    'SRO': [[0, 0, 0], [0, 50, 100], [0, 200, 400]],  # per-node sampling rate offsets [PPM]
    'SRO_label': ['noSROs', 'mediumSROs', 'largeSROs'],  # label for per-node sampling rate offsets
    'tau': [2, 10, 30],  # time constant for exponential averaging [s]
    # 'tau': [2, 30],  # time constant for exponential averaging [s]
}

def main(
        testParams=TEST_PARAMS,
        outputFolder=OUTPUT_FOLDER,
        cfgFilename=BASE_CONFIG_FILE_PATH,
    ):
    """Main function (called by default when running script)."""
    
    nTestsRemaining = len(testParams['t60']) * len(testParams['Mk']) * len(testParams['SRO']) * len(testParams['tau'])
    print(f'Running {nTestsRemaining} tests...')

    # Define combinations
    comb = []
    for tau in testParams['tau']:
        for idxSROs, sros in enumerate(testParams['SRO']):
            for idxMk, nSensorPerNode in enumerate(testParams['Mk']):
                for rt in testParams['t60']:
                    comb.append({
                        'tau': tau,
                        'sros': sros,
                        'sros_label': testParams['SRO_label'][idxSROs],
                        'nSensorPerNode': nSensorPerNode,
                        'nSensorPerNode_label': testParams['Mk_label'][idxMk],
                        'rt': rt,
                    })

    # Run tests
    for currParams in comb:
        tau = currParams['tau']
        sros = currParams['sros']
        nSensorPerNode = currParams['nSensorPerNode']
        rt = currParams['rt']
        labelSROs = currParams['sros_label']
        labelMk = currParams['nSensorPerNode_label']

        nTestsRemaining -= 1
        print(f'\n>>> Running test with tau={tau} s, sros={sros} PPM, Mk={nSensorPerNode}, t60={rt} s...({nTestsRemaining} tests remaining.)\n')

        # Build test name
        testName = f'tau{tau}s_{labelSROs}_{labelMk}_rt{int(round(rt * 1e3))}ms'

        # Skip test if already run
        if SKIP_IF_ALREADY_RUN:
            if Path(f'{outputFolder}/{testName}').is_dir():
                print(f'Skipping test {testName} because it has already been run.')
                continue

        # Run test
        run_test(
            tau,
            sros,
            nSensorPerNode,
            rt,
            testName,
            outputFolder,
            cfgFilename
        )


def run_test(
        tau,
        sros,
        nSensorPerNode,
        rt,
        testName,
        outputFolder,
        baseConfigFile
    ):
    """Run a single test."""

    # Load parameters from config file
    print('Loading parameters...')
    p = TestParameters().load_from_yaml(baseConfigFile)
    print('Parameters loaded.')

    # Update parameters
    p.danseParams.t_expAvg50p = tau
    p.danseParams.t_expAvg50pExternalFilters = tau
    p.wasnParams.SROperNode = sros
    p.wasnParams.nSensorPerNode = nSensorPerNode
    p.wasnParams.t60 = rt
    p.wasnParams.__post_init__()  # re-initialize WASN parameters
    p.exportParams.exportFolder = f'{outputFolder}/{testName}'
    p.__post_init__()  # re-initialize WASN parameters
    p.danseParams.get_wasn_info(p.wasnParams)  # complete parameters
    p.danseParams.printoutsAndPlotting.verbose = False  # disable verbose mode
    print(f'Parameters updated for current test (tau={tau} s, sros={sros} PPM, Mk={nSensorPerNode}, t60={rt} s).')
    print(f'(test name: "{testName}")')

    # Ensure all figures are closed
    plt.close('all')

    # Run test
    sandbox.main(p=p)


if __name__ == '__main__':
    sys.exit(main())