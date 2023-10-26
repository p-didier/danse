# Purpose of script:
# Test effect of self-noise level on GEVD-DANSE vs. GEVD-MWF performance,
# and link to choice of time segment for metrics computation - no SROs involved. 
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Created on 2023.10.13 - 13:30.

import time
import sys, os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from .sandbox import main as sandbox_main
from danse_toolbox.d_classes import TestParameters

# General parameters
EXPORT_FOLDER = f'./danse/out/{Path(__file__).stem}'
MK = [2, 3]  # number of sensors per node
BASE_CONFIG_FILE = './danse/config_files/sandbox_config.yaml'

# SROs to test
SN_SNR_TO_TEST = np.linspace(start=0, stop=50, num=11)  # [PPM]

# Booleans
SKIP_ALREADY_RUN_TESTS = True

def main(
        baseConfigFile: str=BASE_CONFIG_FILE,
        exportFolder: str=EXPORT_FOLDER,
    ):
    """Main function (called by default when running script)."""
    
    battery = prepare_test_battery()
    print(f"Test battery: {len(battery)} tests.")

    t0 = time.time()
    for test in battery:
        print('----------------------------------------')
        print(f"Test {battery.index(test) + 1}/{len(battery)} (ref: {test['ref']}) in progress...")
        print('----------------------------------------')
        t = time.time()
        # Check if test has already been run
        if SKIP_ALREADY_RUN_TESTS and\
            os.path.exists(f"{exportFolder}/{test['ref']}/metrics.pkl"):
            print(f">>>>>>> Test {battery.index(test) + 1}/{len(battery)} (ref: {test['ref']}) already run. Skipping.")
            continue
        launch(test, baseConfigFile, exportFolder)  # launch test
        print(f">>>>>>> Test {battery.index(test) + 1}/{len(battery)} (ref: {test['ref']}) completed in {time.time() - t} s.\n")

    print(f"\n\nTest battery completed in {time.time() - t0} s.")


def prepare_test_battery():
    """Prepare the test battery."""
    
    battery = []
    
    for snSNR in SN_SNR_TO_TEST:
        # No compensation
        battery.append({
            'selfNoiseSNR': snSNR,
            'ref': f'snsnr_{snSNR:.0f}dB',
        })
    
    return battery


def launch(
        test: dict,
        baseConfigFile: str,
        exportFolder: str=EXPORT_FOLDER
    ):
    """Launch a test."""
    # Load base parameters from config file
    p = TestParameters().load_from_yaml(baseConfigFile)
    # Set baseline parameters
    p.danseParams.broadcastType = 'wholeChunk'  # ensure
    p.danseParams.compensateSROs = False
    p.wasnParams.nSensorPerNode = MK
    p.wasnParams.SROperNode = [0]  # no SROs
    p.exportParams.wavFiles = False
    p.exportParams.filterNorms = False
    p.exportParams.sroEstimPerfPlot = False
    p.wasnParams.sigDur = 30  # seconds
    p.danseParams.startComputeMetricsAt ='after_5s'  # seconds
    p.danseParams.performGEVD = True
    # Set test-specific parameters
    p.exportParams.exportFolder = f'{exportFolder}/{test["ref"]}'
    p.wasnParams.selfnoiseSNR = test['selfNoiseSNR']

    p.danseParams.get_wasn_info(p.wasnParams)  # complete parameters
    p.__post_init__()

    # Run test
    plt.close('all')
    _ = sandbox_main(p=p)


if __name__ == '__main__':
    sys.exit(main())