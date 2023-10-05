# Purpose of script:
# Test effect of broadcast length on DANSE performance, in the presence of SROs.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Created on 2023.09.19 - 10:57.

import time
import sys, os
import matplotlib.pyplot as plt
from .sandbox import main as sandbox_main
from danse_toolbox.d_base import get_divisors
from danse_toolbox.d_classes import TestParameters

N = 1024            # DFT size
WOLA_OVLP = 0.5     # WOLA overlap factor
# MK = [1, 1]         # number of sensors per node (`len(MK) = nNodes`)
MK = [2, 3]         # number of sensors per node (`len(MK) = nNodes`)
BASE_CONFIG_FILE = './danse/config_files/sandbox_config.yaml'

# Broadcast lengths to test
L_TO_TEST = list(get_divisors(N * (1 - WOLA_OVLP)))
# L_TO_TEST = [1, 2, 4, 8, 16, 32]
# SROs to test
SROS = [0, 200]  # [PPM]

# Booleans
SKIP_ALREADY_RUN_TESTS = True

# General parameters
EXPORT_FOLDER = './danse/out/battery20230919_perf_asfctofL'

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
        # blockPrint()
        # Check if test has already been run
        if SKIP_ALREADY_RUN_TESTS and\
            os.path.exists(f"{exportFolder}/{test['ref']}/metrics.pkl"):
            print(f">>>>>>> Test {battery.index(test) + 1}/{len(battery)} (ref: {test['ref']}) already run. Skipping.")
            continue
        launch(test, baseConfigFile, exportFolder)  # launch test
        # enablePrint()
        print(f">>>>>>> Test {battery.index(test) + 1}/{len(battery)} (ref: {test['ref']}) completed in {time.time() - t} s.\n")

    print(f"\n\nTest battery completed in {time.time() - t0} s.")


def prepare_test_battery():
    """Prepare the test battery."""
    
    battery = []
    
    for l in L_TO_TEST:
        # No compensation
        battery.append({
            'L': l,
            'compensateSRO': False,
            'flagsOn': False,
            'ref': f'L{l}_noComp',
        })
        # SRO compensation without flags
        battery.append({
            'L': l,
            'compensateSRO': True,
            'flagsOn': False,
            'ref': f'L{l}_compNoFlags',
        })
        # SRO compensation incl. flags
        battery.append({
            'L': l,
            'compensateSRO': True,
            'flagsOn': True,
            'ref': f'L{l}_comp',
        })
    
    return battery


def launch(test: dict, baseConfigFile: str, exportFolder: str=EXPORT_FOLDER):
    """Launch a test."""
    # Load base parameters from config file
    p = TestParameters().load_from_yaml(baseConfigFile)
    # Adapt parameters
    p.danseParams.DFTsize = N
    p.danseParams.WOLAovlp = WOLA_OVLP
    p.danseParams.broadcastType = 'fewSamples'  # ensure
    p.danseParams.broadcastLength = test['L']
    p.danseParams.compensateSROs = test['compensateSRO']
    p.danseParams.estimateSROs = 'Oracle'  # ensure
    p.danseParams.includeFSDflags = test['flagsOn']
    p.wasnParams.nSensorPerNode = MK
    p.wasnParams.SROperNode = SROS
    p.exportParams.exportFolder = f'{exportFolder}/{test["ref"]}'
    p.exportParams.wavFiles = False
    p.exportParams.filterNorms = False
    p.exportParams.sroEstimPerfPlot = False
    p.danseParams.get_wasn_info(p.wasnParams)  # complete parameters
    p.__post_init__()

    # Run test
    plt.close('all')
    _ = sandbox_main(p=p)


if __name__ == '__main__':
    sys.exit(main())