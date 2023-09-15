# Purpose of script:
# Test effect of SROs on DANSE performance, for various
# broadcast lengths.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Created on 2023.09.15 - 12:12.

import time
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from .sandbox import main as sandbox_main
from danse_toolbox.d_base import get_divisors
from danse_toolbox.d_classes import TestParameters

N = 1024            # DFT size
WOLA_OVLP = 0.5     # WOLA overlap factor
MK = [1, 1]         # number of sensors per node (`len(MK) = nNodes`)
# MK = [2, 3]         # number of sensors per node (`len(MK) = nNodes`)
BASE_CONFIG_FILE = './danse/config_files/sandbox_config.yaml'

# Broadcast lengths to test
# L_TO_TEST = list(get_divisors(N * (1 - WOLA_OVLP)))
L_TO_TEST = [1, 2, 4, 8, 16, 32]
# SROs to test
SROS = np.linspace(start=0, stop=400, num=5, dtype=int)

# General parameters
EXPORT_FOLDER = './danse/out/battery20230915_sros_asfctofL'
# EXPORT_FOLDER = './danse/out/battery20230915_sros_asfctofL/_quicktest'

def main():
    """Main function (called by default when running script)."""
    
    battery = prepare_test_battery()

    for test in battery:
        print(f"Test {battery.index(test) + 1}/{len(battery)} (ref: {test['ref']}) in progress...")
        t0 = time.time()
        # blockPrint()
        launch(test)  # launch test
        # enablePrint()
        print(f"-- Test {battery.index(test) + 1}/{len(battery)} (ref: {test['ref']}) completed in {time.time() - t0} s.\n")


def prepare_test_battery():
    """Prepare the test battery."""
    
    battery = []
    
    for sro in SROS:
        for l in L_TO_TEST:
            battery.append({
                'N': N,
                'WOLA_OVLP': WOLA_OVLP,
                'MK': MK,
                'L': l,
                'SRO': [0, sro],
                'mode': 'fewSamples',
                'ref': f'L{l}_SRO{sro}',
            })
        battery.append({
            'N': N,
            'WOLA_OVLP': WOLA_OVLP,
            'MK': MK,
            'L': l,
            'SRO': [0, sro],
            'mode': 'wholeChunk',
            'ref': f'wholeChunk_SRO{sro}',
        })
    
    return battery


def launch(test: dict):
    """Launch a test."""
    # Load base parameters from config file
    p = TestParameters().load_from_yaml(BASE_CONFIG_FILE)
    # Adapt parameters
    p.danseParams.DFTsize = test['N']
    p.danseParams.WOLAovlp = test['WOLA_OVLP']
    p.danseParams.broadcastType = test['mode']
    p.danseParams.broadcastLength = test['L']
    p.wasnParams.nSensorPerNode = test['MK']
    p.wasnParams.SROperNode = test['SRO']
    p.exportParams.exportFolder = f'{EXPORT_FOLDER}/{test["ref"]}'
    p.exportParams.wavFiles = False
    p.exportParams.filterNorms = False
    p.exportParams.sroEstimPerfPlot = False
    p.exportParams.acousticScenarioPlot = False
    p.danseParams.get_wasn_info(p.wasnParams)  # complete parameters
    p.__post_init__()

    # Run test
    plt.close('all')
    _ = sandbox_main(p=p)


# Disable printing (https://stackoverflow.com/a/8391735)
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore printing (https://stackoverflow.com/a/8391735)
def enablePrint():
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    sys.exit(main())