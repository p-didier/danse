# Purpose of script:
# Automated tests for the effect of including diffuse noise in the DANSE
# simulations.
# >> Corresponding journal entry: 2023, week22, THU.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import copy
import numpy as np
import tests.sandbox
from pathlib import Path
from danse_toolbox.d_classes import TestParameters

# Diffuse noise SNRs to consider
DN_POWER_FACTORS = list(np.flipud(np.arange(
    start=-30,
    stop=15,
    step=5,
)))
PATH_TO_BASE_CONFIG_FILE = f'{Path(__file__).parent}/config_files/sandbox_config.yaml'
BASE_EXPORT_PATH = f'{Path(__file__).parent}/out/20230601_tests/diffusenoise_effect'

def main(
        dnPowFacts=DN_POWER_FACTORS,
        baseCfgFilename: str=PATH_TO_BASE_CONFIG_FILE,
        baseExportPath: str=BASE_EXPORT_PATH
    ):
    """Main function (called by default when running script)."""
    # Load parameters from base config file
    print('Loading base parameters...')
    p = TestParameters().load_from_yaml(baseCfgFilename)
    p.danseParams.get_wasn_info(p.wasnParams)  # complete parameters
    print('Base parameters loaded.')

    for ii in range(len(dnPowFacts)):
        currPowFact = int(dnPowFacts[ii])
        print(f'\n>>> Running test {ii+1}/{len(dnPowFacts)} for diffuse noise power factor = {currPowFact} dB...')
        currParams = copy.deepcopy(p)
        currParams.wasnParams.diffuseNoise = True
        currParams.wasnParams.diffuseNoisePowerFactor = currPowFact
        currParams.exportParams.exportFolder =\
            f'{baseExportPath}/dnPowFact_{currPowFact}dB'
        # Complete parameters
        currParams.__post_init__()
        currParams.danseParams.get_wasn_info(p.wasnParams)
        currParams.danseParams.printoutsAndPlotting.verbose = False
        # Run test
        tests.sandbox.main(p=currParams)

if __name__ == '__main__':
    sys.exit(main())