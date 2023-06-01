# Purpose of script:
# Automated tests for the effect of including diffuse noise in the DANSE
# simulations.
# >> Corresponding journal entry: 2023, week22, THU.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import copy
import tests.sandbox
from pathlib import Path
from danse_toolbox.d_classes import TestParameters

# Diffuse noise SNRs to consider
DN_SNRS = [0, 5, 10, 15, 20, 25, 30, 35, 40]  # dB
PATH_TO_BASE_CONFIG_FILE = f'{Path(__file__).parent}/config_files/sandbox_config.yaml'
BASE_EXPORT_PATH = f'{Path(__file__).parent}/out/20230601_tests/diffusenoise_effect'

def main(
        dnSNRs=DN_SNRS,
        baseCfgFilename: str=PATH_TO_BASE_CONFIG_FILE,
        baseExportPath: str=BASE_EXPORT_PATH
    ):
    """Main function (called by default when running script)."""
    # Load parameters from base config file
    print('Loading base parameters...')
    p = TestParameters().load_from_yaml(baseCfgFilename)
    p.danseParams.get_wasn_info(p.wasnParams)  # complete parameters
    print('Base parameters loaded.')

    for ii in range(len(dnSNRs)):
        currDNSNR = int(dnSNRs[ii])
        print(f'\n>>> Running test {ii+1}/{len(dnSNRs)} for diffuse noise SNR = {currDNSNR} dB...')
        currParams = copy.deepcopy(p)
        currParams.wasnParams.diffuseNoiseSNR = currDNSNR
        currParams.exportParams.exportFolder =\
            f'{baseExportPath}/dnSNR_{currDNSNR}dB'
        # Complete parameters
        currParams.__post_init__()
        currParams.danseParams.get_wasn_info(p.wasnParams)
        currParams.danseParams.printoutsAndPlotting.verbose = False
        # Run test
        tests.sandbox.main(p=currParams)

if __name__ == '__main__':
    sys.exit(main())