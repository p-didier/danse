# Purpose of script:
#  - Run tests from the `tests` directory.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import time
import datetime
from pathlib import Path
#
import tests.sandbox
import tests.battery20230915_pp
import tests.battery20230915_sros_asfctofL
import tests.battery20230919_perf_asfctofL
import tests.battery20230919_pp

CONFIG_FILES_FOLDER = f'{Path(__file__).parent}/config_files'

def main(configFilesFolder: str=CONFIG_FILES_FOLDER):

    t0 = time.time()

    tests.sandbox.main(
        cfgFilename=f'{configFilesFolder}/sandbox_config.yaml',
    )

    # tests.battery20230915_sros_asfctofL.main(
    #     baseConfigFile=f'{configFilesFolder}/sandbox_config_battery20230915.yaml',
    #     exportFolder='./danse/out/battery20230915_sros_asfctofL_v2',
    # )
    # tests.battery20230915_pp.main(dataFolder='./danse/out/battery20230915_sros_asfctofL_v2')

    # tests.battery20230919_perf_asfctofL.main(
    #     baseConfigFile=f'{configFilesFolder}/sandbox_config_battery20230919.yaml',
    #     exportFolder='./danse/out/battery20230919_perf_asfctofL_v3_20s',
    # )
    # tests.battery20230919_pp.main(dataFolder='./danse/out/battery20230919_perf_asfctofL_v2')

    print(f'\n\nTotal runtime: {str(datetime.timedelta(seconds=time.time() - t0))}.')

if __name__ == '__main__':
    sys.exit(main())