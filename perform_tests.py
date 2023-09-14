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

CONFIG_FILES_FOLDER = f'{Path(__file__).parent}/config_files'

def main(configFilesFolder: str=CONFIG_FILES_FOLDER):

    t0 = time.time()

    tests.sandbox.main(
        cfgFilename=f'{configFilesFolder}/sandbox_config.yaml',
    )

    print(f'\n\nTotal runtime: {str(datetime.timedelta(seconds=time.time() - t0))}.')

if __name__ == '__main__':
    sys.exit(main())