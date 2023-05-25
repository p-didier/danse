# Purpose of script:
#  - Run tests from the `_quick_tests` directory.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import time
import datetime
from pathlib import Path
from _quick_tests.test_asc_plot import main as test_asc_plot
from _quick_tests.test_get_stft_istft import main as test_get_stft_istft

CONFIG_FILES_FOLDER = f'{Path(__file__).parent}/config_files'

def main():

    t0 = time.time()

    # test_asc_plot(
    #     cfgFilename=f'{CONFIG_FILES_FOLDER}/sandbox_config.yaml'
    # )

    test_get_stft_istft()

    print(f'\n\nTotal runtime: {str(datetime.timedelta(seconds=time.time() - t0))}.')

if __name__ == '__main__':
    sys.exit(main())