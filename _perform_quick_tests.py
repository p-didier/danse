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
import _figures_edits.for_20230531_SOUNDSSCmeeting.combine_sro_plots as soundssc10

CONFIG_FILES_FOLDER = f'{Path(__file__).parent}/config_files'

def main():

    t0 = time.time()

    # test_asc_plot(
    #     cfgFilename=f'{CONFIG_FILES_FOLDER}/sandbox_config.yaml'
    # )

    # test_get_stft_istft()
    
    # 31.05.2023 -- SOUNDS SC #10 meeting figures
    soundssc10.main(
        folderOnline=f'{Path(__file__).parent}/out/20230530_tests/sros_effect/online_danse_extFilt',
        folderBatch=f'{Path(__file__).parent}/out/20230530_tests/sros_effect/fullbatch_danse'
    )

    print(f'\n\nTotal runtime: {str(datetime.timedelta(seconds=time.time() - t0))}.')

if __name__ == '__main__':
    sys.exit(main())