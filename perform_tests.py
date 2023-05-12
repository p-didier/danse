# Purpose of script:
#  - Run tests from the `tests` directory.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import time
import datetime
from pathlib import Path
import tests.sandbox
import tests.danse_robustness_to_sros
import tests.danse_robustness_to_sros_postproc
import tests.write_yaml_template
import tests.useless_microphones
import tests.useless_microphones_postproc
import tests.benchmark_danse
import tests.benchmark_danse_postproc
import out.format_adjustement_scripts.ylim_adjust_barplots as ylim_adjust_barplots

CONFIG_FILES_FOLDER = f'{Path(__file__).parent}/config_files'

def main():

    t0 = time.time()

    # tests.sandbox.main(
    #     cfgFilename=f'{CONFIG_FILES_FOLDER}/sandbox_config.yaml',
    # )

    # tests.benchmark_danse.main(
    #     cfgFilename=f'{CONFIG_FILES_FOLDER}/sandbox_config_tigevddanse_week19_2023.yaml',
    #     outputFolder=f'{Path(__file__).parent}/out/20230512_tests/tigevddanse_battery',
    # )
    tests.benchmark_danse_postproc.main(
        resultsFolder=f'{Path(__file__).parent}/out/20230512_tests/tigevddanse_battery',
        suptitlePrefix='TI-GEVD-DANSE',
    )

    # tests.danse_robustness_to_sros.main()
    # tests.danse_robustness_to_sros_postproc.main(
    #     folder=f'{Path(__file__).parent}/out/20230508_tests/sros_effect_danse'
    # )

    # ylim_adjust_barplots.main(
    #     resultsFolder=f'{Path(__file__).parent}/out/20230508_tests/tigevddanse',
    #     resultsFilename='DANSEoutputs.pkl.gz',
    #     knowSNRylims=[0, 30],
    # )


    print(f'\n\nTotal runtime: {str(datetime.timedelta(seconds=time.time() - t0))}.')

if __name__ == '__main__':
    sys.exit(main())