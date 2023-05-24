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

    # Benchmark tests for the TI-GEVD-DANSE online implementation.
    # Various combinations of the following parameters are tested:
    #   - reverberation time
    #   - number of sensors per node
    #   - per-node sampling rate offsets
    #   - time constant for exponential averaging
    # 
    # tests.benchmark_danse.main(
    #     cfgFilename=f'{CONFIG_FILES_FOLDER}/sandbox_config_tigevddanse_week19_2023.yaml',
    #     outputFolder=f'{Path(__file__).parent}/out/20230512_tests/tigevddanse_battery',
    # )
    # tests.benchmark_danse_postproc.main(
    #     resultsFolder=f'{Path(__file__).parent}/out/20230512_tests/tigevddanse_battery',
    #     suptitlePrefix='TI-GEVD-DANSE',
    # )

    # Test the inherent robustness of the DANSE algorithm to SROs.
    #
    tests.danse_robustness_to_sros.main(
        cfgFilename=f'{CONFIG_FILES_FOLDER}/sros_effect.yaml',
        outputFolder='20230524_tests/sros_effect/batch_danse',  # relative to `danse/out`
    )
    # tests.danse_robustness_to_sros_postproc.main(
    #     folder=f'{Path(__file__).parent}/out/20230508_tests/sros_effect_danse'
    # )

    # ylim_adjust_barplots.main(
    #     resultsFolder=f'{Path(__file__).parent}/out/20230512_tests/tigevddanse_battery',
    #     resultsFilename='DANSEoutputs.pkl.gz',
    #     knowSNRylims=[0, 35],
    # )

    print(f'\n\nTotal runtime: {str(datetime.timedelta(seconds=time.time() - t0))}.')

if __name__ == '__main__':
    sys.exit(main())