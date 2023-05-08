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

CONFIG_FILES_FOLDER = f'{Path(__file__).parent}/config_files'

def main():

    t0 = time.time()

    tests.sandbox.main(
        cfgFilename=f'{CONFIG_FILES_FOLDER}/sandbox_config_tigevddanse_week19_2023.yaml',
    )
    # tests.danse_robustness_to_sros.main()
    # tests.danse_robustness_to_sros_postproc.main()
    # tests.multiple_runs.main()
    # tests.write_yaml_template.main()
    # tests.useless_microphones.main()
    # tests.useless_microphones_postproc.main()


    print(f'\n\nTotal runtime: {str(datetime.timedelta(seconds=time.time() - t0))}.')

if __name__ == '__main__':
    sys.exit(main())