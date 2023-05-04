# Purpose of script:
#  - Run tests from the `tests` directory.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
from pathlib import Path
import tests.sandbox
import tests.danse_robustness_to_sros
import tests.danse_robustness_to_sros_postproc
import tests.write_yaml_template
import tests.useless_microphones
import tests.useless_microphones_postproc

CONFIG_FILES_FOLDER = f'{Path(__file__).parent}/config_files'

def main():
    # tests.sandbox.main(cfgFilename=f'{CONFIG_FILES_FOLDER}/sandbox_config_batch.yaml')
    tests.sandbox.main(cfgFilename=f'{CONFIG_FILES_FOLDER}/sandbox_config.yaml')
    # tests.danse_robustness_to_sros.main()
    # tests.danse_robustness_to_sros_postproc.main()
    # tests.multiple_runs.main()
    # tests.write_yaml_template.main()
    # tests.useless_microphones.main()
    # tests.useless_microphones_postproc.main()

if __name__ == '__main__':
    sys.exit(main())