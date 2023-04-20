# Purpose of script:
#  - Run tests from the `tests` directory.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
# import tests.multiple_runs
import tests.sandbox
import tests.danse_robustness_to_sros
import tests.danse_robustness_to_sros_postproc
import tests.write_yaml_template
import tests.useless_microphones
import tests.useless_microphones_postproc

def main():
    # tests.danse_robustness_to_sros.main()
    # tests.danse_robustness_to_sros_postproc.main()
    # tests.sandbox.main()
    # tests.multiple_runs.main()
    # tests.write_yaml_template.main()
    tests.useless_microphones.main()
    # tests.useless_microphones_postproc.main()

if __name__ == '__main__':
    sys.exit(main())