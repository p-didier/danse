# Purpose of script:
#  - Run tests from the `tests` directory.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
# import tests.multiple_runs
import tests.sandbox
import tests.danse_robustness_to_sros
import tests.write_yaml_template

def main():
    # tests.danse_robustness_to_sros.main()
    tests.sandbox.main()
    # tests.multiple_runs.main()
    # tests.write_yaml_template.main()

if __name__ == '__main__':
    sys.exit(main())
