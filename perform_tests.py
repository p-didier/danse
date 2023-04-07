# Purpose of script:
#  - Run tests from the `tests` directory.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
# import tests.multiple_runs
import tests.sandbox
import tests.danse_robustness_to_sros

def main():
    # tests.danse_robustness_to_sros.main()
    tests.sandbox.main()
    # tests.multiple_runs.main()

if __name__ == '__main__':
    sys.exit(main())
