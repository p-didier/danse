# Purpose of script:
#  - Run tests from the `tests` directory.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import tests.multiple_runs
import tests.sandbox
import tests.danse_robustness_to_sros

# TEST_TO_RUN = tests.multiple_runs.main
# TEST_TO_RUN = tests.sandbox.main
TEST_TO_RUN = tests.danse_robustness_to_sros.main

def main():
    """Main function (called by default when running script)."""
    TEST_TO_RUN()

if __name__ == '__main__':
    sys.exit(main())
