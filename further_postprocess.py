# Purpose of script:
# Launch post-processing of results from a test.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
from pathlib import Path
import tests.postprocess_script as pp

SUBFOLDER = '20230908_tests/efficient_pp_tests/test1'
BASE_RESULTS_FOLDER = f'{Path(__file__).parent}/out/'

def main():
    """Main function (called by default when running script)."""
    
    # Build full path to results file
    foldername = f'{BASE_RESULTS_FOLDER}/{SUBFOLDER}'

    # Perform post-processing
    pp.main(foldername)

if __name__ == '__main__':
    sys.exit(main())