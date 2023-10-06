# Purpose of script:
# Launch post-processing of results from a test.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
from pathlib import Path
import tests.postprocess_script as pp

BASE_RESULTS_FOLDER = f'{Path(__file__).parent}/out/'
SUBFOLDER = '20231006_tests/metrics_comp/test3_as2_selfnoise15dB'

def main():
    """Main function (called by default when running script)."""
    
    # Build full path to results file
    foldername = f'{BASE_RESULTS_FOLDER}/{SUBFOLDER}'

    # Perform post-processing
    pp.main(foldername, params={
        'frameDur': None,  # [s] duration of metrics computation frame. If `None`, use entire signal, starting from the start time
        'tmax': 20,
        'frameShift': 1
    })
    pp.pp_20231006_combine_metrics(foldername, frameDur=None)

if __name__ == '__main__':
    sys.exit(main())