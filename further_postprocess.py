# Purpose of script:
# Launch post-processing of results from a test.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import tests.postprocess_script as pp

BASE_RESULTS_FOLDER = f'{Path(__file__).parent}/out/'
# SUBFOLDER = '20231006_tests/metrics_comp/test3_as2_selfnoise15dB'
SUBFOLDERS = ['battery20231013_perf_asfctofSelfNoise/' + x\
            for x in os.listdir(f'{BASE_RESULTS_FOLDER}/battery20231013_perf_asfctofSelfNoise')\
            if x[0] != '_']
SUBFOLDERS = SUBFOLDERS[-4:]

def main(subfolders: str=SUBFOLDERS):
    """Main function (called by default when running script)."""
    
    if isinstance(subfolders, str):
        subfolders = [subfolders]

    for sub in subfolders:
        # Build full path to results file
        foldername = f'{BASE_RESULTS_FOLDER}/{sub}'
        # Perform post-processing
        pp.main(foldername, params={
            'frameDur': 5,   # [s] duration of metrics computation frame.
                                # If `None`, use entire signal, starting from the start time.
            'tmax': 30,
            'frameShift': 1
        })
        # pp.pp_20231006_combine_metrics(foldername, frameDur=None)


if __name__ == '__main__':
    sys.exit(main())