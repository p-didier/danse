# Purpose of script:
# Compare filter coefficients between different runs of DANSE.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import gzip
import pickle
import numpy as np
from pathlib import Path

BASE_PATH_TO_RESULTS = f'{Path(__file__).parent.parent}/out'

# RUNS = [
#     f'{BASE_PATH_TO_RESULTS}/20230505_tests/batch/test1_seqNU',
#     f'{BASE_PATH_TO_RESULTS}/20230505_tests/batch/test2_asyNU',
# ]
RUNS = [
    f'{BASE_PATH_TO_RESULTS}/20230505_tests/batch/centr/test3_ssnodes_seq',
]

FREQ_BIN_IDX = 100
ITERATION_IDX = 120

def main():
    """Main function (called by default when running script)."""
    
    for path in RUNS:
        # Find filter pickle file
        filterPickleFile = f'{path}/filters.pkl.gz'
        if not Path(filterPickleFile).exists():
            print(f'No filter pickle file found in {path}.')
            continue

        # Load filter
        print(f'Loading filter from {filterPickleFile}...')
        filterObj = pickle.load(gzip.open(filterPickleFile, 'r'))

        # Print last filter coefficients
        for k in range(len(filterObj)):
            print(f'Filter coefficients node {k + 1}, freq. bin #{FREQ_BIN_IDX}, iter. #{ITERATION_IDX}: {np.round(filterObj[k][FREQ_BIN_IDX, ITERATION_IDX, :], 3)}')

        # Check if centralized filters have been computed and exported
        filterCentrPickleFile = f'{path}/filters.pkl.gz'
        if Path(filterCentrPickleFile).exists():
            # Load filter
            print(f'Loading centralized filter from {filterCentrPickleFile}...')
            filterObj = pickle.load(gzip.open(filterCentrPickleFile, 'r'))

            # Print last filter coefficients
            for k in range(len(filterObj)):
                print(f'Centralized filter coefficients node {k + 1}, freq. bin #{FREQ_BIN_IDX}, iter. #{ITERATION_IDX}: {np.round(filterObj[k][FREQ_BIN_IDX, ITERATION_IDX, :], 3)}')

if __name__ == '__main__':
    sys.exit(main())