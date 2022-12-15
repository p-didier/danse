
# Post-processing script for benchmarking tests assessing the performance of
# the DANSE toolbox.
#
# ~P. Didier -- 14.12.2022

import sys
from pathlib import Path

PATHTODATA = f'{Path(__file__).parent}/out/benchmark'
POSTPROCPARAMS = dict([
    (''),   # TODO: selection of parameters to compare, etc.
])

def main() -> None:
    
    # Read data
    read_bm_data()




def read_bm_data():

    # TODO: dir(PATHTODATA) ...
    pass


if __name__ == '__main__':
    sys.exit(main())
