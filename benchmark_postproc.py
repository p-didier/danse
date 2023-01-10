
# Post-processing script for benchmarking tests assessing the performance of
# the DANSE toolbox.
#
# ~P. Didier -- 14.12.2022

import sys, os
from pathlib import Path
from dataclasses import dataclass, field

PATHTODATA = f'{Path(__file__).parent}/out/benchmark__dec2022'
POSTPROCPARAMS = dict([
    (''),
])

@dataclass
class BenchmarkData:
    parameters: list[str] = field(default_factory=list)
    # ^^^ list of benchmark test parameters
    path: str = ''  # path to benchmark data



def main() -> None:

    # Initialise
    bmData = init()
    
    # Read data
    read_bm_data(bmData)


def init():
    """Initialisation."""

    path = PATHTODATA

    bmd = BenchmarkData(
        parameters=get_param_from_folder_names(path),
        path=path
    )

    def get_param_from_folder_names(path):
        """Extracts names of test parameters from subfolder name."""
        subdirs = [f.path for f in os.scandir(path) if f.is_dir()]
        words = Path(subdirs[0]).stem.split('_')
        params = [w[:-1] for w in words]
        return params

    return bmd


def read_bm_data(bmData: BenchmarkData):
    """Reads benchmark data."""

    subdirs = [f.path for f in os.scandir(bmData.path) if f.is_dir()]
    # Folder selection


if __name__ == '__main__':
    sys.exit(main())
