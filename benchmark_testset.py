# Set of benchmarking tests for assessing performance of DANSE toolbox at any
# point in its implementation.
#
# ~P. Didier -- 01.12.2022

import sys
import random
import itertools
import numpy as np
from pathlib import Path
from siggen.classes import *
from danse_toolbox.d_classes import *
import danse_toolbox.dataclass_methods as met

SIGNALSPATH = f'{Path(__file__).parent}/testing/sigs'

# List of booleans parameters to consider.
# 01.12.2022: all booleans commutations are considered.
# - 'reverb': if True, reverberant room. Else, anechoic.
# - 'SROs': if True, asynchronous WASN. Else, synchronous.
# - 'SROestcomp': if True, estimate and compensate for SROs. Else, do not.
# - 'simplestWASN': if True, consider a 2-nodes WASN with 1 sensor each.
#    Else, consider a 'realistic' WASN with more nodes, multiple sensors each.
PARAMETERS = ['reverb', 'SROs', 'SROestcomp', 'simplestWASN']

def main():

    # Generate test battery
    bools = list(itertools.product([0, 1], repeat=len(PARAMETERS)))
    tests = []
    for ii in range(len(bools)):
        tests.append(
            dict([(PARAMETERS[jj], bools[ii][jj]) for jj in range(len(PARAMETERS))])
        )

    stop = 1


# @dataclass
# class TestParameters:
#     selfnoiseSNR: int = -50 # [dB] microphone self-noise SNR
#     referenceSensor: int = 0    # Index of the reference sensor at each node
#     wasn: WASNparameters = WASNparameters(
#         sigDur=15
#     )
#     danseParams: DANSEparameters = DANSEparameters()
#     exportFolder: str = f'{Path(__file__).parent}/out'  # folder to export outputs
#     seed: int = 12345

#     def __post_init__(self):
#         np.random.seed(self.seed)  # set random seed
#         random.seed(self.seed)  # set random seed
#         # Check consistency
#         if self.danseParams.nodeUpdating == 'sym' and\
#             any(self.wasn.SROperNode != 0):
#             raise ValueError('Simultaneous node-updating impossible in the presence of SROs.')

#     def save(self, exportType='pkl'):
#         """Saves dataclass to Pickle archive."""
#         met.save(self, self.exportFolder, exportType=exportType)

#     def load(self, foldername, dataType='pkl'):
#         """Loads dataclass to Pickle archive in folder `foldername`."""
#         return met.load(self, foldername, silent=True, dataType=dataType)



if __name__ == '__main__':
    sys.exit(main())
