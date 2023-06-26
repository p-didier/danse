# Purpose of script:
# Monte-Carlo runs for DANSE-related experiments. 
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import copy
import numpy as np
from danse_toolbox.d_mc import MCExperiment
from danse_toolbox.d_classes import TestParameters
from .sandbox import main as sandbox_main

def main(mcCfgFileName: str):
    """Main function (called by default when running script)."""
    mcExpConfig = MCExperiment().load_from_yaml(mcCfgFileName)

    # Set random seed
    if mcExpConfig.seed is not None:
        np.random.seed(mcExpConfig.seed)
    # Get base parameters from config file
    tpBase = TestParameters().load_from_yaml(mcExpConfig.baseConfigFile)

    for idxMCrun in range(mcExpConfig.nMCruns):

        print(f'Running MC run {idxMCrun+1}/{mcExpConfig.nMCruns}...')

        # Adapt parameters
        tp = copy.deepcopy(tpBase)
        tp.wasnParams.signalType = 'random'     # random signals
        # tp.wasnParams.trueRoom = False          # no true room (random IRs)

        # Run test
        sandbox_main(
            p=tp,
            plotASCearly=False,
            cfgFilename='',
        )

    stop = 1

if __name__ == '__main__':
    sys.exit(main())