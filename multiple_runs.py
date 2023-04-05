# Purpose of script:
# Running proof-of-concept tests to show the effect of SROs on the functioning
# of TI-DANSE (fast and simple, without YALM config file, for 03.04.2023 
# update meeting w/ G. Enzner and A. Chinaev).
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from sandbox import TestParameters
from sandbox import main as main_sandbox
from danse_toolbox.d_classes import *
from danse_toolbox.d_base import DANSEparameters, CohDriftParameters, PrintoutsAndPlotting

SEED = 12347
SIGNALS_PATH = f'{Path(__file__).parent}/testing/sigs'
BYPASS_DYNAMIC_PLOTS = True  # if True, bypass all runtime (dynamic) plotting
SKIP_EXISTING_FOLDERS = True
IMPOSED_YLIM_SNR = 30  # if None, use auto ylim
#
OUT_FOLDER = '20230405_tests/battery/Mk3'  # export path relative to `danse/out`

@dataclass
class GlobalTestParameters:
    sros : np.ndarray = np.array([0])   # SROs to test
    nodeUpdatings : np.ndarray = np.array(['seq'])   # node-updating schemes to test
    RTs : np.ndarray = np.array([0.0])   # reverberation times to test
    nSensors : list = field(default_factory=list)   # number of sensors per node to test
    gevdBool : list = field(default_factory=list)   # GEVD booleans to test

params = GlobalTestParameters(
    sros=np.array([
        (np.array([0, 0]), 'sync'),
        (np.array([20, 40]), 'sSROs'),
        (np.array([50, 100]), 'mSROs'),
        (np.array([200, 400]), 'lSROs')
    ]),
    nodeUpdatings=np.array([
        # 'seq',
        'asy'
    ]),
    RTs=np.array([
        # 0.0,
        0.2
    ]),
    nSensors=[
        # [1, 1, 1],
        # [1, 2, 3],
        [3, 3, 3],
    ],
    # gevdBool=[True, False],
    # gevdBool=[True],
    gevdBool=[False],
)

def main():
    """Main function (called by default when running script)."""

    # Run tests
    run_test_batch(params)


def run_test_batch(params: GlobalTestParameters):
    """
    Runs a test batch based on a (YAML) configuration.
    
    Parameters
    ----------
    TODO:
    
    Returns
    ----------
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """
    
    for idxSROs in range(len(params.sros)):
        currSROs = np.concatenate((np.array([0.]), params.sros[idxSROs][0]))
        strSROs = params.sros[idxSROs][1]
        for idxNUs in range(len(params.nodeUpdatings)):
            currNodeUpdating = params.nodeUpdatings[idxNUs]
            for idxNSs in range(len(params.nSensors)):
                currSensors = params.nSensors[idxNSs]
                if all(np.array(currSensors) == 1):
                    strSensors = 'sNodes'
                else:
                    strSensors = 'mNodes'
                for idxRT in range(len(params.RTs)):
                    currRT = params.RTs[idxRT]
                    for idxGEVDbool in range(len(params.gevdBool)):
                        currGEVDbool = params.gevdBool[idxGEVDbool]
                        if currGEVDbool:
                            strGEVD = '_gevd'
                        else:
                            strGEVD = ''

                        folderName = f'{strSROs}_{currNodeUpdating}_{strSensors}_{int(currRT * 1000)}msRT{strGEVD}'
                        print(f'CURRENT FOLDER: {folderName}')
                        fullExportPath = f'{Path(__file__).parent}/out/{OUT_FOLDER}/{folderName}',
                        fullExportPath = fullExportPath[0]
                        if Path(fullExportPath).is_dir() and SKIP_EXISTING_FOLDERS:
                            print(f'The folder exists already --> skipping iteration (SKIP_EXISTING_FOLDERS==True)')
                            continue

                        # Create test parameters
                        testParams = TestParameters(
                            exportFolder=fullExportPath,
                            seed=SEED,
                            snrYlimMax=IMPOSED_YLIM_SNR,  # =========================<<<<<
                            wasnParams=WASNparameters(
                                VADenergyDecrease_dB=35,  # [dB]
                                topologyParams=TopologyParameters(
                                    topologyType='user-defined',
                                    commDistance=4.,  # [m]
                                    seed=SEED,
                                    userDefinedTopo=np.array([
                                        [1, 1, 0],  # Node 1
                                        [1, 1, 1],  # Node 2
                                        [0, 1, 1],  # Node 3
                                    ]),
                                ),
                                sigDur=15,
                                rd=np.array([5, 5, 5]),
                                fs=16000,
                                t60=currRT,  # =========================<<<<<
                                interSensorDist=0.2,
                                nNodes=3,
                                nSensorPerNode=currSensors,  # =========================<<<<<
                                selfnoiseSNR=99,
                                desiredSignalFile=[f'{SIGNALS_PATH}/01_speech/{file}'\
                                    for file in [
                                        'speech1.wav',
                                        'speech2.wav'
                                    ]],
                                noiseSignalFile=[f'{SIGNALS_PATH}/02_noise/{file}'\
                                    for file in [
                                        'ssn/ssn_speech1.wav',
                                        'ssn/ssn_speech2.wav'
                                    ]],
                                SROperNode=currSROs,  # =========================<<<<<
                            ),
                            danseParams=DANSEparameters(
                                DFTsize=1024,
                                WOLAovlp=.5,
                                nodeUpdating=currNodeUpdating,  # =========================<<<<<
                                broadcastType='wholeChunk',
                                estimateSROs='CohDrift',
                                compensateSROs=False,
                                cohDrift=CohDriftParameters(
                                    loop='open',
                                    alpha=0.95
                                ),
                                filterInitType='selectFirstSensor_andFixedValue',
                                filterInitFixedValue=1,
                                computeCentralised=True,
                                computeLocal=True,
                                noExternalFilterRelaxation=False,
                                performGEVD=currGEVDbool,  # =========================<<<<<
                                t_expAvg50p=10,
                                timeBtwExternalFiltUpdates=1,
                                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                printoutsAndPlotting=PrintoutsAndPlotting(
                                    showWASNs=False if BYPASS_DYNAMIC_PLOTS else True,
                                    onlySNRandESTOIinPlots=True
                                )
                            )
                        )
                        # Complete parameters
                        testParams.danseParams.get_wasn_info(testParams.wasnParams)

                        # Run 
                        main_sandbox(testParams)

            stop = 1


if __name__ == '__main__':
    sys.exit(main())