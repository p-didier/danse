# Purpose of script:
#  - Test the inherent robustness of the DANSE algorithm to SROs.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from danse_toolbox.d_classes import *
from .sandbox import main as main_sandbox
from danse_toolbox.d_post import DANSEoutputs
from danse_toolbox.d_base import DANSEparameters, CohDriftParameters, PrintoutsAndPlotting

PATH_TO_CONFIG_FILE = f'{Path(__file__).parent.parent}/config_files/sros_effect_20230406.yaml'
SIGNALS_PATH = f'{Path(__file__).parent.parent}/tests/sigs'

def main():
    """Main function (called by default when running script)."""
    cfg = read_config(filePath=PATH_TO_CONFIG_FILE)

    # Run tests
    res = run_test_batch(cfg)

    # Post-process results
    fig = post_process_results(res)
    plt.show()
    stop = 1

    return None


def read_config(filePath: str):
    """
    Reads the YAML configuration file.
    
    Parameters
    ----------
    filePath : str
        Path to YAML configuration file.
    
    Returns
    ----------
    cfg : dict
        Configuration object.
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """
    with open(filePath, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)

    return cfg


def run_test_batch(cfg: dict):
    """
    Runs a test batch based on a (YAML) configuration.
    
    Parameters
    ----------
    cfg : dict
        Configuration object.
    
    Returns
    ----------
    allVals : list[dict]
        List of dictionaries containing the speech enhancement metrics results.
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """
    
    allVals = []
    for ii, sros in enumerate(cfg['sros']):
        # Set up SRO parameter
        currSROs = np.concatenate((np.array([0.]), np.array(sros)))
        print(f'Running SRO test {ii+1}/{len(cfg["sros"])}: {currSROs} PPM.')
        # Set up test parameters
        testParams = setup_test_parameters(cfg, currSROs)
        # Run test
        res = main_sandbox(p=testParams, bypassPostprocess=True)
        # Extract single test results
        vals = extract_single_test_results(res)
        allVals.append(vals)

    return allVals


def extract_single_test_results(res: DANSEoutputs):
    """
    Extracts the results of a single SRO test run.
    
    Parameters
    ----------
    res : DANSEoutputs
        DANSE outputs object.
    
    Returns
    ----------
    vals : dict
        Dictionary containing the speech enhancement metrics results.
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """
    vals = dict([
        ('snr', res.metrics.snr),
        ('estoi', res.metrics.stoi),
        ('sros', res.SROgroundTruth)
    ])
    return vals


def setup_test_parameters(cfg: dict, currSROs: np.ndarray) -> TestParameters:
    """
    Sets up the test parameters.
    
    Parameters
    ----------
    cfg : dict
        Configuration object.
    currSROs : np.ndarray
        Current SROs per node.
    
    Returns
    ----------
    testParams : TestParameters
        Test parameters object.
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """

    # Create test parameters
    testParams = TestParameters(
        exportFolder='',  # <-- no export folder
        seed=cfg['seed'],
        wasnParams=WASNparameters(
            layoutType='vert_spinning_top',
            VADenergyDecrease_dB=35,  # [dB]
            topologyParams=TopologyParameters(
                topologyType='user-defined',
                commDistance=4.,  # [m]
                seed=cfg['seed'],
                userDefinedTopo=np.array([
                    [1, 1, 0],  # Node 1
                    [1, 1, 1],  # Node 2
                    [0, 1, 1],  # Node 3
                ]),
            ),
            sigDur=cfg['sigDur'],  # [s]
            rd=np.array([5, 5, 5]),
            fs=16000,
            t60=cfg['t60'],  # =========================<<<<<
            interSensorDist=0.2,
            nNodes=3,
            nSensorPerNode=cfg['nSensorPerNode'],  # =========================<<<<<
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
            nodeUpdating=cfg['nodeUpdating'],  # =========================<<<<<
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
            performGEVD=cfg['performGEVD'],  # =========================<<<<<
            t_expAvg50p=10,
            timeBtwExternalFiltUpdates=1,
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            printoutsAndPlotting=PrintoutsAndPlotting(
                showWASNs=False,
                onlySNRandESTOIinPlots=True
            )
        )
    )
    # Complete parameters
    testParams.danseParams.get_wasn_info(testParams.wasnParams)

    return testParams


def post_process_results(res: list[dict]):
    """
    Post-processes the results of a test batch.
    
    Parameters
    ----------
    res : list[dict]
        Results.
    
    Returns
    ----------
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """
    # Convert to dataframe
    df = pd.DataFrame(res)
    # Compute mean and standard deviation
    df_mean = df.groupby('sros').mean()
    df_std = df.groupby('sros').std()
    # Plot
    fig, ax = plt.subplots()
    ax.errorbar(df_mean.index, df_mean['snr'], yerr=df_std['snr'], label='SNR')
    ax.errorbar(df_mean.index, df_mean['estoi'], yerr=df_std['estoi'], label='ESTOI')
    ax.set_xlabel('SRO [PPM]')
    ax.set_ylabel('Speech enhancement metrics')
    ax.legend()
    # plt.show()

    return fig


if __name__ == '__main__':
    sys.exit(main())