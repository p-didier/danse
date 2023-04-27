# Purpose of script:
#  - Test the inherent robustness of the DANSE algorithm to SROs.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import yaml
import numpy as np
import pickle, gzip
from pathlib import Path
from danse_toolbox.d_classes import *
from .sandbox import main as main_sandbox
from danse_toolbox.d_post import DANSEoutputs
from danse_toolbox.d_base import DANSEparameters, CohDriftParameters, PrintoutsAndPlotting

# PATH_TO_CONFIG_FILE = f'{Path(__file__).parent.parent}/config_files/sros_effect_20230406.yaml'
PATH_TO_CONFIG_FILE = f'{Path(__file__).parent.parent}/config_files/sros_effect.yaml'
N_SRO_TESTS = 10    # number of SRO tests to run
MAX_SRO_PPM = 500   # maximum SRO in PPM
EXPORT_DATA = True
OUT_FOLDER = '20230426_tests/sros_effect/run1'  # export path relative to `danse/out`
# OUT_FOLDER = '20230414_tests/sros_effect/test_new2'  # export path relative to `danse/out`
SKIP_ALREADY_RUN = True  # if True, skip tests that have already been run
SIGNALS_PATH = f'{Path(__file__).parent.parent}/tests/sigs'

def main():
    """Main function (called by default when running script)."""

    # SROs to consider
    srosToConsider = [np.array([ii / 2, ii]) * MAX_SRO_PPM / N_SRO_TESTS for\
            ii in np.arange(0, N_SRO_TESTS)]
    
    # Run tests
    res = run_test_batch(
        configFilePath=PATH_TO_CONFIG_FILE,
        srosToConsider=srosToConsider,
    )

    if EXPORT_DATA:
        basePath = f'{Path(__file__).parent.parent}/out/{OUT_FOLDER}'
        # Export res object
        with open(f'{basePath}/metricsAsFctOfSROs.pkl', 'wb') as f:
            pickle.dump(res, f)
        # Export srosToConsider object
        with open(f'{basePath}/srosConsidered.pkl', 'wb') as f:
            pickle.dump(srosToConsider, f)


def setup_config(filePath: str):
    """
    Reads the YAML configuration file and sets up the `cfg` object.
    
    Parameters
    ----------
    filePath : str
        Path to YAML configuration file.
    
    Returns
    ----------
    testParams : TestParameters
        Test parameters object.
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """
    # Check number of lines in config file
    with open(filePath, 'r') as f:
        nLines = len(f.readlines())

    if nLines > 20:  # -- Cfg file type after 14.04.2023. Use newer method.
        testParams = TestParameters().load_from_yaml(filePath)
        
    else:  # -- Cfg file type prior to 14.04.2023. Use old method.

        with open(filePath, "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        
        # Create test parameters
        testParams = TestParameters(
            exportParams=ExportParameters(
                bypassAllExports=True  # bypass all sounds and figures exports
            ),
            seed=cfg['seed'],
            wasnParams=WASNparameters(
                layoutType=cfg['layoutType'],
                VADenergyDecrease_dB=35,  # [dB]
                topologyParams=TopologyParameters(
                    topologyType=cfg['topologyType'],
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
                t60=cfg['t60'],  # ====<<<<<
                interSensorDist=0.2,
                nNodes=3,
                nSensorPerNode=cfg['nSensorPerNode'],  # ====<<<<<
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
                SROperNode=[],  # <-- will be set in `run_test_batch`
            ),
            danseParams=DANSEparameters(
                DFTsize=1024,
                WOLAovlp=.5,
                nodeUpdating=cfg['nodeUpdating'],  # ====<<<<<
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
                performGEVD=cfg['performGEVD'],  # ====<<<<<
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


def run_test_batch(
        configFilePath: str,
        srosToConsider: list
    ) -> list[dict]:
    """
    Runs a test batch based on a (YAML) configuration.
    
    Parameters
    ----------
    configFilePath : str
        Path to YAML configuration file.
    srosToConsider : list[list[float]]
        List of lists of SROs to consider (per node) [PPM].
    
    Returns
    ----------
    allVals : list[dict]
        List of dictionaries containing the speech enhancement metrics results.
    
    (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
    """
    # Set up configuration
    testParams = setup_config(configFilePath)
    
    allVals = []
    for ii, sros in enumerate(srosToConsider):
        # Set up SRO parameter
        currSROs = np.concatenate((np.array([0.]), sros))
        pickleFilePath = f'{Path(__file__).parent.parent}/out/{OUT_FOLDER}/backupvals/vals_sros_{ii+1}.pkl.gz'
        if SKIP_ALREADY_RUN and Path(pickleFilePath).exists():
            print(f'>>> Skipping SRO test {ii+1} / {len(srosToConsider)}: {currSROs} PPM\n')
            # Load results
            vals = pickle.load(gzip.open(pickleFilePath, 'r'))
        else:
            print(f'\n\n>>> Running SRO test {ii+1} / {len(srosToConsider)}: {currSROs} PPM\n')
            # Set up test parameters
            testParams.wasnParams.SROperNode = currSROs
            testParams.exportParams.exportFolder =\
                f'{Path(__file__).parent.parent}/out/{OUT_FOLDER}/sros_{ii+1}'
            testParams.__post_init__()
            testParams.danseParams.get_wasn_info(testParams.wasnParams)
            # Run test
            res = main_sandbox(p=testParams)
            # Extract single test results
            vals = extract_single_test_results(res)
            # Save results
            if not Path(pickleFilePath).parent.exists():
                Path(pickleFilePath).parent.mkdir(parents=True)
            pickle.dump(vals, gzip.open(pickleFilePath, 'wb'))
        allVals.append(vals)
        plt.close(fig='all')  # close all figures (avoid memory overload)

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

    def _process_values(values):
        """Processes the values from a Metrics object, transforming
        them into a dict (for easy export as CSV through Pandas)."""
        dictList = []
        for ii, val in enumerate(values):
            subDict = dict([
                ('raw', val.before),
                ('local', val.afterLocal),
                ('danse', val.after),
                ('centr', val.afterCentr),
            ])
            dictList.append((f'Node{ii + 1}', subDict))
        return dict(dictList)
    
    vals = dict([
        ('snr', _process_values(res.metrics.snr.values())),
        ('estoi', _process_values(res.metrics.stoi.values())),
        ('sros', res.SROgroundTruth)
    ])
    return vals