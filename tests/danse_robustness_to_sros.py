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

PATH_TO_CONFIG_FILE = f'{Path(__file__).parent.parent}/config_files/sros_effect_20230406.yaml'
SIGNALS_PATH = f'{Path(__file__).parent.parent}/tests/sigs'
N_SRO_TESTS = 10
MAX_SRO_PPM = 500
EXPORT_FIGURES = True
OUT_FOLDER = '20230406_tests/sros_effect'  # export path relative to `danse/out`
SKIP_ALREADY_RUN = True  # skip tests that have already been run


def main():
    """Main function (called by default when running script)."""
    cfg = read_config(filePath=PATH_TO_CONFIG_FILE)

    # Get SROs
    cfg['sros'] = [np.array([ii / 2, ii]) * MAX_SRO_PPM / N_SRO_TESTS for\
        ii in np.arange(0, N_SRO_TESTS)]

    # Run tests
    res = run_test_batch(cfg)

    # Post-process results
    fig = post_process_results(res)
    if EXPORT_FIGURES:
        fig.savefig(
            f'{Path(__file__).parent.parent}/out/{OUT_FOLDER}/sros_effect.png',
            dpi=300
        )
        fig.savefig(
            f'{Path(__file__).parent.parent}/out/{OUT_FOLDER}/sros_effect.pdf'
        )
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
        currSROs = np.concatenate((np.array([0.]), sros))
        pickleFilePath = f'{Path(__file__).parent.parent}/out/{OUT_FOLDER}/backupvals/vals_{ii+1}.pkl.gz'
        if SKIP_ALREADY_RUN and Path(pickleFilePath).exists():
            print(f'>>> Skipping SRO test {ii+1} / {len(cfg["sros"])}: {currSROs} PPM\n')
            # Load results
            vals = pickle.load(gzip.open(pickleFilePath, 'r'))
        else:
            print(f'\n\n>>> Running SRO test {ii+1} / {len(cfg["sros"])}: {currSROs} PPM\n')
            # Set up test parameters
            testParams = setup_test_parameters(cfg, currSROs)
            # Run test
            res = main_sandbox(p=testParams, bypassPostprocess=True)
            # Extract single test results
            vals = extract_single_test_results(res)
            # Save results
            if not Path(pickleFilePath).parent.exists():
                Path(pickleFilePath).parent.mkdir(parents=True)
            pickle.dump(vals, gzip.open(pickleFilePath, 'wb'))
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
        bypassExport=True,  # <-- BYPASSING FIGURES AND SOUNDS EXPORT
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

    # Get useful variables
    nNodes = len(res[0]['snr'])
    # Extract local and raw results (same for all SROs)
    localResSNR = np.zeros(nNodes)
    localResSTOI = np.zeros(nNodes)
    rawResSNR = np.zeros(nNodes)
    rawResSTOI = np.zeros(nNodes)
    for k in range(nNodes):
        localResSNR[k] = res[0]['snr'][f'Node{k+1}']['local']
        localResSTOI[k] = res[0]['estoi'][f'Node{k+1}']['local']
        rawResSNR[k] = res[0]['snr'][f'Node{k+1}']['raw']
        rawResSTOI[k] = res[0]['estoi'][f'Node{k+1}']['raw']

    # Build arrays for DANSE and centralized results
    danseResSNR = np.zeros((len(res), nNodes))
    danseResSTOI = np.zeros((len(res), nNodes))
    centralResSNR = np.zeros((len(res), nNodes))
    centralResSTOI = np.zeros((len(res), nNodes))
    for ii in range(len(res)):
        for k in range(nNodes):
            danseResSNR[ii, k] = res[ii]['snr'][f'Node{k+1}']['danse']
            danseResSTOI[ii, k] = res[ii]['estoi'][f'Node{k+1}']['danse']
            centralResSNR[ii, k] = res[ii]['snr'][f'Node{k+1}']['centr']
            centralResSTOI[ii, k] = res[ii]['estoi'][f'Node{k+1}']['centr']
    
    # Plot
    for k in range(nNodes):
        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(6.5, 2)
        axes[0].plot(danseResSNR[:, k], color='C1', marker='o', label='DANSE')
        axes[0].plot(centralResSNR[:, k], color='C2', marker='s', label='Centralized')
        axes[0].hlines(localResSNR[k], 0, len(res) - 1, color='C3', linestyles='dashed', label='Local')
        axes[0].hlines(rawResSNR[k], 0, len(res) - 1, color='C0', linestyles='dashdot', label='Raw')
        axes[0].set_xlabel('SROs [PPM]')
        axes[0].set_ylabel('SNR [dB]')
        axes[0].set_xticks(np.arange(len(res)))
        axes[0].set_xticklabels(
            [str(res[ii]['sros'][1:]) for ii in range(len(res))],
            rotation=30
        )
        axes[0].legend()
        axes[0].grid()
        # plt.show()
        axes[1].plot(danseResSTOI[:, k], color='C1', marker='o', label='DANSE')
        axes[1].plot(centralResSTOI[:, k], color='C2', marker='s', label='Centralized')
        axes[1].hlines(localResSTOI[k], 0, len(res) - 1, color='C3', linestyles='dashed', label='Local')
        axes[1].hlines(rawResSTOI[k], 0, len(res) - 1, color='C0', linestyles='dashdot', label='Raw')
        axes[1].set_xlabel('SROs [PPM]')
        axes[1].set_ylabel('eSTOI')
        axes[1].set_xticks(np.arange(len(res)))
        axes[1].set_xticklabels(
            [str(res[ii]['sros'][1:]) for ii in range(len(res))],
            rotation=30
        )
        axes[1].legend()
        axes[1].grid()
        axes[1].set_ylim([0, 1])  # eSTOI limits
        fig.tight_layout()

    return fig


if __name__ == '__main__':
    sys.exit(main())