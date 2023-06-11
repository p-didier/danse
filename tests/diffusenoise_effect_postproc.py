# Purpose of script:
# Automated tests for the effect of including diffuse noise in the DANSE
# simulations - post-processing script.
# >> Corresponding journal entry: 2023, week22, THU.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys, re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from danse_toolbox.d_post import DANSEoutputs

# Diffuse noise SNRs to consider
BASE_EXPORT_PATH = f'{Path(__file__).parent}/out/20230601_tests/diffusenoise_effect'

def main(
        baseExportPath: str=BASE_EXPORT_PATH,
        exportFigure: bool=True,
        forcedSNRylims: list=None
    ):
    """Main function (called by default when running script)."""
    # Load data
    snr, estoi, dnPowFacts = load_data(baseExportPath)

    fig = plot_data(snr, estoi, dnPowFacts, forcedSNRylims)
    if exportFigure:
        fig.savefig(f'{baseExportPath}/fig_diffusenoise_effect.png', dpi=300)
        fig.savefig(f'{baseExportPath}/fig_diffusenoise_effect.pdf')


def load_data(baseExportPath):
    # List all folders
    folders = [f for f in Path(baseExportPath).iterdir() if f.is_dir()]

    # Load data
    dnPowFacts = np.zeros(len(folders))
    for ii, folder in enumerate(folders):
        print(f'Loading data from {folder.name}...')
        out = DANSEoutputs().load(str(folder))
        if ii == 0:
            # Initialize data
            snrDataOut = np.zeros(
                (len(out.metrics.snr.keys()), len(folders), 4)
            )
            estoiDataOut = np.zeros(
                (len(out.metrics.stoi.keys()), len(folders), 4)
            )
        if isinstance(out.metrics.snr['Node1'].before, float):
            snrDataOut[:, ii, :] = np.array([[
                out.metrics.snr[key].before,        # raw
                out.metrics.snr[key].afterLocal,    # local
                out.metrics.snr[key].afterCentr,    # centralized
                out.metrics.snr[key].after,         # DANSE
            ] for key in out.metrics.snr.keys()])
        else:
            snrDataOut[:, ii, :] = np.array([[
                out.metrics.snr[key].before[0],        # raw
                out.metrics.snr[key].afterLocal[0],    # local
                out.metrics.snr[key].afterCentr[0],    # centralized
                out.metrics.snr[key].after[0],         # DANSE
            ] for key in out.metrics.snr.keys()])
        estoiDataOut[:, ii] = np.array([[
            out.metrics.stoi[key].before,        # raw
            out.metrics.stoi[key].afterLocal,    # local
            out.metrics.stoi[key].afterCentr,    # centralized
            out.metrics.stoi[key].after,         # DANSE
        ] for key in out.metrics.stoi.keys()])
        # Extract DN SNR from folder name
        dnPowFacts[ii] = re.findall(r'-?\d+', folder.name)[0]

    # Order by increasing DN SNR
    idx = np.argsort(dnPowFacts)
    dnPowFacts = dnPowFacts[idx]
    snrDataOut = snrDataOut[:, idx, :]
    estoiDataOut = estoiDataOut[:, idx, :]

    return snrDataOut, estoiDataOut, dnPowFacts


def plot_data(
        snr: np.ndarray,
        estoi: np.ndarray,
        dnPowFacts: np.ndarray,
        forcedSNRylims: list=None
    ):
    """
    Parameters:
    -----------
    snr: [nNodes x nSNRs x 4] np.ndarray (float)
        SNR data. Reference for last index: [raw, local, centralized, DANSE].
    estoi: [nNodes x nSNRs x 4] np.ndarray (float)
        eSTOI data. Reference for last index: [raw, local, centralized, DANSE].
    dnSNRs: [nSNRs x 1] np.ndarray (int)
        Diffuse noise power factors considered [dB].
    forcedSNRylims: [2 x 1] list (float) or None
        If not None, use these values to fix the SNR plots' y-axis limits.
        If None, determine from current data.
    """

    symbols = ['o', 's', 'd', '^']

    # Determine SNR ylims
    if forcedSNRylims is None:
        snrYlims = np.zeros((2, snr.shape[0], snr.shape[2]))
        for ii in range(snr.shape[0]):
            for jj in range(snr.shape[2]):
                snrYlims[0, ii, jj] = np.min(snr[ii, :, jj])
                snrYlims[1, ii, jj] = np.max(snr[ii, :, jj])
        snrYlims = np.array([np.min(snrYlims[0, :, :]), np.max(snrYlims[1, :, :])])
        snrYlims = np.array([np.min(snrYlims), np.max(snrYlims)])
        # Add some margin
        snrYlims[0] -= 1
        snrYlims[1] += 1
    else:
        snrYlims = forcedSNRylims

    fig, axes = plt.subplots(2, snr.shape[0], sharex=True)
    fig.set_size_inches(10.5, 6.5)
    for ii in range(snr.shape[0]):
        # Plot SNR
        for jj in range(snr.shape[2]):
            axes[0, ii].plot(dnPowFacts, snr[ii, :, jj], f'{symbols[jj]}-', markersize=4)
        axes[0, ii].grid()
        axes[0, ii].set_title(f'Node {ii+1}')
        if ii == 0:
            axes[0, ii].set_ylabel('Output SNR [dB]')
        axes[0, ii].set_ylim(snrYlims)
        # Plot eSTOI
        for jj in range(snr.shape[2]):
            axes[1, ii].plot(dnPowFacts, estoi[ii, :, jj], f'{symbols[jj]}-', markersize=4)
        axes[1, ii].grid()
        axes[1, ii].set_title(f'Node {ii+1}')
        axes[1, ii].set_xlabel('Diffuse noise (DN) pow. fact. [dB]')
        if ii == 0:
            axes[1, ii].set_ylabel('Output eSTOI')
        axes[1, ii].set_ylim([0, 1])
        if ii == snr.shape[0] - 1:
            axes[1, ii].legend(['raw', 'local', 'centralized', 'DANSE'])
    fig.tight_layout()

    return fig


if __name__ == '__main__':
    sys.exit(main())