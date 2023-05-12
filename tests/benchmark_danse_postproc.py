# Purpose of script:
# Post-processing script for benchmark tests for the TI-GEVD-DANSE
#v online implementation.
# >> Journal entry reference: 2023, week19, TUE. 
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import re
import sys
import matplotlib.pyplot as plt
from pathlib import Path, WindowsPath
from danse_toolbox.d_post import DANSEoutputs, metrics_subplot

REPO_ROOT_PATH = f'{Path(__file__).parent.parent}'  # Path to the root of the repository
RESULTS_FOLDER = f'{REPO_ROOT_PATH}/out/20230508_tests/tigevddanse'  # Path to the output folder

METRIC_TO_PLOT = 'snr'  # Metric to plot
REVERB_SETTING = {
    'anechoic': 0,
    'rt200ms': 200
}  # Reverb settings to plot

FORCED_YLIMITS = [0, 35]  # Force y-axis limits for all plots

# Format: 1st column: single-sensor nodes, 2nd column: multi-sensor nodes
FIGURES_FORMAT = [
    ['ss_noSROs', 'ms_noSROs'],             # 1st row: no SROs
    ['ss_mediumSROs', 'ms_mediumSROs'],     # 2nd row: medium SROs
    ['ss_largeSROs', 'ms_largeSROs'],       # 3rd row: large SROs
]

def main(
        resultsFolder=RESULTS_FOLDER,
        suptitlePrefix='TI-GEVD-DANSE',
    ):
    """
    Main function (called by default when running script).
    
    Parameters
    ----------
    resultsFolder : str
        Path to the folder containing the results.
    suptitlePrefix : str
        Prefix for the suptitle of the figure.
    """
    
    # List subfolders
    subdirs = [f for f in Path(resultsFolder).iterdir() if\
        f.is_dir() and re.search('sro', f.name, re.IGNORECASE)]
    # ^ re.search command https://stackoverflow.com/a/6579933

    # Cluster by tau using the folder name
    tauLabs = []
    for f in subdirs:
        tauLabs.append(re.findall(r'\d+', f.name)[0])
    tauLabs = list(set(tauLabs))

    # Loop over tau
    for idxTauLab, currTauLab in enumerate(tauLabs):
        # Inform user
        print(
            f'Processing tau={currTauLab} s ({idxTauLab + 1}/{len(tauLabs)})...'
        )
        folders = [
            f for f in subdirs if re.findall(r'\d+', f.name)[0] == currTauLab
        ]
        figs = plot_for_curr_tau(folders, suptitlePrefix)

        # Save figure
        if not Path(f'{resultsFolder}/pp').exists():
            Path(f'{resultsFolder}/pp').mkdir()
        for ii, fig in enumerate(figs):
            fig.savefig(
                f'{resultsFolder}/pp/tau_{currTauLab}s_{METRIC_TO_PLOT}_{list(REVERB_SETTING.keys())[ii]}.png',
                dpi=300
            )
            plt.close(fig)


def plot_for_curr_tau(folders: list[WindowsPath], suptitlePrefix=''):
    """
    Plot for the given folders.
    """
    
    figs = []
    for revTime in REVERB_SETTING.values():
    
        dataToPlot = select_data_to_plot(
            folders,
            format=FIGURES_FORMAT,
            revTime=revTime
        )

        nRows = len(dataToPlot)
        nCols = len(dataToPlot[0])

        tau = re.findall(r"\d+", folders[0].name)[0]

        fig, axes = plt.subplots(nRows, nCols, sharex=True, sharey=True)
        fig.set_size_inches(8, 8)
        fig.suptitle(
            f'{suptitlePrefix} (tau = {tau} s) -- Metric: {METRIC_TO_PLOT}'
        )
        for idxRow, currRow in enumerate(dataToPlot):
            for idxCol, currFolder in enumerate(currRow):
                # Load data
                data = load_data(currFolder)
                # Plot data
                metrics_subplot(ax=axes[idxRow, idxCol], data=data)
                axes[idxRow, idxCol].set_title(currFolder.name)
                axes[idxRow, idxCol].legend(
                    bbox_to_anchor=(1, 0),
                    loc="lower left"
                )
                axes[idxRow, idxCol].set_ylim(FORCED_YLIMITS)
        fig.tight_layout()
        figs.append(fig)
    
    return figs


def load_data(folder: WindowsPath):
    """Load data from the given folder."""
    dataCurr = DANSEoutputs().load(folder)
    return getattr(dataCurr.metrics, METRIC_TO_PLOT)


def select_data_to_plot(
        folders: list[WindowsPath],
        format: list[list[str]],
        revTime: int = 200  # [ms]
    ) -> list[list[WindowsPath]]:
    """
    Select data to plot from the given folders, following the given format.
    """
    
    dataToPlot = []
    for currRowFormats in format:
        dataToPlot.append([])
        for currColFormat in currRowFormats:
            keys = currColFormat.split('_')
            revTimeKey = f'rt{revTime}ms'
            keys.append(revTimeKey)  # Add reverb time key
            for currFolder in folders:
                # Append `currFolder` if it matches the `currColFormat`
                if all([key in currFolder.name for key in keys]):
                    dataToPlot[-1].append(currFolder)
                    break
    return dataToPlot

if __name__ == '__main__':
    sys.exit(main())