# ~created by P. Didier on 31.03.2023
# Purpose of script:
#
# Create simpler (less overwhelming) figures for presentation to G. Enzner,
# based on simulation results shown to SOUNDS SC on 29.03.2023
# --> only show SNR and eSTOI.

import sys
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append('C:/Users/pdidier/Dropbox/PC/Documents/sounds-phd/danse')
from danse_toolbox.d_post import DANSEoutputs, metrics_subplot

FOLDERS_TO_CONSIDER = [f'{Path(__file__).parent}/scenario{ii + 1}' for ii in range(4)]
EXPORT = True
FIGSIZE = (8,3)

def main():
    for ii, folder in enumerate(FOLDERS_TO_CONSIDER):
        print(f'Processing folder {ii + 1}/{len(FOLDERS_TO_CONSIDER)}...')
        fig = process_folder(folder)
        if EXPORT:
            fig.savefig(f'{Path(__file__).parent}/out_sc{ii + 1}.png')
            fig.savefig(f'{Path(__file__).parent}/out_sc{ii + 1}.pdf')
    print('ALL DONE.')


def process_folder(folder):
    # Load `DANSEoutputs` object
    outputs = DANSEoutputs()
    outputs = outputs.load(folder)
    # Plot
    fig = plot_metrics_simpler(outputs)
    return fig


def plot_metrics_simpler(out: DANSEoutputs):
    """
    Visualize evaluation metrics.

    Parameters
    ----------
    out : `DANSEoutputs` object
        DANSE outputs.
    """

    # Useful variables
    barWidth = 1
    
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1, 2, 1)   # Unweighted SNR
    metrics_subplot(out.nNodes, ax, barWidth, out.metrics.snr)
    ax.set(title='SNR', ylabel='[dB]')
    #
    ax = fig.add_subplot(1, 2, 2)   # eSTOI
    metrics_subplot(out.nNodes, ax, barWidth, out.metrics.stoi)
    ax.set(title='eSTOI', ylabel='[dB]')
    ax.legend(bbox_to_anchor=(1, 0), loc="lower left")
    ax.set_ylim([0, 1])

    plt.tight_layout()

    return fig

if __name__ == '__main__':
    sys.exit(main())