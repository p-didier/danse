# Purpose of script:
# Post-processing of battery test "battery20231013_perf_asfctofSelfNoise"
# plotting the MMSE as a function of iterations.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Created on 2023.10.19 - 11:30.

# Note on 2023.10.19 - 12:10:
# - Not very useful because the target speech is nonstationary, thus the 
#   MMSE does not evolve smoothly.

import sys
import gzip
import copy
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from danse_toolbox.d_classes import OutputsForPostProcessing

DATA_FOLDER = './danse/out/battery20231013_perf_asfctofSelfNoise_wReverb'

def main(dataFolder: str=DATA_FOLDER):
    """Main function (called by default when running script)."""
    
    data, params = get_data(dataFolder)

    # Plot
    fig, _ = plot_data(data, params)

    # Export
    fig.savefig(f'{dataFolder}/mmse.pdf', bbox_inches='tight')
    fig.savefig(f'{dataFolder}/mmse.png', dpi=300, bbox_inches='tight')

    print('Done.')

    return 0


def get_data(dataFolder: str) -> dict[dict[pd.DataFrame]]:
    """Load and organize data from DANSE battery test output folder."""

    # Load data
    data, params = load_data(dataFolder)
    
    return data, params


def load_data(dataFolder: str) -> dict:
    """Load data from DANSE battery test output folder."""
    # List all subfolders
    subfolders = [f for f in Path(dataFolder).iterdir() if f.is_dir()]
    # Loop over subfolders
    out = {}
    params = {}
    params['snsnr'] = []
    for ii, subfolder in enumerate(subfolders):
        if subfolder.name[0] == '_':  # skip folders starting with '_'
            continue
        print(f'Loading simulation data from {subfolder.name} ({ii+1}/{len(subfolders)})...')
        subfolderName = subfolder.name
        out[subfolderName] = {}  # init
        currSNSNR = int(subfolderName.split('_')[1][:-2])
        out[subfolderName]['snsnr'] = currSNSNR
        if currSNSNR not in params['snsnr']:
            params['snsnr'].append(currSNSNR)
        
        # Load DANSE outputs from pkl
        with open(f'{subfolder}/OutputsForPostProcessing.pkl.gz', 'rb') as f:
            outputs = pickle.load(gzip.open(f, 'r'))

        # Compute MMSE
        mmse, mmse_c, mmse_l = compute_mmse(outputs)
        
        out[subfolderName]['mmse'] = mmse
        out[subfolderName]['mmse_c'] = mmse_c
        out[subfolderName]['mmse_l'] = mmse_l
    params['snsnr'].sort()

    return out, params


def compute_mmse(out: OutputsForPostProcessing):
    # Get desired signal target in time-domain (TODO: STFT-domain)
    mmse = np.zeros((len(out.wasnObj.wasn[0].cleanspeech), len(out.wasnObj.wasn)))
    mmse_c = copy.deepcopy(mmse)
    mmse_l = copy.deepcopy(mmse)
    for k in range(len(out.wasnObj.wasn)):
        d = out.wasnObj.wasn[k].cleanspeech[:, out.wasnObj.wasn[k].refSensorIdx]
        wy = out.danseOutputs.TDdesiredSignals_est[:, k]
        wy_c = out.danseOutputs.TDdesiredSignals_est_c[:, k]  # centralized
        wy_l = out.danseOutputs.TDdesiredSignals_est_l[:, k]  # local
        # Compute MMSE
        mmse[:, k] = (d - wy)**2
        mmse_c[:, k] = (d - wy_c)**2
        mmse_l[:, k] = (d - wy_l)**2

    # Average over blocks
    blockDuration = .5  # [s]
    blockLength = int(blockDuration * out.wasnObj.wasn[0].fs)
    nBlocks = int(np.floor(len(out.wasnObj.wasn[0].cleanspeech) / blockLength))
    mmse = np.mean(mmse[:nBlocks * blockLength, :].reshape(nBlocks, blockLength, len(out.wasnObj.wasn)), axis=1)
    mmse_c = np.mean(mmse_c[:nBlocks * blockLength, :].reshape(nBlocks, blockLength, len(out.wasnObj.wasn)), axis=1)
    mmse_l = np.mean(mmse_l[:nBlocks * blockLength, :].reshape(nBlocks, blockLength, len(out.wasnObj.wasn)), axis=1)

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(len(out.wasnObj.wasn), 1)
    # fig.set_size_inches(8.5, 3.5)
    # for k in range(len(out.wasnObj.wasn)):
    #     axes[k].semilogy(mmse[:, k], label='MMSE DANSE')
    #     axes[k].semilogy(mmse_c[:, k], label='MMSE Centr.')
    #     # axes[k].semilogy(mmse_l[:, k], label='MMSE Local')
    #     axes[k].grid()
    #     axes[k].legend()
    # fig.tight_layout()	
    # plt.show()

    return mmse, mmse_c, mmse_l  



def plot_data(data: dict[dict], params):
    """Plot data.""",
    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    for ii, snsnr in enumerate(data.keys()):
        sns = data[snsnr]
        axes.plot(sns['mmse'][:, 0] - sns['mmse_c'][:, 0], label=f'SNSNR = {snsnr} dB')
    axes.grid()
    axes.legend()
    plt.tight_layout()

    return fig, axes


if __name__ == '__main__':
    sys.exit(main())