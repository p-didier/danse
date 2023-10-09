# Purpose of script:
# Post-processing of battery test "20231009_filtnorms_asfctofSROs".
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Created on 2023.10.09 - 12:02.

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

DATA_FOLDER = './danse/out/battery20231009_filtnorms_asfctofSROs'
METRICS_TO_PLOT = [
    'stoi',
    'sisnr',
]

def main(dataFolder: str=DATA_FOLDER):
    """Main function (called by default when running script)."""
    
    metricsData, filtNormsData = get_data(dataFolder)

    # Plot
    fig, _ = plot_data(metricsData, filtNormsData)

    # Export
    fig.savefig(f'{dataFolder}/combined_metrics.pdf', bbox_inches='tight')
    fig.savefig(f'{dataFolder}/combined_metrics.png', dpi=300, bbox_inches='tight')

    print('Done.')

    return 0


def get_data(dataFolder: str) -> dict[dict[pd.DataFrame]]:
    """Load and organize data from DANSE battery test output folder."""

    # Load data
    data, params = load_data(dataFolder)
    
    # Create one dataframe for each node, for each metric, for each type
    dfs = dict([
        (f'Node{k+1}', dict([
            (metric, dict([
                ('danse', None), ('centr', None)
            ])) for metric in METRICS_TO_PLOT
        ])) for k in range(params['nNodes'])
    ])
    for k in range(params['nNodes']):
        for metric in METRICS_TO_PLOT:
            df = pd.DataFrame(
                index=['danse', 'centr'],
                columns=params['sro']
            )
            for subfolder in data.keys():
                currData = data[subfolder]                
                df.loc['danse', currData['sro']] =\
                    getattr(currData['metrics'], metric)[f'Node{k+1}'].after
                df.loc['centr', currData['sro']] =\
                    getattr(currData['metrics'], metric)[f'Node{k+1}'].afterCentr
                
            dfs[f'Node{k+1}'][metric] = df

    return dfs, currData['filtNorms']


def load_data(dataFolder: str) -> dict:
    """Load data from DANSE battery test output folder."""
    # List all subfolders
    subfolders = [f for f in Path(dataFolder).iterdir() if f.is_dir()]
    # Loop over subfolders
    out = {}
    params = {}
    params['sro'] = []
    for ii, subfolder in enumerate(subfolders):
        if subfolder.name[0] == '_':  # skip folders starting with '_'
            continue
        print(f'Loading simulation data from {subfolder.name} ({ii+1}/{len(subfolders)})...')
        subfolderName = subfolder.name
        out[subfolderName] = {}  # init
        currSRO = int(subfolderName.split('_')[1])

        out[subfolderName]['sro'] = currSRO
        if currSRO not in params['sro']:
            params['sro'].append(currSRO)
        
        # Load metrics from pkl
        with open(f'{subfolder}/metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
        # Load filter norms from pkl
        with open(f'{subfolder}/filtNorms/filtNorms.pkl', 'rb') as f:
            filtNorms = pickle.load(f)
        
        if ii == 0:
            params['nNodes'] = len(getattr(metrics, METRICS_TO_PLOT[0]))
            
        out[subfolderName]['metrics'] = metrics
        out[subfolderName]['filtNorms'] = filtNorms
    params['sro'].sort()

    return out, params


def plot_data(metricsData: dict[dict[pd.DataFrame]], filtNormsData):
    """Plot data."""
    nodes = list(metricsData.keys())
    metrics = list(metricsData[nodes[0]].keys())
    sros = list(metricsData[nodes[0]][metrics[0]].columns)

    nCols = len(nodes)   # danse vs. centralized
    nRows = len(metrics)
    fig, axes = plt.subplots(nRows, nCols, sharex=True, sharey='row')
    fig.set_size_inches(5 * nRows, 3 * nCols)
    for ii, metric in enumerate(METRICS_TO_PLOT):
        for k, node in enumerate(nodes):  # for each node
            # General plot settings
            axes[ii, k].set_title(f'{metric.upper()} - {node}')
            if ii == nRows - 1:
                axes[ii, k].set_xlabel('SRO (ppm)')
            if k == 0:
                axes[ii, k].set_ylabel(metric.upper())
            if 'stoi' in metric:
                axes[ii, k].set_ylim([0, 1])
            axes[ii, k].set_xlim([0, np.amax(sros)])
            axes[ii, k].grid(True)
            axes[ii, k].plot(
                sros,
                metricsData[node][metric].loc['danse'],
                'C0-o',
                label='GEVD-DANSE',
            )
            axes[ii, k].plot(
                sros,
                metricsData[node][metric].loc['centr'],
                'C1--x',
                label='GEVD-MWF',
            )
            axes[ii, k].legend()
    plt.tight_layout()

    return fig, axes


if __name__ == '__main__':
    sys.exit(main())