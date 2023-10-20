# Purpose of script:
# Post-processing of battery test "battery20231013_perf_asfctofSelfNoise".
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Created on 2023.10.16 - 10:15.

import sys
import pickle
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

DATA_FOLDER = './danse/out/battery20231013_perf_asfctofSelfNoise_wReverb'
METRICS_TO_PLOT = [
    # 'stoi',
    'sisnr',
]

# METRICS_COMPUTATION_SEGMENT_DUR = 5  # s
METRICS_COMPUTATION_SEGMENT_DUR = None  # s

def main(dataFolder: str=DATA_FOLDER):
    """Main function (called by default when running script)."""
    
    data = get_data(dataFolder)

    # Plot
    fig, _ = plot_data(data)

    # Export
    if METRICS_COMPUTATION_SEGMENT_DUR is None:
        suffix = 'full'
    else:
        suffix = f'dur{METRICS_COMPUTATION_SEGMENT_DUR}s'
    fig.savefig(f'{dataFolder}/combined_metrics_{suffix}.pdf', bbox_inches='tight')
    fig.savefig(f'{dataFolder}/combined_metrics_{suffix}.png', dpi=300, bbox_inches='tight')

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
                (f'snsnr_{snsnr}dB', None) for snsnr in params['snsnr']
            ])) for metric in METRICS_TO_PLOT
        ])) for k in range(params['nNodes'])
    ])
    for k in range(params['nNodes']):
        for metric in METRICS_TO_PLOT:    
            for snsnr in params['snsnr']:
                currData = data[f'snsnr_{snsnr}dB']

                df = pd.DataFrame(
                    index=['danse', 'centr', 'local'],
                    columns=params['startMetricComp']
                )
                for t in params['startMetricComp']:
                    df.loc['danse', t] =\
                        getattr(currData['metrics'][f'start_{t}s'], metric)[f'Node{k+1}'].after
                    df.loc['centr', t] =\
                        getattr(currData['metrics'][f'start_{t}s'], metric)[f'Node{k+1}'].afterCentr
                    df.loc['local', t] =\
                        getattr(currData['metrics'][f'start_{t}s'], metric)[f'Node{k+1}'].afterLocal
                
                dfs[f'Node{k+1}'][metric][f'snsnr_{snsnr}dB'] = df

    return dfs


def load_data(dataFolder: str) -> dict:
    """Load data from DANSE battery test output folder."""
    # List all subfolders
    subfolders = [f for f in Path(dataFolder).iterdir() if f.is_dir()]
    # Loop over subfolders
    out = {}
    params = {}
    params['snsnr'] = []
    params['startMetricComp'] = []
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
        
        # Define "further post-processing" subfolder
        if METRICS_COMPUTATION_SEGMENT_DUR is not None:
            furtherPPpath = f'{subfolder}/further_pp/dur{METRICS_COMPUTATION_SEGMENT_DUR}s'
        else:
            furtherPPpath = f'{subfolder}/further_pp/full'
        # List all subfolders of "further post-processing" subfolder
        subss = [f for f in Path(furtherPPpath).iterdir() if f.is_dir()]
        out[subfolderName]['metrics'] = {}  # init
        for jj, sub in enumerate(subss):
            # Load metrics from pkl
            with open(f'{sub}/metrics.pkl', 'rb') as f:
                metrics = pickle.load(f)
            
            if ii == 0:
                if jj == 0:
                    params['nNodes'] = len(getattr(metrics, METRICS_TO_PLOT[0]))
                params['startMetricComp'].append(int(sub.name.split('_')[1][:-1]))
            
            out[subfolderName]['metrics'][sub.name] = metrics
    params['snsnr'].sort()
    params['startMetricComp'].sort()

    return out, params


def plot_data(data: dict[dict[pd.DataFrame]]):
    """Plot data."""
    nodes = list(data.keys())
    metrics = list(data[nodes[0]].keys())
    snSNRs = list(data[nodes[0]][metrics[0]].keys())
    metricStartInstants = list(data[nodes[0]][metrics[0]][snSNRs[0]].columns)

    cmap = matplotlib.cm.get_cmap('Spectral')
    snSNRtoInclude = [0, 5, 10, 20, 30, 40, 50]

    nCols = len(nodes)   # danse vs. centralized
    nRows = len(metrics)
    fig, axes = plt.subplots(nRows, nCols, sharex=True, sharey='row')
    fig.set_size_inches(5 * nCols, 4 * nRows)
    for ii, metric in enumerate(METRICS_TO_PLOT):
        for k, node in enumerate(nodes):  # for each node
            # Select subplot
            if nRows == 1:
                currAx = axes[k]
            elif nCols == 1:
                currAx = axes[ii]
            else:
                currAx = axes[ii, k]
            # General plot settings
            currAx.set_title(f'{metric.upper()} - {node}')
            if ii == nRows - 1:
                currAx.set_xlabel('Metrics computation segment start time [s]')
            if k == 0:
                currAx.set_ylabel(metric.upper())
            if 'stoi' in metric:
                currAx.set_ylim([0, 1])
            currAx.set_xlim([0, np.amax(metricStartInstants)])
            currAx.grid(True)
            for jj, snSNR in enumerate(snSNRs):
                if int(list(snSNR.split('_'))[1][:-2]) not in snSNRtoInclude:
                    continue
                currAx.plot(
                    metricStartInstants,
                    data[node][metric][snSNR].loc['danse'],
                    '-o',
                    color=cmap(jj / len(snSNRs)),
                    label=snSNR,
                )
                currAx.plot(
                    metricStartInstants,
                    data[node][metric][snSNR].loc['centr'],
                    '-x',
                    color=cmap(jj / len(snSNRs)),
                    label=f'{snSNR} (c)',
                )
            if ii == 0 and k == len(nodes) - 1:
                # Create legend on right side of figure
                handles, labels = currAx.get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc='center right',
                    bbox_to_anchor=(0.98, 0.5),
                    title='Self-noise SNR [dB]',
                )
    plt.tight_layout()
    # Make sure the legend is not cut off
    plt.subplots_adjust(right=0.8)


    return fig, axes


if __name__ == '__main__':
    sys.exit(main())