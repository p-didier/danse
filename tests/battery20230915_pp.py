# Purpose of script:
# Post-processing of battery test "20230915_sros_asfctofL".
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Created on 2023.09.15 - 15:08.

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from danse_toolbox.d_post import DANSEoutputs

DATA_FOLDER = './danse/out/battery20230915_sros_asfctofL'
METRICS_TO_PLOT = [
    'stoi',
    'sisnr',
]
NS = 512  # TODO: improve handling of this -- could be inferred from results

# TO_PLOT = ['danse']
TO_PLOT = ['centr']
# TO_PLOT = [
#     'danse',
#     'centr'
# ]

FORCED_YLIMITS_SNR = None
FORCED_YLIMITS_SNR = [0, 17.5]

def main(dataFolder: str=DATA_FOLDER):
    """Main function (called by default when running script)."""
    
    data = get_data(dataFolder, Ns=NS)

    # Plot
    fig, _ = plot_data(data)

    # Export
    fig.savefig(f'{dataFolder}/combined_metrics_{"_".join(TO_PLOT)}.pdf', bbox_inches='tight')
    fig.savefig(f'{dataFolder}/combined_metrics_{"_".join(TO_PLOT)}.png', dpi=300, bbox_inches='tight')

    print('Done.')

    return 0


def plot_data(data: dict[dict[pd.DataFrame]]):
    """Plot data."""
    lineStyles = ['-', '--', '-.', ':']

    nodes = list(data.keys())
    metrics = list(data[nodes[0]].keys())  # `== METRICS_TO_PLOT`

    nRows = len(metrics)
    nCols = len(nodes)
    fig, axes = plt.subplots(nRows, nCols, sharex=True, sharey='row')
    fig.set_size_inches(5 * nRows, 3 * nCols)
    for ii in range(nRows):
        for jj in range(nCols):
            # Axes object selection
            if nRows == 1 and nCols == 1:
                currAx = axes
            elif nRows == 1:
                currAx = axes[jj]
            elif nCols == 1:
                currAx = axes[ii]
            else:
                currAx = axes[ii, jj]
            
            # Plot
            k = nodes[jj]
            m = metrics[ii]
            currData = data[k][m]
            for idxTyp, typ in enumerate(list(currData.keys())):
                sros = currData[typ].index
                bcLengths = currData[typ].columns
                # Cycle through line styles
                currLineStyle = lineStyles[idxTyp % len(lineStyles)]
                for idxL, l in enumerate(bcLengths):
                    if np.isnan(currData[typ][l][0]):
                        currData[typ][l][0] = currData[typ][1][0]  # complete redundant data 
                    currAx.plot(
                        sros,
                        currData[typ][l],
                        f'C{idxL}.{currLineStyle}',
                        label=f'$L = {l}$ ({typ})' if len(list(currData.keys())) > 1 else f'$L = {l}$'
                    )
            currAx.grid()
            if ii == jj == 0:
                currAx.legend(loc='lower left')
            currAx.set_xlabel('SRO $\\varepsilon_{{kq}}$ [PPM]')
            if 'snr' in m:
                currAx.set_ylabel('[dB]')
                if FORCED_YLIMITS_SNR is not None:
                    currAx.set_ylim(FORCED_YLIMITS_SNR)
                else:
                    currAx.set_ylim([
                        np.amin([0, np.amin(currAx.get_ylim())]),
                        np.amax(currAx.get_ylim())
                    ])
            currAx.set_title(f'{k}, {m}')
            if 'stoi' in m:
                currAx.set_ylim([0, 1])

    fig.suptitle(' and '.join(TO_PLOT))
    plt.tight_layout()
    return fig, axes



def get_data(
        dataFolder: str,
        Ns=512
    ) -> dict[dict[pd.DataFrame]]:
    """Load and organize data from DANSE battery test output folder."""
    data, params = load_data(dataFolder, Ns)
    
    # Create one dataframe for each node
    dfs = dict([
        (
            f'Node{k+1}',
            dict([
                (metric, dict([
                    (typ, None) for typ in TO_PLOT
                ])) for metric in METRICS_TO_PLOT
            ])
        ) for k in range(params['nNodes'])
    ])
    for k in range(params['nNodes']):
        for metric in METRICS_TO_PLOT:
            for typ in TO_PLOT:
                df = pd.DataFrame(
                    index=params['SRO'],
                    columns=params['L']
                )
                for subfolder in data.keys():
                    currData = data[subfolder]
                    if typ == 'danse':
                        attr = getattr(currData['metrics'], metric)[f'Node{k+1}'].after
                    elif typ == 'centr':
                        attr = getattr(currData['metrics'], metric)[f'Node{k+1}'].afterCentr
                    df.loc[currData['SRO'], currData['L']] = attr
                    
                dfs[f'Node{k+1}'][metric][typ] = df

    return dfs


def load_data(dataFolder: str, Ns=512) -> dict:
    """Load data from DANSE battery test output folder."""
    # List all subfolders
    subfolders = [f for f in Path(dataFolder).iterdir() if f.is_dir()]
    # Loop over subfolders
    out = {}
    params = {}
    params['L'] = []
    params['SRO'] = []
    for ii, subfolder in enumerate(subfolders):
        if subfolder.name[0] == '_':  # skip folders starting with '_'
            continue
        print(f'Loading simulation data from {subfolder.name} ({ii+1}/{len(subfolders)})...')
        subfolderName = subfolder.name
        out[subfolderName] = {}  # init
        if 'wholeChunk' in subfolderName:
            currL = Ns
        else:
            currL = int(subfolderName.split('_')[0][1:])
        currSRO = int(subfolderName.split('_')[1][3:])

        out[subfolderName]['L'] = currL
        if currL not in params['L']:
            params['L'].append(currL)
        out[subfolderName]['SRO'] = currSRO
        if currSRO not in params['SRO']:
            params['SRO'].append(currSRO)
        
        if 'metrics.pkl' in [f.name for f in subfolder.iterdir()]:
            # Load metrics from pkl
            with open(f'{subfolder}/metrics.pkl', 'rb') as f:
                metrics = pickle.load(f)
        else:
            metrics = DANSEoutputs().load(subfolder).metrics
            # Save as pkl for faster loading next time
            with open(f'{subfolder}/metrics.pkl', 'wb') as f:
                pickle.dump(metrics, f)
        if ii == 0:
            params['nNodes'] = len(getattr(metrics, METRICS_TO_PLOT[0]))
        out[subfolderName]['metrics'] = metrics
    params['L'].sort()
    params['SRO'].sort()

    return out, params

if __name__ == '__main__':
    sys.exit(main())