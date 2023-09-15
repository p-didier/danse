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

def main(dataFolder: str=DATA_FOLDER):
    """Main function (called by default when running script)."""
    
    data = get_data(dataFolder, Ns=NS)

    # Plot
    plot_data(data)


def plot_data(data: dict[dict[pd.DataFrame]]):

    nodes = list(data.keys())
    metrics = list(data[nodes[0]].keys())

    nRows = len(metrics)
    nCols = len(nodes)
    fig, axes = plt.subplots(nRows, nCols, sharex=True, sharey='row')
    fig.set_size_inches(3 * nRows, 2 * nCols)
    for ii in range(nRows):
        for jj in range(nCols):
            # if nRows == 1:   # TODO: special cases
            #     if nCols == 1:
            #         currAx = axes
            #     else:
            #         currAx = axes[jj]
            currAx = axes[ii, jj]
            k = nodes[jj]
            m = metrics[ii]
            currData = data[k][m]
            sros = currData.index
            bcLengths = currData.columns
            for idxL, l in enumerate(bcLengths):
                if np.isnan(currData[l][0]):
                    currData[l][0] = currData[1][0]  # complete redundant data 
                currAx.plot(
                    sros,
                    currData[l],
                    f'C{idxL}.-',
                    label=f'$L = {l}$'
                )
            currAx.grid()
            if ii == jj == 0:
                currAx.legend(loc='best')
            currAx.set_xlabel('SRO $\\varepsilon_{{kq}}$ [PPM]')
            if 'snr' in m:
                currAx.set_ylabel('[dB]')
                currAx.set_ylim([np.amin([0, np.amin(currAx.get_ylim())]), np.amax(currAx.get_ylim())])
            currAx.set_title(f'{k}, {m}')
            if 'stoi' in m:
                currAx.set_ylim([0, 1])

    plt.show()  



def get_data(dataFolder: str, Ns=512) -> dict[dict[pd.DataFrame]]:
    data, params = load_data(dataFolder, Ns)
    
    # Create one dataframe for each node
    dfs = dict([
        (f'Node{k+1}', dict([
            (metric, None) for metric in METRICS_TO_PLOT
        ])) for k in range(params['nNodes'])
    ])
    for k in range(params['nNodes']):
        for metric in METRICS_TO_PLOT:
            df = pd.DataFrame(
                index=params['SRO'],
                columns=params['L']
            )
            for subfolder in data.keys():
                currData = data[subfolder]
                df.loc[currData['SRO'], currData['L']] =\
                    getattr(currData['metrics'], metric)[f'Node{k+1}'].after
                
            dfs[f'Node{k+1}'][metric] = df

    return dfs


def load_data(dataFolder: str, Ns=512) -> dict:
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