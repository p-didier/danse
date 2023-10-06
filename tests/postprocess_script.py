# Purpose of script:
# Performs further post-processing of the results of a given test, without
# having to re-run the test.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from danse_toolbox.d_classes import OutputsForPostProcessing, ExportParameters
from .sandbox import postprocess

def main(foldername='', params=dict()):
    """Main function (called by default when running script)."""
    
    a = OutputsForPostProcessing().load(foldername)

    # Change export parameters
    newExportParams = ExportParameters(
        # basics
        wavFiles=False,
        acousticScenarioPlot=False,
        sroEstimPerfPlot=False,
        metricsPlot=True,
        waveformsAndSpectrograms=True,
        bestPerfReference=True,
        # others
        filterNormsPlot=False,
        conditionNumberPlot=False,
        convergencePlot=False,
        mmsePerfPlot=False,
        mseBatchPerfPlot=False,
        # pickles
        danseOutputsFile=False,
        metricsFile=True,
        # folder
        exportFolder=a.testParams.exportParams.exportFolder,
    )
    # Update test parameters
    a.testParams.exportParams = newExportParams
    a.testParams.exportParams.exportFolder += '\\further_pp'
    baseExportFolder = a.testParams.exportParams.exportFolder

    # ------------------ 06.10.2023 test
    frameDur = params['frameDur']  # [s] duration of metrics computation frame
    tmax = params['tmax']  # [s] full simulation signal duration
    frameShift = params['frameShift']  # [s] frame shift

    # Frames start times
    if frameDur is None:
        t = np.arange(0, tmax - 5, frameShift)
    else:
        t = np.arange(0, tmax - frameDur, frameShift)
    for ii in range(len(t)):
        # Change other as desired
        a.testParams.danseParams.startComputeMetricsAt =\
            f'after_{int(t[ii])}s'
        if frameDur is not None:
            a.testParams.danseParams.endComputeMetricsAt =\
                f'after_{int(t[ii]) + frameDur}s'
            subfolder = f'dur{int(frameDur)}s'
        else:
            subfolder = 'full'
        a.testParams.exportParams.exportFolder = baseExportFolder +\
            f'\\{subfolder}\\start_{int(t[ii])}s'
        # Ensure folder exists
        a.testParams.exportParams.check_export_folder()
        # Ensure overall parameters consistency
        a.ensure_consistency()
        # Perform post-processing
        postprocess(
            out=a.danseOutputs,
            wasnObj=a.wasnObj,
            room=None,
            p=a.testParams,
            bypassGlobalPickleExport=True,
        )
    # ------------------

    print('Further post-processing done.')


def pp_20231006_combine_metrics(
        foldername,
        frameDur,
        fs=16e3,
        metricsToPlot=['stoi','snr']
    ):
    """
    Plots the evolution of metrics as a function of the chunk of signal over
    which they are computed - used to evaluated appropriate time chunk to use
    to evaluate performance of online speech enhancement algorithms such as
    DANSE or the MWF.
    """
    # Get list of subfolders
    if frameDur is None:
        sf = 'full'
    else:
        sf = f'dur{int(frameDur)}s'
    subfolders = [f.path for f in os.scandir(foldername + f'\\further_pp\\{sf}')\
                  if f.is_dir() and f.name.startswith('start_')]

    # Get list of metrics.pkl files
    metricsFiles = []
    for subfolder in subfolders:
        metricsFiles.append(subfolder + '\\metrics.pkl')

    # Read metrics.pkl files
    startTimes = []  # metrics computation start times
    metricsDANSE = dict([(metric, []) for metric in metricsToPlot])  # metrics
    metricsCentr = dict([(metric, []) for metric in metricsToPlot])  # metrics
    metricsLocal = dict([(metric, []) for metric in metricsToPlot])  # metrics
    metricsRaw = dict([(metric, []) for metric in metricsToPlot])  # metrics
    for metricsFile in metricsFiles:
        with open(metricsFile, 'rb') as f:
            metricsObj = pickle.load(f)
            startTimes.append(metricsObj.startIdx[0] / fs)
            for metric in metricsToPlot:
                metricsDANSE[metric].append(
                    getattr(metricsObj, metric)['Node1'].after
                )
                metricsCentr[metric].append(
                    getattr(metricsObj, metric)['Node1'].afterCentr
                )
                metricsLocal[metric].append(
                    getattr(metricsObj, metric)['Node1'].afterLocal
                )
                metricsRaw[metric].append(
                    getattr(metricsObj, metric)['Node1'].before
                )
    # Order according to start times
    startTimes = np.array(startTimes)
    idx = np.argsort(startTimes)
    startTimes = startTimes[idx]
    for metric in metricsToPlot:
        metricsDANSE[metric] = np.array(metricsDANSE[metric])[idx]
        metricsCentr[metric] = np.array(metricsCentr[metric])[idx]
        metricsLocal[metric] = np.array(metricsLocal[metric])[idx]
        metricsRaw[metric] = np.array(metricsRaw[metric])[idx]

    # Plot
    fig, axes = plt.subplots(len(metricsToPlot), 1, sharex=True)
    fig.set_size_inches(12.5 * 0.7, 6.5 * 0.7)
    for ii, metric in enumerate(metricsToPlot):
        axes[ii].plot(startTimes, metricsRaw[metric], '.-', label='Raw')
        axes[ii].plot(startTimes, metricsLocal[metric], '.-', label='Local')
        axes[ii].plot(startTimes, metricsCentr[metric], '.-', label='Centralized')
        axes[ii].plot(startTimes, metricsDANSE[metric], '.-', label='DANSE')
        axes[ii].set_ylabel(metric)
        axes[ii].set_xlabel('Metrics computation start time [s]')
        axes[ii].grid()
        axes[ii].set_xlim([0, np.amax(startTimes)])
        if ii == 0:
            axes[ii].legend(loc='lower left')
        if 'stoi' in metric:
            axes[ii].set_ylim([0, 1])
        else:
            axes[ii].set_ylim([0, axes[ii].get_ylim()[1]])
    plt.tight_layout()

    # Save
    plt.savefig(foldername + f'\\further_pp\\{sf}\\metrics_combined.png', dpi=300)
    plt.savefig(foldername + f'\\further_pp\\{sf}\\metrics_combined.pdf')

if __name__ == '__main__':
    sys.exit(main())