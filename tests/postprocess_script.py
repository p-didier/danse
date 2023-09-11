# Purpose of script:
# Performs further post-processing of the results of a given test, without
# having to re-run the test.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
from danse_toolbox.d_classes import OutputsForPostProcessing, ExportParameters
from .sandbox import postprocess

def main(foldername=''):
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
        # folder
        exportFolder=a.testParams.exportParams.exportFolder,
    )
    # Update test parameters
    a.testParams.exportParams = newExportParams
    a.testParams.exportParams.exportFolder += '\\further_pp'

    # Change other as desired
    a.testParams.danseParams.startComputeMetricsAt = 'after_3s'

    # Ensure consistency
    a.ensure_consistency()

    # Perform post-processing
    postprocess(
        out=a.danseOutputs,
        wasnObj=a.wasnObj,
        room=None,
        p=a.testParams,
        bypassGlobalPickleExport=True,
    )
    
    print('Further post-processing done.')

if __name__ == '__main__':
    sys.exit(main())