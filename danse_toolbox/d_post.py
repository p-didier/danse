# Post-processing functions and scripts for, e.g.,
# visualizing DANSE outputs.
#
# ~created on 20.10.2022 by Paul Didier
import os
import gzip
import time
import copy
import pickle
import numpy as np
from pathlib import Path
from scipy.io import wavfile
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from danse_toolbox.d_eval import *
from dataclasses import dataclass, fields
import danse_toolbox.dataclass_methods as met
from siggen.classes import Node, WASNparameters, WASN
from danse_toolbox.d_batch import BatchDANSEvariables
from danse_toolbox.d_base import DANSEparameters, get_stft, get_istft, get_y_tilde_batch
from danse_toolbox.d_classes import DANSEvariables, ConditionNumbers, TestParameters
from siggen.utils import plot_asc_3d

@dataclass
class ConvergenceData:
    DANSEfilters: np.ndarray = np.array([])
    DANSEfiltersCentr: np.ndarray = np.array([])

@dataclass
class DANSEoutputs(DANSEparameters):
    """
    Dataclass to assemble all useful outputs
    of the DANSE algorithm.
    """
    initialised : bool = False
    # ^^^ turns to True when `self.from_variables()` is called
    
    def import_params(self, p: DANSEparameters):
        self.__dict__.update(p.__dict__)
    
    def from_variables(self, dv: DANSEvariables):
        """
        Selects useful output values from `DANSEvariables` object
        after DANSE processing.
        """
        # Inits
        self.TDdesiredSignals_est_c = None
        self.STFTDdesiredSignals_est_c = None
        self.TDdesiredSignals_est_l = None
        self.STFTDdesiredSignals_est_l = None
        self.TDdesiredSignals_est_ssbc = None
        self.STFTDdesiredSignals_est_ssbc = None
        self.TDfiltSpeech_c = None
        self.STFTfiltSpeech_c = None
        self.TDfiltNoise_c = None
        self.STFTfiltNoise_c = None
        self.TDfiltSpeech_l = None
        self.STFTfiltSpeech_l = None
        self.TDfiltNoise_l = None
        self.STFTfiltNoise_l = None
        self.TDfiltSpeech_ssbc = None
        self.STFTfiltSpeech_ssbc = None
        self.TDfiltNoise_ssbc = None
        self.STFTfiltNoise_ssbc = None

        # Original microphone signals
        self.micSignals = dv.yin
        # MMSE cost
        if self.simType == 'batch':
            self.mmseCost = dv.mmseCost  # <-- initialised in `BatchDANSEvariables` class
            self.mmseCostInit = dv.mmseCostInit  # <-- initialised in `BatchDANSEvariables` class
            if self.computeLocal:
                self.mmseCostLocal = dv.mmseCostLocal  # <-- initialised in `BatchDANSEvariables` class
            if self.computeCentralised:
                self.mmseCostCentr = dv.mmseCostCentr  # <-- initialised in `BatchDANSEvariables` class
        # DANSE desired signal estimates
        self.TDdesiredSignals_est = dv.d
        self.STFTDdesiredSignals_est = dv.dhat
        if self.computeCentralised:
            # Centralised desired signal estimates
            self.TDdesiredSignals_est_c = dv.dCentr
            self.STFTDdesiredSignals_est_c = dv.dHatCentr
        if self.computeLocal:
            # Local desired signal estimates
            self.TDdesiredSignals_est_l = dv.dLocal
            self.STFTDdesiredSignals_est_l = dv.dHatLocal
        if self.computeSingleSensorBroadcast:
            # Single-sensor broadcast desired signal estimates
            self.TDdesiredSignals_est_ssbc = dv.dSSBC
            self.STFTDdesiredSignals_est_ssbc = dv.dHatSSBC
        # DANSE fused signals
        self.TDfusedSignals = dv.zFullTD
        if hasattr(dv, 'etaMkFullTD'):
            self.TDfusedSignalsTI = dv.etaMkFullTD  # <-- initialised in `TIDANSEVariables` class
        # SROs
        self.SROgroundTruth = dv.SROsppm
        self.SROsEstimates = dv.SROsEstimates
        self.SROsResiduals = dv.SROsResiduals
        self.flagIterations = dv.flagIterations
        self.firstUpRefSensor = dv.firstDANSEupdateRefSensor
        # Filters
        self.filters = dv.wTilde
        if self.simType == 'online':
            self.filtersEXT = dv.wTildeExt
            self.yinSTFT = dv.yinSTFT
            self.yCentrBatch = dv.yCentrBatch
            self.neighbors = dv.neighbors
            self.fs = dv.fs
            self.cleanTargets = dv.cleanSpeechSignalsAtNodes
            self.mseCostOnline = dv.mseCostOnline
            if self.computeCentralised:
                self.mseCostOnline_c = dv.mseCostOnline_c
            if self.computeLocal:
                self.mseCostOnline_l = dv.mseCostOnline_l
        if self.computeCentralised:
            self.filtersCentr = dv.wCentr
        else:
            self.filtersCentr = None
        if self.computeLocal:
            self.filtersLocal = dv.wLocal
        else:
            self.filtersLocal = None
        # Condition numbers
        if self.saveConditionNumber:
            self.condNumbers = dv.condNumbers
        # Other useful things
        self.beta = dv.expAvgBeta
        self.vadFrames = dv.oVADframes

        # Show initialised status
        self.initialised = True

        return self
    
    def from_snr_signals(self, snrSigs: dict):
        """
        Selects output values from `snrSigs` dict, for
        subsequent SNR computation with filtered speech-only
        and noise-only signals. The dictionary is created in
        `d_classes.generate_signals_for_snr_computation`.
        """
        self.TDfiltSpeech = snrSigs['s']
        self.TDfiltNoise = snrSigs['n']
        self.TDfiltSpeech_c = snrSigs['s_c']
        self.TDfiltNoise_c = snrSigs['n_c']
        self.TDfiltSpeech_l = snrSigs['s_l']
        self.TDfiltNoise_l = snrSigs['n_l']
        self.TDfiltSpeech_ssbc = snrSigs['s_ssbc']
        self.TDfiltNoise_ssbc = snrSigs['n_ssbc']
        return self
    
    def include_best_perf_data(
            self,
            outBP: BatchDANSEvariables,
            sigsSnr: dict
        ):
        """
        Includes the "best performance" data (computed in centralized,
        no SROs, batch mode).
        The `sigsSnr` dict is created in `d_classes.generate_signals_for_snr_computation`
        and contains the signals used for SNR computation.
        """
        self.bestPerfData = {
            'dCentr': outBP.dCentr,
            'dHatCentr': outBP.dHatCentr,
            'dCentr_s': sigsSnr['s_bp'],
            # 'dHatCentr_s': outBP.dHatCentr_s,
            'dCentr_n': sigsSnr['n_bp'],
            # 'dHatCentr_n': outBP.dHatCentr_n,
            'mseCostCentr': outBP.mmseCostCentr,
            'wCentr': outBP.wCentr,
            'fs': outBP.baseFs,
            'cleanSpeech': np.array([
                outBP.cleanSpeechSignalsAtNodes[k][
                    :, outBP.referenceSensor
                ] for k in range(outBP.nNodes)
            ]).T,
            'cleanNoise': np.array([
                outBP.cleanNoiseSignalsAtNodes[k][
                    :, outBP.referenceSensor
                ] for k in range(outBP.nNodes)
            ]).T,
        }

    def check_init(self):
        """Check if object is correctly initialised."""
        if not self.initialised:
            return ValueError('The DANSEoutputs object is empty.')

    def save(self, foldername, light=False, exportType='pkl'):
        """Saves dataclass to file."""
        if light:
            mycls = copy.copy(self)
            for f in fields(mycls):
                if 'signal' in f.name.lower():
                    delattr(mycls, f)
            met.save(self, foldername, exportType=exportType)
        else:
            met.save(self, foldername, exportType=exportType)

    def save_metrics(self, foldername):
        """Saves metrics to file."""
        self.check_init()  # check if object is correctly initialised
        with open(f'{foldername}/metrics.pkl', 'wb') as f:
            pickle.dump(self.metrics, f)

    def load(self, foldername, dataType='pkl') -> 'DANSEoutputs':
        """Loads dataclass to Pickle archive in folder `foldername`."""
        return met.load(self, foldername, silent=True, dataType=dataType)

    def export_sounds(self, wasn, exportFolder, fullyConnected=True):
        self.check_init()  # check if object is correctly initialised
        export_sounds(self, wasn, exportFolder, fullyConnected)

    def plot_batch_cost_at_each_update(self):
        """
        Plots the evolution of the MMSE cost at each update in batch-mode,
        i.e., computes the cost over the entire signal filtered with the 
        filters at iteration `i`, for each iteration, and plots that cost.
        """
        # Compute batch cost for each update
        mseCostRaw = np.zeros(self.nNodes)
        for k in range(self.nNodes):
            # Get unprocessed signals cost
            mseCostRaw[k] = np.mean(np.abs(
                    self.cleanTargets[k][:, self.referenceSensor] -\
                        self.micSignals[k][:, self.referenceSensor]
            )**2)
        
        # Plot batch cost for each update, one subplot by node
        nRows = int(np.floor(np.sqrt(self.nNodes))) + 1
        nCols = int(np.ceil(self.nNodes / nRows))
        fig, axes = plt.subplots(nRows, nCols, sharex=True, sharey=True)
        fig.set_size_inches(8.5, 6.5)
        for k in range(self.nNodes):
            if nRows == 1:
                currAx = axes[k]
            else:
                currAx = axes[k // nCols, k % nCols]
            currAx.plot(
                self.mseCostOnline[:, k],
                'C3-',
                label='DANSE'
            )
            # Plot local and centralized costs
            if self.computeLocal:
                currAx.plot(
                    self.mseCostOnline_l[:, k],
                    'C1-',
                    label='Local'
                )
            if self.computeCentralised:
                currAx.plot(
                    self.mseCostOnline_c[:, k],
                    'C2-',
                    label='Centralized'
                )
            currAx.plot(
                mseCostRaw[k] * np.ones_like(self.mseCostOnline[:, k]),
                'C0--',
                label='Raw'
            )
            # Plot legend and labels
            currAx.grid()
            currAx.set_xlabel('DANSE iteration $i$')
            currAx.set_ylabel('MSE cost $E\{ | d - \hat{d} |^2 \}$')
            currAx.set_title(f'Node {k+1}')
            if k == 0:
                currAx.legend(loc='upper right')
            currAx.set_yscale('log')  # logarithmic scale for better visualization
        fig.tight_layout()
        return fig


    def plot_mmse_cost(self):
        """Plots the evolution of the MMSE cost."""
        self.check_init()
        fig, axes = plt.subplots(1,1)
        fig.set_size_inches(8.5, 3.5)
        for k in range(self.nNodes):
            axes.plot(
                np.insert(self.mmseCost[:, k], 0, self.mmseCostInit[k]),
                f'C{k}.-',
                label=f'Node {k+1}'
            )
            # Plot local and centralized costs
            if self.computeLocal:
                axes.plot(
                    np.insert(
                        self.mmseCostLocal[k] * np.ones_like(self.mmseCost[:, k]),
                        0,
                        self.mmseCostInit[k]
                    ),
                    f'C{k}:',
                    label=f'Local node {k+1}'
                )
            if self.computeCentralised:
                axes.plot(
                    np.insert(
                        self.mmseCostCentr[k] * np.ones_like(self.mmseCost[:, k]),
                        0,
                        self.mmseCostInit[k]
                    ),
                    f'C{k}--',
                    label='Centralised'
                )
        # Plot legend and labels
        axes.grid()
        axes.set_xlabel('DANSE iteration $i$')
        axes.set_ylabel('MMSE cost $E\{ | d - \hat{d} |^2 \}$')
        axes.legend(loc='upper right')
        axes.set_yscale('log')  # logarithmic scale for better visualization
        fig.tight_layout()
        return fig
    
    def plot_framewise_mse_online_cost(self, wasn: list[Node]):
        """Plots the evolution of the MSE cost in online-mode."""
        mseCost = np.zeros((0, self.nNodes))
        mseCost_c = np.zeros((0, self.nNodes))
        idxBegin = 0
        idxEnd = self.DFTsize
        while idxEnd <= self.TDdesiredSignals_est.shape[0]:
            currMSE = np.zeros((1, self.nNodes))
            currMSE_c = np.zeros((1, self.nNodes))
            for k in range(len(wasn)):
                currMSE[:, k] = np.mean(
                    np.abs(wasn[k].cleanspeechRefSensor[idxBegin:idxEnd] -\
                        self.TDdesiredSignals_est[idxBegin:idxEnd, k])**2,
                    axis=0
                )
                currMSE_c[:, k] = np.mean(
                    np.abs(wasn[k].cleanspeechRefSensor[idxBegin:idxEnd] -\
                        self.TDdesiredSignals_est_c[idxBegin:idxEnd, k])**2,
                    axis=0
                )
            mseCost = np.vstack((mseCost, currMSE))
            mseCost_c = np.vstack((mseCost_c, currMSE_c))
            idxBegin += self.Ns
            idxEnd += self.Ns

        # Plot MSE cost with logarithmic scale
        fig, axes = plt.subplots(1,1)
        fig.set_size_inches(8.5, 3.5)
        axes.plot(mseCost, '-')
        axes.plot(mseCost_c, '--')
        axes.set_yscale('log')
        axes.set_xlabel('Window index $i$')
        axes.set_ylabel('MSE cost $E_N\{ | \\tilde{d}_k^i(n) - \\hat{d}_k(n) |^2 \}$')
        axes.grid()
        axes.legend([f'Node {k+1}' for k in range(self.nNodes)] +
                    [f'Centr. node {k+1}' for k in range(self.nNodes)])
        fig.tight_layout()
        return fig

    def plot_filter_evol(
            self,
            exportFolder=None,
            exportNormsAsPickle=False,
            plots=['norm', 'real-imag'],
            tiDANSEflag=False,
        ):
        """
        Plots a visualization of the evolution of filters in DANSE.

        Parameters
        ----------
        exportFolder : str, optional
            Folder to export figures to. If None, figures are not exported.
        exportNormsAsPickle : bool, optional
            If True, export filters norms as Pickle files.
        plots : list of str, optional
            List of plots to export. Possible values are:
            - 'norm' : plot of the norm of each filter
            - 'real-imag' : plot of the real and imaginary parts of each filter
        tiDANSEflag : bool, optional
            If True, we consider TI-DANSE (else, DANSE in a fully connected
            WASN).
        """
        self.check_init()  # check if object is correctly initialised

        def _export(figs, dataFigs, subfolder='filtNorms'): 
            # Export figures
            if exportFolder is not None:
                for title, fig in figs.items():
                    fullExportFolder = f'{exportFolder}/{subfolder}'
                    if not os.path.exists(fullExportFolder):
                        os.makedirs(fullExportFolder)
                    fig.savefig(f'{fullExportFolder}/{title}.png', dpi=300)
                    fig.savefig(f'{fullExportFolder}/{title}.pdf')
            else:
                plt.close(fig)
            # Export data
            if exportNormsAsPickle:
                fullExportFolder = f'{exportFolder}/{subfolder}'
                if not os.path.exists(fullExportFolder):
                    os.makedirs(fullExportFolder)
                with open(f'{fullExportFolder}/{subfolder}.pkl', 'wb') as f:
                    pickle.dump(dataFigs, f)

        for ii in range(len(plots)):
            if plots[ii] not in ['norm', 'real-imag']:
                raise NotImplementedError(f'Unknown/unimplemented plot type {plots[ii]}')
            
            kwargs = {
                'filters': [np.abs(filt) for filt in self.filters],
                'filtersEXT': [np.abs(filt) for filt in self.filtersEXT],
                'filtersCentr': [np.abs(filt) for filt in self.filtersCentr]\
                    if self.filtersCentr is not None else None,
                'nSensorsPerNode': self.nSensorPerNode,
                'refSensorIdx': self.referenceSensor,
                'bestPerfData': self.bestPerfData,
                'tiDANSEflag': tiDANSEflag,
            }
            if plots[ii] == 'norm':
                figs, dataFigs = plot_filters(figPrefix='filtnorm', **kwargs)
                _export(figs, dataFigs, subfolder='filtNorms')
            if plots[ii] == 'real-imag':
                # Plot real part figures
                figs, dataFigs = plot_filters(figPrefix='filtreal', **kwargs)
                _export(figs, dataFigs, subfolder='filtReal')
                # Plot iamginary part
                figs, dataFigs = plot_filters(figPrefix='filtimag', **kwargs)
                _export(figs, dataFigs, subfolder='filtImag')

    def plot_cond(self, exportFolder=None):
        """Plots a visualization of the condition numbers in DANSE."""
        self.check_init()  # check if object is correctly initialised        
        fig = plot_cond_numbers(self.condNumbers, self.nSensorPerNode)
        if exportFolder is not None:
            fig.savefig(f'{exportFolder}/cond_numbers.png', dpi=300)
            fig.savefig(f'{exportFolder}/cond_numbers.pdf')
        else:
            plt.close(fig)

    def plot_perf(
            self,
            wasn,
            exportFolder=None,
            metricsToPlot=False,
            snrYlimMax=None
        ):
        """Plots DANSE performance."""
        self.check_init()  # check if object is correctly initialised
        self.metrics = compute_metrics(self, wasn, metricsToPlot)
        figStatic, figDynamic = plot_metrics(self, metricsToPlot, snrYlimMax)
        if exportFolder is not None:
            figStatic.savefig(f'{exportFolder}/metrics.png', dpi=300)
            figStatic.savefig(f'{exportFolder}/metrics.pdf')
            if figDynamic is not None:
                figDynamic.savefig(f'{exportFolder}/metrics_dyn.png', dpi=300)
                figDynamic.savefig(f'{exportFolder}/metrics_dyn.pdf')
        else:
            plt.close(figStatic)
            if figDynamic is not None:
                plt.close(figDynamic)

    def plot_convergence(self, exportFolder, bypassExport=False):
        """
        Shows convergence of DANSE.
        Created on 19.01.2023 (as a result of OJSP reviewers' suggestions).
        """

        DANSEfilters_all = np.zeros((
            self.filters[0].shape[1],
            self.filters[0].shape[0],  # inverted dimensions because of transpose
            self.nNodes
        ), dtype=complex)
        DANSEfiltersCentr_all = np.zeros((
            self.filtersCentr[0].shape[1],
            self.filtersCentr[0].shape[0],  # inverted dimensions because of transpose
            self.nNodes
        ), dtype=complex)
        for k in range(self.nNodes):
            np.seterr(divide = 'ignore')  # ignore division by zero
            diffFiltersReal = 20 * np.log10(np.mean(np.abs(
                np.real(self.filters[k][:, :, self.referenceSensor].T) - \
                np.real(self.filtersCentr[k][:, :, self.referenceSensor].T)
            ), axis=1))
            diffFiltersImag = 20 * np.log10(np.mean(np.abs(
                np.imag(self.filters[k][:, :, self.referenceSensor].T) - \
                np.imag(self.filtersCentr[k][:, :, self.referenceSensor].T)
            ), axis=1))
            np.seterr(divide = 'warn')  # reset to default

            fig, axes = plt.subplots(1, 1)
            fig.set_size_inches(5.5 * 1.5, 3.5 * 1.5)
            #
            # Plot the VAD itself
            scaling = 0.5 * np.amax(np.concatenate(
                (diffFiltersReal, diffFiltersImag)
            ))  # scaling for VAD plot
            axes.plot(
                self.vadFrames[k] * scaling,
                color='0.5',
                label='VAD',
                zorder=0
            )
            #
            axes.plot(
                diffFiltersReal,
                label=f'$20\\log_{{10}}(E_{{\\nu}}\\{{|Re(\\tilde{{w}}_{{{k+1}{k+1},{self.referenceSensor+1}}}[\\nu,i]) - Re(\\hat{{w}}_{{{k+1}{k+1},{self.referenceSensor+1}}}[\\nu,i])|\\}})$',
                zorder=1
            )
            axes.plot(
                diffFiltersImag,
                label=f'$20\\log_{{10}}(E_{{\\nu}}\\{{|Im(\\tilde{{w}}_{{{k+1}{k+1},{self.referenceSensor+1}}}[\\nu,i]) - Im(\\hat{{w}}_{{{k+1}{k+1},{self.referenceSensor+1}}}[\\nu,i])|\\}})$',
                zorder=2
            )
            #
            axes.set_title(f'DANSE convergence towards centr. MWF estimate: node {k+1}')
            axes.set_xlim([0, len(diffFiltersReal)])
            # Make double x-axesis (top and bottom)
            axes2 = axes.twiny()
            axes2.set_xlabel('DANSE updates (frame index $i$)', loc='left')
            axes2.set_xticks(copy.copy(axes.get_xticks()))
            xticks = np.linspace(
                start=0, stop=self.filters[k].shape[1], num=9
            )
            axes.set_xticks(xticks)
            axes.set_xticklabels(
                np.round(
                    xticks * self.Ns / self.baseFs + self.firstUpRefSensor
                    , 2
                )
            )
            # Instantiate a second axes that shares the same x-axis for the
            # VAD plot (y axis on the right).
            axes3 = axes.twinx()
            axes3.set_ylim(axes.get_ylim())
            axes3.set_yticks([0, scaling])
            axes3.set_yticklabels(
                ['Speech\nAbsent', 'Speech\nPresent'],
                rotation=90,
                va='center'
            )
            #
            axes.set_xlabel('Time [s]', loc='left')
            axes.set_ylabel('[dB]')
            axes.legend()
            axes.grid()
            axes.set_axisbelow(True)  # grid below the plot
            #
            plt.tight_layout()
            # Export
            if not bypassExport:
                fullExportFolder = f'{exportFolder}/conv'
                if not os.path.exists(fullExportFolder):
                    os.makedirs(fullExportFolder)
                fig.savefig(f'{fullExportFolder}/converg_node{k+1}.png', dpi=300)
                fig.savefig(f'{fullExportFolder}/converg_node{k+1}.pdf')
            # Aggregate for export
            DANSEfilters_all[:, :, k] =\
                self.filters[k][:, :, self.referenceSensor].T
            DANSEfiltersCentr_all[:, :, k] =\
                self.filtersCentr[k][:, :, self.referenceSensor].T
        
        # Export for further post-processing
        if not bypassExport:
            convData = ConvergenceData(
                DANSEfilters=DANSEfilters_all,
                DANSEfiltersCentr=DANSEfiltersCentr_all
            )
            pickle.dump(
                convData,
                gzip.open(
                    f'{exportFolder}/{type(convData).__name__}.pkl.gz', 'wb'
                )
            )

        return None

    def plot_sro_perf(self, Ns, fs, xaxistype='both'):
        """
        Shows evolution of SRO estimates / residuals through time.
        
        Parameters
        ----------
        fs : float or int
            Sampling frequency of the reference node [Hz].
        Ns : int
            Number of new samples per DANSE iteration.
        xaxistype : str
            Type of x-axis ticks/label:
            "iterations" = DANSE iteration indices
            "time" = time instants [s]
            "both" = both of the above

        Returns
        -------
        fig : figure handle
            Figure handle for further processing.
        """
        nNodes = len(self.SROsResiduals)

        self.neighbourIndex = [0 if k > 0 else 1 for k in range(nNodes)] # TODO: for now just valid for fully connected topology
        idxFirstNeighbour = [0 for _ in range(nNodes)]   # TODO: for now just valid for fully connected topology

        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(111)
        for k in range(nNodes):
            if self.compensateSROs:
                if k == 0:
                    ax.plot(
                        self.SROsResiduals[k][:, idxFirstNeighbour[k]] * 1e6, f'C{k}-',
                        label=f'$\\Delta\\hat{{\\varepsilon}}_{{{k+1}{self.neighbourIndex[k]+1}}}$'
                    )
                    ax.plot(
                        self.SROsEstimates[k][:, idxFirstNeighbour[k]] * 1e6,
                        f'C{k}--', label=f'$\\hat{{\\varepsilon}}_{{{k+1}{self.neighbourIndex[k]+1}}}$'
                    )
                else:  # adapt labels for condensed legend
                    ax.plot(
                        self.SROsResiduals[k][:, idxFirstNeighbour[k]] * 1e6,
                        f'C{k}-',
                        label=f'Node {k+1}'
                    )
                    ax.plot(self.SROsEstimates[k][:, idxFirstNeighbour[k]] * 1e6, f'C{k}--')
            else:
                if k == 0:
                    ax.plot(
                        self.SROsResiduals[k][:, idxFirstNeighbour[k]] * 1e6,
                        f'C{k}-',
                        label=f'$\\hat{{\\varepsilon}}_{{{k+1}{self.neighbourIndex[k]+1}}}$'
                    )
                else:  # adapt labels for condensed legend
                    ax.plot(
                        self.SROsResiduals[k][:, idxFirstNeighbour[k]] * 1e6,
                        f'C{k}-',
                        label=f'$(k,q)=({k+1},{self.neighbourIndex[k]+1})$'
                    )
            # Always showing ground truths with respect to 1st neighbour:
            # - For node k==0: 1st neighbour is k==1
            # - For any other node: 1st neighbour is k==0
            if k == 0:
                ax.hlines(
                    y=(self.SROgroundTruth[self.neighbourIndex[k]] -\
                        self.SROgroundTruth[k]),
                    xmin=0,
                    xmax=len(self.SROsResiduals[0]),
                    colors=f'C{k}',
                    linestyles='dotted',
                    label=f'$\\varepsilon_{{{k+1}{self.neighbourIndex[k]+1}}}$'
                )
            else:  # adapt labels for condensed legend
                
                ax.hlines(
                    y=(self.SROgroundTruth[self.neighbourIndex[k]] -\
                        self.SROgroundTruth[k]),
                    xmin=0,
                    xmax=len(self.SROsResiduals[0]),
                    colors=f'C{k}',
                    linestyles='dotted'
                )
        ylims = ax.get_ylim()

        for k in range(nNodes):
            if len(self.flagIterations[k]) > 0:
                if k == 0:
                    ax.vlines(
                        x=self.flagIterations[k],
                        ymin=np.amin(ylims),
                        ymax=np.amax(ylims),
                        colors=f'C{k}',
                        linestyles='dashdot',
                        label=f'Flags {k+1}--{self.neighbourIndex[k]+1}'
                    )
                else:
                    ax.vlines(
                        x=self.flagIterations[k],
                        ymin=np.amin(ylims),
                        ymax=np.amax(ylims),
                        colors=f'C{k}',
                        linestyles='dashdot'
                    )
        ax.grid()
        ax.set_ylabel('[ppm]')
        ax.set_xlabel('DANSE iteration $i$', loc='left')
        ax.set_xlim([0, len(self.SROsResiduals[k])])
        handles, labels = plt.gca().get_legend_handles_labels()

        # Adapt legend handles/labels order
        if not self.compensateSROs and len(self.flagIterations[0]) > 0:
            order = [0] + list(np.arange(nNodes, nNodes+2)) +\
                list(np.arange(1, nNodes))
        elif not self.compensateSROs:
            order = [0] + list(np.arange(nNodes-1, nNodes+1)) +\
                list(np.arange(1, nNodes-1))
        elif len(self.flagIterations[0]) > 0:
            # if there were flags, change legend handles order counting
            # the flags-handle in.
            order = [0,1] + list(np.arange(nNodes+1, nNodes+3)) +\
                list(np.arange(2, nNodes+1))
        else:  
            # otherwise, do not count the flags-handle in, because it
            # has not been generated.
            order = [0,1] + list(np.arange(nNodes, nNodes+2)) +\
                list(np.arange(2, nNodes))

        plt.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            bbox_to_anchor=(1.05, 1.05)
        )
        if xaxistype == 'both':
            ax2 = ax.twiny()
            ax2.set_xlabel('DANSE iteration $i$', loc='left')
            ax2.set_xticks(ax.get_xticks())
        if xaxistype in ['time', 'both']:
            xticks = np.linspace(
                start=0,
                stop=len(self.SROsResiduals[0]),
                num=9
            )
            ax.set_xticks(xticks)
            ax.set_xticklabels(
                np.round(xticks * Ns / fs + self.firstUpRefSensor, 2)
            )
            ax.set_xlabel('Time at reference node [s]', loc='left')
        plt.title('SRO estimation through time')
        plt.tight_layout()
        
        return fig

    def plot_sigs(self, wasn, exportFolder=None):
        """Plots signals before/after DANSE."""
        figs = plot_signals_all_nodes(self, wasn)
        if exportFolder is not None:
            for k in range(len(figs)):
                figs[k].savefig(f'{exportFolder}/sigs_node{k+1}.png', dpi=300)


def compute_metrics(
        out: DANSEoutputs,
        wasn: list[Node],
        metricsToPlot: list[str]=['snr', 'stoi']
    ) -> EnhancementMeasures:
    """
    Compute and store evaluation metrics after signal enhancement.

    Parameters
    ----------
    out : `DANSEoutputs` object
        DANSE run outputs.
    wasn : list of `Node` objects
        WASN under consideration.
    metricsToPlot : list of str, optional
        List of metrics to compute. Possible values are:
        - 'snr' : unweighted SNR
        - 'sisnr' : speech-intelligibility-weighted SNR
        - 'fwSNRseg' : frequency-weighted segmental SNR
        - 'stoi'/'estoi' : extended Short-Time Objective Intelligibility
        - 'pesq' : Perceptual Evaluation of Speech Quality
    
    Returns
    -------
    measures : `EnhancementMeasures` object
        The enhancement metrics.
    """

    def _ndict(N):
        """Initialize a 'node' dictionary."""
        return dict([(key, []) for key in [f'Node{n+1}' for n in range(N)]])

    # Initialisations
    startIdx = np.zeros(out.nNodes, dtype=int)
    endIdx = np.zeros(out.nNodes, dtype=int)
    snr = _ndict(out.nNodes)  # Unweighted SNR
    sisnr = _ndict(out.nNodes)  # Speech-Intelligibility-weighted SNR
    fwSNRseg = _ndict(out.nNodes)  # Frequency-weighted segmental SNR
    stoi = _ndict(out.nNodes)  # (extended) Short-Time Objective Intelligibility
    pesq = _ndict(out.nNodes)  # Perceptual Evaluation of Speech Quality
    tStart = time.perf_counter()
    for k in range(out.nNodes):
        # Derive starting/ending samples for metrics computations
        startIdx[k] = int(np.floor(wasn[k].metricStartTime * wasn[k].fs))
        endIdx[k] = int(np.floor(wasn[k].metricEndTime * wasn[k].fs))
        print(f"Node {k+1}: computing metrics from {startIdx[k] + 1}th sample (t_start = {np.round(wasn[k].metricStartTime, 3)} s) till the {endIdx[k] + 1}th sample (t_end = {np.round(wasn[k].metricEndTime, 3)} s).")
        print(f'Computing metrics for node {k + 1}/{out.nNodes} (sensor {out.referenceSensor + 1}/{wasn[k].nSensors})...')

        # Compute starting indices for centralised and local estimates
        TDdesiredSignals_est_c, TDdesiredSignals_est_l, TDdesiredSignals_est_ssbc = None, None, None
        TDfilteredSpeech_c, TDfilteredSpeech_l, TDfilteredSpeech_ssbc = None, None, None
        TDfilteredNoise_c, TDfilteredNoise_l, TDfilteredNoise_ssbc = None, None, None
        if out.computeCentralised:
            TDdesiredSignals_est_c = out.TDdesiredSignals_est_c[:, k]
            TDfilteredSpeech_c = out.TDfiltSpeech_c[:, k]
            TDfilteredNoise_c = out.TDfiltNoise_c[:, k]
            print(f"Node {k+1}: computing metrics for CENTRALISED PROCESSING from {startIdx[k] + 1}th sample on (t_start = {np.round(wasn[k].metricStartTime, 3)} s) till the {endIdx[k] + 1}th sample (t_end = {np.round(wasn[k].metricEndTime, 3)} s).")
        if out.computeLocal:
            TDdesiredSignals_est_l = out.TDdesiredSignals_est_l[:, k]
            TDfilteredSpeech_l = out.TDfiltSpeech_l[:, k]
            TDfilteredNoise_l = out.TDfiltNoise_l[:, k]
            print(f"Node {k+1}: computing metrics for LOCAL PROCESSING from {startIdx[k] + 1}th sample on (t_start = {np.round(wasn[k].metricStartTime, 3)} s) till the {endIdx[k] + 1}th sample (t_end = {np.round(wasn[k].metricEndTime, 3)} s).")
        if out.computeSingleSensorBroadcast:
            TDdesiredSignals_est_ssbc = out.TDdesiredSignals_est_ssbc[:, k]
            TDfilteredSpeech_ssbc = out.TDfiltSpeech_ssbc[:, k]
            TDfilteredNoise_ssbc = out.TDfiltNoise_ssbc[:, k]
            print(f"Node {k+1}: computing metrics for SINGLE-SENSOR BROADCAST from {startIdx[k] + 1}th sample on (t_start = {np.round(wasn[k].metricStartTime, 3)} s) till the {endIdx[k] + 1}th sample (t_end = {np.round(wasn[k].metricEndTime, 3)} s).")

        if out.simType == 'batch' and\
            endIdx[k] > wasn[k].cleanspeechRefSensor.shape[0] - out.DFTsize:
            # Discard the very end of the signal due to STFT/ISTFT artefacts
            endIdx[k] = int(
                wasn[k].cleanspeechRefSensor.shape[0] - out.DFTsize
            )
        
        metricsDict = get_metrics(
            # Clean speech mixture (desired signal)
            clean=np.squeeze(wasn[k].cleanspeechRefSensor),
            noiseOnly=np.squeeze(wasn[k].cleannoiseRefSensor),
            # Microphone signals
            noisy=wasn[k].data[:, out.referenceSensor],
            filtSpeech=out.TDfiltSpeech[:, k],
            filtNoise=out.TDfiltNoise[:, k],
            filtSpeech_c=TDfilteredSpeech_c,
            filtNoise_c=TDfilteredNoise_c,
            filtSpeech_l=TDfilteredSpeech_l,
            filtNoise_l=TDfilteredNoise_l,
            filtSpeech_ssbc=TDfilteredSpeech_ssbc,
            filtNoise_ssbc=TDfilteredNoise_ssbc,
            # DANSE outputs (desired signal estimates)
            enhan=out.TDdesiredSignals_est[:, k],
            enhan_c=TDdesiredSignals_est_c,
            enhan_l=TDdesiredSignals_est_l,
            enhan_ssbc=TDdesiredSignals_est_ssbc,
            # Start/end indices
            startIdx=startIdx[k],
            endIdx=endIdx[k],
            # Other parameters
            fs=wasn[k].fs,
            vad=wasn[k].vadCombined,
            dynamic=out.dynMetrics,
            gamma=out.gammafwSNRseg,
            fLen=out.frameLenfwSNRseg,
            metricsToPlot=metricsToPlot,
            bestPerfData=out.bestPerfData,
            k=k
        )

        for key in metricsDict.keys():
            if key == 'snr':
                snr[f'Node{k + 1}'] = metricsDict[key]
            elif key == 'sisnr':
                sisnr[f'Node{k + 1}'] = metricsDict[key]
            elif key == 'fwSNRseg':
                fwSNRseg[f'Node{k + 1}'] = metricsDict[key]
            elif key == 'stoi':
                stoi[f'Node{k + 1}'] = metricsDict[key]
            elif key == 'pesq':
                pesq[f'Node{k + 1}'] = metricsDict[key]

    print(f'All signal enhancement evaluation metrics computed in {np.round(time.perf_counter() - tStart, 3)} s.')

    # Group measures into EnhancementMeasures object
    metrics = EnhancementMeasures(
        fwSNRseg=fwSNRseg,
        stoi=stoi,
        pesq=pesq,
        snr=snr,
        sisnr=sisnr,
        startIdx=startIdx,
        endIdx=endIdx
    )

    return metrics

def plot_metrics(
        out: DANSEoutputs,
        metricsToPlot=['snr', 'estoi'],
        snrYlimMax=None
    ):
    """
    Visualize evaluation metrics.

    Parameters
    ----------
    out : `DANSEoutputs` object
        DANSE outputs.
    metricsToPlot : list[str]
        List of metrics to plot. Possible values are:
        - 'snr' : unweighted SNR
        - 'sisnr' : speech intelligibility-weighted SNR
        - 'estoi'/'stoi' : extended STOI
        - 'fwSNRseg' : frequency-weighted segmental SNR
        - 'pesq' : PESQ
    snrYlimMax : float or int
        If not None, set a particular y-axis limit for the SNR plot.
    """

    # Hard-coded variables
    barWidth = 1
    titlesDict = {
        'snr': 'SNR',
        'sisnr': 'SI-SNR',
        'estoi': 'eSTOI',
        'stoi': 'eSTOI',
        'fwSNRseg': 'fwSNRseg',
        'pesq': 'PESQ'
    }
    yLabelsDict = {
        'snr': '[dB]',
        'sisnr': '[dB]',
        'estoi': '[-]',
        'stoi': '[-]',
        'fwSNRseg': '[dB]',
        'pesq': '[-]'
    }

    # Prepare subplots for static metrics
    nCols = len(metricsToPlot)
    fig1 = plt.figure(figsize=(4 * len(metricsToPlot),3))
    
    for ii in range(nCols):
        ax = fig1.add_subplot(1, nCols, ii + 1)
        
        if metricsToPlot[ii] == 'snr':
            metricToPlot = out.metrics.snr
        elif metricsToPlot[ii] == 'sisnr':
            metricToPlot = out.metrics.sisnr
        elif 'stoi' in metricsToPlot[ii]:
            metricToPlot = out.metrics.stoi
        elif metricsToPlot[ii] == 'fwSNRseg':
            metricToPlot = out.metrics.fwSNRseg
        elif metricsToPlot[ii] == 'pesq':
            metricToPlot = out.metrics.pesq
        else:
            raise ValueError(f'Unknown metric {metricsToPlot[ii]}')
        # Plot
        metrics_subplot(ax, barWidth, metricToPlot)
        ax.set(
            title=titlesDict[metricsToPlot[ii]],
            ylabel=yLabelsDict[metricsToPlot[ii]]
        )
        if 'snr' in metricsToPlot[ii] and snrYlimMax is not None:
            ax.set_ylim([0, snrYlimMax])
        elif 'stoi' in metricsToPlot[ii]:
            ax.set_ylim([0, 1])
    ax.legend(bbox_to_anchor=(1, 0), loc="lower left")

    plt.tight_layout()

    # Check where dynamic metrics were computed
    flagsDynMetrics = np.zeros(len(fields(out.metrics)), dtype=bool)
    for ii, field in enumerate(fields(out.metrics)):
        if type(field) == dict:
            if getattr(out.metrics, field.name)['Node1'].dynamicFlag:
                flagsDynMetrics[ii] = True

    nDynMetrics = np.sum(flagsDynMetrics)

    if nDynMetrics > 0:
        # Prepare subplots for dynamic metrics
        if nDynMetrics < 4:
            nRows, nCols = 1, nDynMetrics
        else:
            nRows, nCols = 2, int(np.ceil(nDynMetrics / 2))
        fig2, axes = plt.subplots(nRows, nCols)
        fig2.set_figheight(2.5 * nRows)
        fig2.set_figwidth(5 * nCols)
        axes = axes.flatten()   # flatten axes array for easy indexing
        
        # Select dictionary elements
        dynMetricsNames = [fields(out.metrics)[ii].name\
            for ii in range(len(fields(out.metrics))) if flagsDynMetrics[ii]
        ]
        dynMetrics = [getattr(out.metrics, n) for n in dynMetricsNames]

        # Plot
        for ii, dynMetric in enumerate(dynMetrics):
            for nodeRef, value in dynMetric.items():
                metric = value.dynamicMetric
                idxColor = int(nodeRef[-1]) - 1
                axes[ii].plot(
                    metric.timeStamps,
                    metric.before,
                    color=f'C{idxColor}',
                    linestyle='--',
                    label=f'{nodeRef}: Before'
                )
                axes[ii].plot(
                    metric.timeStamps,
                    metric.after,
                    color=f'C{idxColor}',
                    linestyle='-',
                    label=f'{nodeRef}: After'
                )
            axes[ii].grid()
            axes[ii].set_title(dynMetricsNames[ii])
            if ii == 0:
                axes[ii].legend(loc='lower left', fontsize=8)
            axes[ii].set_xlabel('$t$ [s]')  
        plt.tight_layout()
    else:
        fig2 = None

    return fig1, fig2


def metrics_subplot(ax, barWidth=1, data: dict[Metric]=None):
    """Helper function for <Results.plot_enhancement_metrics()>.
    
    Parameters
    ----------
    ax : Axes handle
        Axes handle to plot on.
    barWidth : float
        Width of bars for bar plot.
    data : dict of np.ndarrays of floats /or/ dict 
            of np.ndarrays of [3 x 1] lists of floats
        Speech enhancement metric(s) per node.
    """

    numNodes = len(data)  # number of nodes in network

    flagZeroBar = False  # flag for plotting a horizontal line at `metric = 0`

    # Columns count
    baseCount = 2
    if data['Node1'].afterCentr != 0.:
        baseCount += 1
    if data['Node1'].afterLocal != 0.:
        baseCount += 1
    if data['Node1'].afterSSBC != 0:
        baseCount += 1
    widthFact = baseCount + 1
    colShifts = np.arange(
        start=1 - baseCount,
        stop=baseCount,
        step=2
    )

    delta = barWidth / (2 * widthFact)

    # Add grid
    ax.grid()
    ax.set_axisbelow(True)
    
    for idxNode in range(numNodes):
        idxColShift = 0
        if idxNode == 0:    # only add legend labels to first node
            ax.bar(
                idxNode + colShifts[idxColShift] * delta,
                data[f'Node{idxNode + 1}'].before,
                width=barWidth / widthFact,
                color='C0',
                edgecolor='k',
                label='Raw signal'
            )
            idxColShift += 1
            if data['Node1'].afterLocal != 0.:
                ax.bar(
                    idxNode + colShifts[idxColShift] *  delta,
                    data[f'Node{idxNode + 1}'].afterLocal,
                    width=barWidth / widthFact,
                    color='C1',
                    edgecolor='k',
                    label='Local est.'
                )
                idxColShift += 1
            if data['Node1'].afterSSBC != 0.:
                ax.bar(
                    idxNode + colShifts[idxColShift] * delta,
                    data[f'Node{idxNode + 1}'].afterSSBC,
                    width=barWidth / widthFact,
                    color='C4',
                    edgecolor='k',
                    label='SSBC est.'
                )
                idxColShift += 1
            if data['Node1'].afterCentr != 0.:
                ax.bar(
                    idxNode + colShifts[idxColShift] * delta,
                    data[f'Node{idxNode + 1}'].afterCentr,
                    width=barWidth / widthFact,
                    color='C2',
                    edgecolor='k',
                    label='Centr. est.'
                )
                idxColShift += 1
            ax.bar(
                idxNode + colShifts[idxColShift] * delta,
                data[f'Node{idxNode + 1}'].after,
                width=barWidth / widthFact,
                color='C3',
                edgecolor='k',
                label='DANSE est.'
            )
            if data['Node1'].best is not None:
                # Plot best-possible-performance as horizontal bar
                ax.hlines(
                    y=data[f'Node{idxNode + 1}'].best,
                    xmin=idxNode - barWidth/2,
                    xmax=idxNode + barWidth/2,
                    colors='k',
                    linestyles='dashed',
                    label='Best possible'
                )
        else:
            ax.bar(
                idxNode + colShifts[idxColShift] * delta,
                data[f'Node{idxNode + 1}'].before,
                width=barWidth / widthFact,
                color='C0',
                edgecolor='k'
            )
            idxColShift += 1
            if data['Node1'].afterLocal != 0.:
                ax.bar(
                    idxNode + colShifts[idxColShift] * delta,
                    data[f'Node{idxNode + 1}'].afterLocal,
                    width=barWidth / widthFact,
                    color='C1',
                    edgecolor='k',
                )
                idxColShift += 1
            if data['Node1'].afterSSBC != 0.:
                ax.bar(
                    idxNode + colShifts[idxColShift] * delta,
                    data[f'Node{idxNode + 1}'].afterSSBC,
                    width=barWidth / widthFact,
                    color='C4',
                    edgecolor='k',
                )
                idxColShift += 1
            if data['Node1'].afterCentr != 0.:
                ax.bar(
                    idxNode + colShifts[idxColShift] * delta,
                    data[f'Node{idxNode + 1}'].afterCentr,
                    width=barWidth / widthFact,
                    color='C2',
                    edgecolor='k',
                )
                idxColShift += 1
            ax.bar(
                idxNode + colShifts[idxColShift] * delta,
                data[f'Node{idxNode + 1}'].after,
                width=barWidth / widthFact,
                color='C3',
                edgecolor='k'
            )
            if data['Node1'].best is not None:
                # Plot best-possible-performance as horizontal bar
                ax.hlines(
                    y=data[f'Node{idxNode + 1}'].best,
                    xmin=idxNode - barWidth/2,
                    xmax=idxNode + barWidth/2,
                    colors='k',
                    linestyles='dashed',
                )

            # Consider case where the metrics was not computed (e.g., PESQ with
            # SRO-affected sampling frequency).
            if data[f'Node{idxNode + 1}'].after == 0 and\
                data[f'Node{idxNode + 1}'].before == 0 and\
                data[f'Node{idxNode + 1}'].afterCentr == 0 and\
                data[f'Node{idxNode + 1}'].afterLocal == 0 and\
                data[f'Node{idxNode + 1}'].afterSSBC == 0:
                ax.text(
                    idxNode,
                    0.05,
                    s='Not computed',  # write "Not computed" on plot
                    horizontalalignment='center',
                    rotation='vertical'
                )
            

        if data[f'Node{idxNode + 1}'].after < 0 or\
            data[f'Node{idxNode + 1}'].before < 0 or\
            data[f'Node{idxNode + 1}'].afterCentr < 0 or\
            data[f'Node{idxNode + 1}'].afterLocal < 0 or\
            data[f'Node{idxNode + 1}'].afterSSBC < 0:
            flagZeroBar = True
    
    plt.xticks(
        np.arange(numNodes),
        [f'N{ii + 1}' for ii in range(numNodes)],
        fontsize=8
    )
    ax.tick_params(axis='x', labelrotation=90)
    if flagZeroBar:
        ax.hlines(  # plot horizontal line at `metric = 0`
            0,
            - barWidth/2,
            numNodes - 1 + barWidth/2,
            colors='k',
            linestyles='dashed'
        )


def export_sounds(
        out: DANSEoutputs,
        wasn: list[Node],
        folder: str,
        fullyConnected: bool=True
    ):
    """
    Exports the enhanced, noisy, and desired signals as WAV files.

    Parameters
    ----------
    out : `DANSEoutputs` object
        DANSE outputs (signals, etc.)
    wasn : list of `Node` objects
        WASN under consideration.
    folder : str
        Folder where to create the "wav" folder where to export files.
    fullyConnected : bool, optional
        Whether the WASN is fully connected or not. The default is True.
    """

    folderShort = met.shorten_path(folder)
    # Check path validity
    if not Path(f'{folder}/wav').is_dir():
        Path(f'{folder}/wav').mkdir()
        print(f'Created .wav export folder ".../{folderShort}/wav".')

    def _export_to_wav(name, data, fs=int(wasn[0].fs)):
        if name[-4:] != '.wav':
            if '.' in name:
                name = name.split('.')[0]
            name += '.wav'
        wavfile.write(name, fs, normalize_toint16(data))

    for k in range(len(wasn)):
        currFsAsInt = int(wasn[k].fs)  # current sampling frequency as int
        _export_to_wav(
            f'{folder}/wav/noisy_N{k + 1}_Sref{out.referenceSensor + 1}.wav',
            wasn[k].data, fs=currFsAsInt
        )
        _export_to_wav(
            f'{folder}/wav/desired_N{k + 1}_Sref{out.referenceSensor + 1}.wav',
            wasn[k].cleanspeech[:, out.referenceSensor], fs=currFsAsInt
        )
        # vvv if enhancement has been performed
        if len(out.TDdesiredSignals_est[:, k]) > 0:
            _export_to_wav(
                f'{folder}/wav/enhanced_N{k + 1}.wav',
                out.TDdesiredSignals_est[:, k], fs=currFsAsInt
            )
            if out.broadcastType == 'wholeChunk' and fullyConnected and\
                out.simType != 'batch':
                # Export the fused signals too
                if not Path(f'{folder}/wav/fused').is_dir():
                    Path(f'{folder}/wav/fused').mkdir()
                _export_to_wav(
                    f'{folder}/wav/fused/z_N{k + 1}.wav',
                    out.TDfusedSignals[k], fs=currFsAsInt
                )
            elif out.broadcastType != 'wholeChunk':
                print(f'Node {k+1}: Fused signals not exported (not yet implemented for per-sample broadcasting).')
            elif not fullyConnected:
                if not Path(f'{folder}/wav/fused_ti').is_dir():
                    Path(f'{folder}/wav/fused_ti').mkdir()
                _export_to_wav(
                    f'{folder}/wav/fused/etaMk_N{k + 1}.wav',
                    out.TDfusedSignalsTI[k], fs=currFsAsInt
                )
        # vvv if enhancement has been performed and centralised estimate computed
        if out.computeCentralised:
            if len(out.TDdesiredSignals_est_c[:, k]) > 0:
                _export_to_wav(
                    f'{folder}/wav/enhancedCentr_N{k + 1}.wav',
                    out.TDdesiredSignals_est_c[:, k], fs=currFsAsInt
                )
        # vvv if enhancement has been performed and local estimate computed
        if out.computeLocal:
            if len(out.TDdesiredSignals_est_l[:, k]) > 0:
                _export_to_wav(
                    f'{folder}/wav/enhancedLocal_N{k + 1}.wav',
                    out.TDdesiredSignals_est_l[:, k], fs=currFsAsInt
                )
        if out.computeSingleSensorBroadcast:
            if len(out.TDdesiredSignals_est_ssbc[:, k]) > 0:
                _export_to_wav(
                    f'{folder}/wav/enhancedSSBC_N{k + 1}.wav',
                    out.TDdesiredSignals_est_ssbc[:, k], fs=currFsAsInt
                )
    print(f'Signals exported in folder ".../{folderShort}/wav".')


def plot_asc(
    asc: pra.room.ShoeBox,
    p: WASNparameters,
    folder='',
    usedAdjacencyMatrix=np.array([]),
    nodeTypes=[],
    originalAdjacencyMatrix=np.array([]),
    plot3Dview=False,
    ):
    """
    Plots an acoustic scenario.

    Parameters
    ----------
    asc : `pyroomacoustics.room.ShoeBox` object
        Acoustic scenario under consideration.
    p : `WASNparameters` object
        WASN parameters.
    folder : str
        Folder where to export figure (if not `None`).
    usedAdjacencyMatrix : [K x K] np.ndarray (int [or float]: 0 [0.] or 1 [1.])
        Adjacency matrix used in the (TI-)DANSE algorithm.
    nodeTypes : list[str]
        List of node types, in order.
    originalAdjacencyMatrix : [K x K] np.ndarray (int [or float]: 0 [0.] or 1 [1.])
        Adjacency matrix set in the original test parameters
        (e.g., before pruning to a tree topology).
    plot3Dview : bool
        Whether to plot a 3D view of the acoustic scenario.
    """

    def _plot_connections(sensorCoords, stnIdx):
        """Helper function to plot connections based on WASN topology.
        
        Parameters
        ----------
        sensorCoords : [2 x N] np.ndarray (float)
            Sensor coordinates.
        stnIdx : [N] np.ndarray (int)
            Sensor-to-node assignment.
        """
        def __plot_from_mat(mat, coords, ax, color, linestyle):
            """Helper sub-function."""
            # Add topology connectivity lines for the original adjacency matrix
            for k in range(mat.shape[0]):
                for q in range(mat.shape[1]):
                    # Only consider upper triangular matrix without diagonal
                    # (redundant, otherwise)
                    if k > q and mat[k, q] != 0:
                        ax.plot(
                            [coords[0, k], coords[0, q]],
                            [coords[1, k], coords[1, q]],
                            color=color,
                            linestyle=linestyle
                        )
        # Get geometrical central coordinates of each node in current 2D plane
        nodeCoords = np.zeros((2, p.nNodes))
        for k in range(p.nNodes):
            nodeCoords[:, k] = np.mean(
                sensorCoords[:, stnIdx == k],
                axis=1
            )
        # Add topology connectivity lines for the original adjacency matrix
        __plot_from_mat(originalAdjacencyMatrix, nodeCoords, ax, '0.75', ':')
        # Add topology connectivity lines for the effective adjacency matrix
        __plot_from_mat(usedAdjacencyMatrix, nodeCoords, ax, '0.25', '--')

    # Select correct variables from parameters
    if len(p.nSensorPerNodeASC) > 0:
        # Some random-noise (unusable) signals have been added to the nodes
        # for experimental purposes - use the original number of sensors per
        # node instead of the current one for plotting.
        nSensorPerNode = p.nSensorPerNodeASC
    else:
        nSensorPerNode = p.nSensorPerNode
    if len(p.sensorToNodeIndicesASC) > 0:
        # Some random-noise (unusable) signals have been added to the nodes
        # for experimental purposes - use the original sensor-to-node indices
        # instead of the current ones for plotting.
        sensorToNodeIdx = p.sensorToNodeIndicesASC
    else:
        sensorToNodeIdx = p.sensorToNodeIndices

    # Determine appropriate node radius for ASC subplots
    nodeRadius = 0
    for k in range(p.nNodes):
        allIndices = np.arange(sum(nSensorPerNode))
        sensorIndices = allIndices[sensorToNodeIdx == k]
        if len(sensorIndices) > 1:
            meanpos = np.mean(asc.mic_array.R[:, sensorIndices], axis=1)
            curr = np.amax(asc.mic_array.R[:, sensorIndices] - \
                meanpos[:, np.newaxis])
        else:
            curr = 0.025 * np.amin(asc.shoebox_dim[0])
        if curr > nodeRadius:
            nodeRadius = copy.copy(curr)

    fig = plt.figure()
    # Plot 3D view if requested
    if plot3Dview:
        nCols = 3
        fig.set_size_inches(9.5, 3.5)
    else:
        fig.set_size_inches(6.5, 3.5)
        nCols = 2
    ax = fig.add_subplot(1, nCols, 1)
    if p.nNoiseSources > 0:
        noiseSourcePos = np.array([ii.position[:2] for ii in asc.sources[-p.nNoiseSources:]])
    else:
        noiseSourcePos = np.array([])
    plot_side_room(
        ax,
        p.rd[:2],
        np.array([ii.position[:2] for ii in asc.sources[:p.nDesiredSources]]), 
        noiseSourcePos, 
        asc.mic_array.R[:2, :].T,
        sensorToNodeIdx,
        dotted=p.t60==0,
        showLegend=False,
        nodeRadius=nodeRadius,
        nodeTypes=nodeTypes
    )
    if usedAdjacencyMatrix.size != 0:
        # Add topology lines
        _plot_connections(
            sensorCoords=asc.mic_array.R[:2, :],
            stnIdx=sensorToNodeIdx
        )
    ax.set(xlabel='$x$ [m]', ylabel='$y$ [m]', title='Top view')
    #
    ax = fig.add_subplot(1, nCols, 2)
    if p.nNoiseSources > 0:
        noiseSourcePos = np.array([ii.position[-2:] for ii in asc.sources[-p.nNoiseSources:]])
    else:
        noiseSourcePos = np.array([])
    plot_side_room(
        ax,
        p.rd[-2:],
        np.array([ii.position[-2:] for ii in asc.sources[:p.nDesiredSources]]), 
        noiseSourcePos, 
        asc.mic_array.R[-2:, :].T,
        sensorToNodeIdx,
        dotted=p.t60==0,
        showLegend=False,
        nodeRadius=nodeRadius,
        nodeTypes=nodeTypes
    )
    if usedAdjacencyMatrix.size != 0:
        # Add topology lines
        _plot_connections(
            sensorCoords=asc.mic_array.R[-2:, :],
            stnIdx=sensorToNodeIdx
        )
    ax.set(xlabel='$y$ [m]', ylabel='$z$ [m]', title='Side view')

    # Add distance info
    distancesToSources = []
    for ii in range(p.nNodes):
        distancesToSources.append([
            np.mean(np.linalg.norm(
                asc.mic_array.R[:, sensorToNodeIdx == ii].T - s.position
            )) for s in asc.sources
        ])
    distancesToSources = np.array(distancesToSources)
    if p.nNodes * p.nDesiredSources + p.nNodes * p.nNoiseSources < 15:
        boxText = 'Node distances\n\n'
        for ii in range(p.nNodes):
            for jj in range(p.nDesiredSources):
                d = distancesToSources[ii, jj]
                boxText += f'{ii + 1}$\\to$D{jj + 1}={np.round(d, 2)}m\n'
            for jj in range(p.nNoiseSources):
                d = distancesToSources[ii, -(jj + 1)]
                boxText += f'{ii + 1}$\\to$N{jj + 1}={np.round(d, 2)}m\n'
            boxText += '\n'
        boxText = boxText[:-1]
        ax.text(
            x=1.1,
            y=0.9,
            s=boxText,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    else:
        # Too many nodes and/or sources to display distances in box text.
        # Instead, display a table of distances in a separate figure.
        fig2 = plot_distances_as_table(
            distancesToSources,
            p.nDesiredSources,
            p.nNoiseSources
        )
        if folder != '':
            # Make sure folder exists
            if not Path(folder).exists():
                Path(folder).mkdir(parents=True, exist_ok=True)
            # Export
            fig2.savefig(f'{folder}/asc_dists.png', dpi=300)
            fig2.savefig(f'{folder}/asc_dists.pdf')
    #
    if plot3Dview:
        ax = fig.add_subplot(1, nCols, 3, projection='3d')
        plot_asc_3d(ax, asc, p)  # plot room in 3d

    # Make sure everything fits
    plt.tight_layout()

    if folder != '':
        # Make sure folder exists
        if not Path(folder).exists():
            Path(folder).mkdir(parents=True, exist_ok=True)
        # Export
        fig.savefig(f'{folder}/asc.png', dpi=300)
        fig.savefig(f'{folder}/asc.pdf')
        
    return fig


def plot_distances_as_table(dists: np.ndarray, nDesired: int, nNoise: int):
    """
    Plots a table of distances between nodes and sources.
    
    Parameters
    ----------
    dists : [Nnodes x Nsources] np.ndarray of floats
        Distances between each node and each source.
    nDesired : int
        Number of desired sources.
    nNoise : int
        Number of noise sources.
    """
    
    # Create table
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.set_size_inches(5.5 * dists.shape[1] / 9, 1.5 * dists.shape[0] / 3)
    ax.axis('off')
    ax.axis('tight')
    # Create table
    table = ax.table(
        cellText=np.round(dists, 2),
        rowLabels=[f'Node {ii + 1}' for ii in range(dists.shape[0])],
        colLabels=[f'Desired {ii + 1}' for ii in range(nDesired)] + \
            [f'Noise {ii + 1}' for ii in range(nNoise)],
        loc='center'
    )
    # Adjust table properties
    table.auto_set_font_size(True)
    # table.set_fontsize(10)
    table.scale(1, 1.5)
    # Add title
    ax.set_title('Distances between nodes and sources [m]')
    # Make sure everything fits
    plt.tight_layout()
    
    return fig


def plot_side_room(
        ax,
        rd2D,
        rs,
        rn,
        r,
        micToNodeTags,
        scatsize=20,
        dotted=False,
        showLegend=True,
        nodeRadius=None,
        nodeTypes=[]
    ):
    """Plots a 2-D room side, showing the positions of
    sources and nodes inside of it.
    Parameters
    ----------
    ax : Axes handle
        Axes handle to plot on.
    rd2D : [2 x 1] list
        2-D room dimensions [m].
    rs : [Ns x 2] np.ndarray (real)
        Desired (speech) source(s) coordinates [m]. 
    rn : [Nn x 2] np.ndarray (real)
        Noise source(s) coordinates [m]. 
    r : [N x 2] np.ndarray (real)
        Sensor(s) coordinates [m].
    sensorToNodeTags : [N x 1] np.ndarray (int)
        Tags relating each sensor to a node number (>=1).
    scatsize : float
        Scatter plot marker size.
    dotted : bool
        If True, use dotted lines. Else, use solid lines (default).
    showLegend : bool
        If True, show legend.
    nodeRadius : float
        Pre-defined node radius. 
    nodeTypes : list[str]
        List of node types, in order.
    """

    numNodes = len(np.unique(micToNodeTags))
    numSensors = len(micToNodeTags)
    
    plot_room2D(ax, rd2D, dotted)
    # Desired sources
    for m in range(rs.shape[0]):
        ax.scatter(
            rs[m,0],
            rs[m,1],
            s=2*scatsize,
            c='lime',
            marker='d',
            edgecolor='black'
        )
    # Noise sources
    for m in range(rn.shape[0]):
        ax.scatter(
            rn[m,0],
            rn[m,1],
            s=2*scatsize,
            c='red',
            marker='P',
            edgecolor='black'
        )
    # Nodes and sensors
    flagIncludeNodeTypes = 'root' in nodeTypes
    circHandles = []
    leg = []
    for k in range(numNodes):
        allIndices = np.arange(numSensors)
        micsIdx = allIndices[micToNodeTags == k]
        for m in micsIdx:
            ax.scatter(
                r[m,0],
                r[m,1],
                s=scatsize,
                # c=f'C{k}',
                c='black',
                edgecolors='black',
                marker='o'
            )
        # Draw circle around node
        if nodeRadius is not None:
            radius = nodeRadius
        else:
            radius = np.amax(r[micsIdx, :] - np.mean(r[micsIdx, :], axis=0))
        circ = plt.Circle(
            (np.mean(r[micsIdx,0]), np.mean(r[micsIdx,1])),
            radius * 2,
            # color=f'C{k}',
            color='black',
            fill=False
        )
        circHandles.append(circ)
        leg.append(f'Node {k + 1}')
        # Add label
        myText = f'$\\mathbf{{{k+1}}}$'  # node number
        if flagIncludeNodeTypes:
            myText += f' ({nodeTypes[k][0].upper()}.)'  # show node type too
        ax.text(
            np.mean(r[micsIdx, 0]) + 1.5 * radius,
            np.mean(r[micsIdx, 1]) + 1.5 * radius,
            myText,
            # c=f'C{k}'
            c='black',
        )
        ax.add_patch(circ)
    ax.grid()
    ax.set_axisbelow(True)
    ax.axis('equal')
    if showLegend:
        nc = 1  # number of columbs in legend object
        if len(circHandles) >= 4:
            nc = 2
        ax.legend(circHandles, leg, loc='lower right', ncol=nc, mode='expand')


def plot_room2D(ax, rd, dotted=False):
    """Plots the edges of a rectangle in 2D on the axes <ax>
    
    Parameters
    ----------
    ax : matplotlib Axes object
        Axes object onto which the rectangle should be plotted.
    rd : [3 x 1] (or [1 x 3], or [2 x 1], or [1 x 2]) np.ndarray or list of float
        Room dimensions [m].
    dotted : bool
        If true, use dotted lines. Else, use solid lines (default).
    """
    fmt = 'k'
    if dotted:
        fmt += '--'
    ax.plot([rd[0],0], [0,0], fmt)
    ax.plot([0,0], [0,rd[1]], fmt)
    ax.plot([rd[0],rd[0]], [0,rd[1]], fmt)
    ax.plot([0,rd[0]], [rd[1],rd[1]], fmt)


def plot_signals_all_nodes(
        out: DANSEoutputs,
        wasn: list[Node],
        fixedWindow=True
    ):
    """
    Plot DANSE output signals, comparing with inputs.

    Parameters
    ----------
    out : `DANSEoutputs` object
        DANSE run outputs.
    wasn : list of `Node` objects
        WASN under consideration.
    fixedWindow : bool, optional
        Whether to use a fixed window for plotting. If False, use the
        original window used for WOLA processing.
    
    Returns
    -------
    figs : list of `matplotlib.figure.Figure` objects
        Figure handle for each node.
    """
    if fixedWindow:
        win = np.sqrt(np.hanning(1024))
    else:
        win = out.winWOLAanalysis

    figs = []
    # Plot per node
    for k in range(out.nNodes):
        fig, axForTitle = plot_signals(
            node=wasn[k],
            win=win,
            ovlp=out.WOLAovlp,
            batchMode=out.simType == 'batch'
        )
        ti = f'Node {k + 1}, {out.nSensorPerNode[k]} sensor(s)'
        if out.simType == 'online':
            ti += f' (online: $\\beta$={np.round(out.beta[k], 4)})'
        axForTitle.set_title(ti)
        plt.tight_layout()
        figs.append(fig)

    return figs


def plot_signals(node: Node, win, ovlp, batchMode=False):
    """
    Creates a visual representation of the signals at a particular sensor.

    Parameters
    ----------
    node : `Node` object
        Node, containing all needed info (fs, signals, etc.).
    win : np.ndarry (float)
        STFT analysis window.
    ovlp : float
        Consecutive STFT window overlap [/100%].

    Returns
    -------
    fig : `matplotlib.figure.Figure` object
        Figure handle.
    """

    # Waveforms
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,2,1)
    delta = np.amax(np.abs(node.data))
    ax.plot(
        node.timeStamps,
        node.cleanspeechRefSensor,
        '0.5',  # grey
        label='Desired (ref. sensor)'
    )
    ax.plot(
        node.timeStamps,
        node.vad * np.amax(node.cleanspeechRefSensor) * 1.1,
        'k-',
        label='VAD (ref. sensor)'
    )
    ax.plot(
        node.timeStamps,
        node.data[:, 0] - 2*delta,
        'C0-',
        label='Noisy (ref. sensor)',
    )
    ax.plot(
        node.timeStamps,
        node.enhancedData - 4*delta,
        'C3-',
        label='Enhanced (global)'
    )
    currDelta = 4*delta
    if len(node.enhancedData_l) > 0:
        ax.plot(
            node.timeStamps,
            node.enhancedData_l - currDelta - 2*delta,
            'C1-',
            label='Enhanced (local)'
        )
        currDelta += 2*delta
    if len(node.enhancedData_c) > 0:
        ax.plot(
            node.timeStamps,
            node.enhancedData_c - currDelta - 2*delta,
            'C2-',
            label='Enhanced (centr.)'
        )
        currDelta += 2*delta
    if len(node.enhancedData_ssbc) > 0:
        ax.plot(
            node.timeStamps,
            node.enhancedData_ssbc - currDelta - 2*delta,
            'C4-',
            label='Enhanced (SSBC)'
        )
    # Plot start/end of enhancement metrics computations
    ymin, ymax = np.amin(ax.get_ylim()), np.amax(ax.get_ylim())
    ax.vlines(
        x=node.metricStartTime,
        ymin=ymin,
        ymax=ymax,
        colors='0.75',
    )
    ax.vlines(
        x=node.metricEndTime,
        ymin=ymin,
        ymax=ymax,
        colors='0.75',
    )

    ax.set_yticklabels([])
    ax.set(xlabel='$t$ [s]')
    ax.grid()
    plt.legend(loc=(0.01, 0.5), fontsize=8)

    # -------- STFTs --------
    # Number of subplot rows
    nRows = 3

    # Compute STFTs
    cleanSTFT, f, t = get_stft(node.cleanspeechRefSensor, node.fs, win, ovlp)
    noisySTFT, _, _ = get_stft(node.data[:, 0], node.fs, win, ovlp)
    enhanSTFT, _, _ = get_stft(node.enhancedData, node.fs, win, ovlp)
    if len(node.enhancedData_l) > 0:
        enhanSTFT_l, _, _ = get_stft(node.enhancedData_l, node.fs, win, ovlp)
        nRows += 1
    if len(node.enhancedData_c) > 0:
        enhanSTFT_c, _, _ = get_stft(node.enhancedData_c, node.fs, win, ovlp)
        nRows += 1
    if len(node.enhancedData_ssbc) > 0:
        enhanSTFT_ssbc, _, _ = get_stft(node.enhancedData_ssbc, node.fs, win, ovlp)
        nRows += 1
    
    # Get color plot limits
    limLow = 20 * np.log10(
        np.amin([np.amin(np.abs(cleanSTFT)), np.amin(np.abs(noisySTFT))])
    )
    # Ensures that pure silences do not bring the limit too low
    limLow = np.amax([-100, limLow])
    limHigh = 20 * np.log10(
        np.amax([np.amax(np.abs(cleanSTFT)), np.amax(np.abs(noisySTFT))])
    )

    # Plot spectrograms
    ax = fig.add_subplot(nRows,2,2)  # Wet desired signal
    axForTitle = copy.copy(ax)
    data = 20 * np.log10(np.abs(np.squeeze(cleanSTFT)))
    stft_subplot(ax, t, f, data, [limLow, limHigh], 'Desired (ref. sensor)')
    plt.xticks([])
    ax = fig.add_subplot(nRows,2,4)  # Sensor signals
    data = 20 * np.log10(np.abs(np.squeeze(noisySTFT)))
    stft_subplot(ax, t, f, data, [limLow, limHigh], 'Noisy (ref. sensor)')
    plt.xticks([])
    ax = fig.add_subplot(nRows,2,6)   # Enhanced signals (global)
    # Avoid divide-by-zero error
    np.seterr(divide = 'ignore')
    data = 20 * np.log10(np.abs(np.squeeze(enhanSTFT)))
    stft_subplot(ax, t, f, data, [limLow, limHigh], 'Global DANSE')
    currSubplotIdx = 6
    if len(node.enhancedData_l) > 0:    # Enhanced signals (local)
        ax = fig.add_subplot(nRows,2,currSubplotIdx + 2)
        data = 20 * np.log10(np.abs(np.squeeze(enhanSTFT_l)))
        stft_subplot(ax, t, f, data, [limLow, limHigh], 'Local est.')
        currSubplotIdx += 2
    if len(node.enhancedData_c) > 0:    # Enhanced signals (centralised)
        ax = fig.add_subplot(nRows,2,currSubplotIdx + 2)
        data = 20 * np.log10(np.abs(np.squeeze(enhanSTFT_c)))
        stft_subplot(ax, t, f, data, [limLow, limHigh], 'Centr. est.')
        currSubplotIdx += 2
    if len(node.enhancedData_ssbc) > 0:    # Enhanced signals (SSBC)
        ax = fig.add_subplot(nRows,2,currSubplotIdx + 2)
        data = 20 * np.log10(np.abs(np.squeeze(enhanSTFT_ssbc)))
        stft_subplot(ax, t, f, data, [limLow, limHigh], 'SSBC est.')
    ax.set(xlabel='$t$ [s]')
    np.seterr(divide = 'warn')  # reset divide-by-zero error warning

    return fig, axForTitle


def stft_subplot(ax, t, f, data, vlims, label=''):
    """Helper function for <Signals.plot_signals()>."""
    # Text boxes properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #
    mappable = ax.pcolormesh(t, f / 1e3, data, vmin=vlims[0], vmax=vlims[1], shading='auto')
    ax.set(ylabel='$f$ [kHz]')
    if label != '':
        ax.text(0.025, 0.9, label, fontsize=8, transform=ax.transAxes,
            verticalalignment='top', bbox=props)
    ax.yaxis.label.set_size(8)
    cb = plt.colorbar(mappable)
    return cb


# def get_stft(x, fs, win, ovlp):
#     """
#     Derives time-domain signals' STFT representation
#     given certain settings.

#     Parameters
#     ----------
#     x : [N x C] np.ndarray (float)
#         Time-domain signal(s).
#     fs : int
#         Sampling frequency [samples/s].
#     win : np.ndarray[float]
#         Analysis window.
#     ovlp : float
#         Amount of window overlap.

#     Returns
#     -------
#     out : [Nf x Nt x C] np.ndarray (complex)
#         STFT-domain signal(s).
#     f : [Nf x C] np.ndarray (real)
#         STFT frequency bins, per channel (because of different sampling rates).
#     t : [Nt x 1] np.ndarray (real)
#         STFT time frames.
#     """

#     if x.ndim == 1:
#         x = x[:, np.newaxis]

#     for channel in range(x.shape[-1]):

#         fcurr, t, tmp = sig.stft(
#             x[:, channel],
#             fs=fs,
#             window=win,
#             nperseg=len(win),
#             noverlap=int(ovlp * len(win)),
#             return_onesided=True
#         )
#         if channel == 0:
#             out = np.zeros(
#                 (tmp.shape[0], tmp.shape[1], x.shape[-1]), dtype=complex
#             )
#             f = np.zeros((tmp.shape[0], x.shape[-1]))
#         out[:, :, channel] = tmp
#         f[:, channel] = fcurr

#     # Flatten array in case of single-channel data
#     if x.shape[-1] == 1:
#         f = np.array([i[0] for i in f])

#     return out, f, t


def normalize_toint16(nparray):
    """Normalizes a NumPy array to integer 16.
    Parameters
    ----------
    nparray : np.ndarray
        Input array to be normalized.

    Returns
    ----------
    nparrayNormalized : np.ndarray
        Normalized array.
    """
    amplitude = np.iinfo(np.int16).max
    nparrayNormalized = (amplitude * nparray / \
        np.amax(nparray) * 0.5).astype(np.int16)  # 0.5 to avoid clipping
    return nparrayNormalized


def plot_cond_numbers(condNumbers: ConditionNumbers, nSensorPerNode: int=None):
    """
    Plot condition numbers.

    Parameters
    ----------
    condNumbers : ConditionNumbers dataclass
        Condition numbers.
    nSensorPerNode : [K x 1] list[int]
        Number of sensors per node (`K` nodes in WASN).

    Returns
    ----------
    fig : matplotlib.figure.Figure
        Figure handle.
    """

    # Get number of nodes in WASN
    nNodes = len(condNumbers.iter_cn_RyyDANSE)

    # Gather condition numbers data and iteration numbers
    nPlots = 1
    dataCns = [condNumbers.cn_RyyDANSE]
    dataIter = [condNumbers.iter_cn_RyyDANSE]
    refs = ['Ryy DANSE ($\\tilde{\mathbf{R}}_{\mathbf{y}_k\mathbf{y}_k}$)']
    if condNumbers.cn_RyyLocalComputed:
        dataCns.append(condNumbers.cn_RyyLocal)
        dataIter.append(condNumbers.iter_cn_RyyLocal)
        nPlots += 1
        refs.append('Ryy local ($\\mathbf{{R}}_{\mathbf{y}_k\mathbf{y}_k}$)')
    if condNumbers.cn_RyyCentrComputed:
        dataCns.append(condNumbers.cn_RyyCentr)
        dataIter.append(condNumbers.iter_cn_RyyCentr)
        refs.append('Ryy centralized ($\\mathbf{{R}}_{\mathbf{y}\mathbf{y}}$)')
        nPlots += 1

    # Plot condition numbers
    scaleFact = 7.5
    fig, ax = plt.subplots(nNodes,
    nPlots, figsize=(
        np.amin((scaleFact * nPlots, 20)),
        np.amin((scaleFact * 0.5 * nNodes, 20))
    ))
    if nPlots == 1:
        ax = ax[:, np.newaxis]
    # Compute maximum colorbar value
    maxCn = np.log10(np.amax([np.amax(np.abs(cn)) for cn in dataCns]))
    for ii, currCNs in enumerate(dataCns):
        for k in range(len(currCNs)):
            mapp = ax[k, ii].pcolormesh(np.log10(currCNs[k]), vmin=0, vmax=maxCn)
            # Format axes
            ti = f'{refs[ii]}, node $k={k + 1}$'
            if nSensorPerNode is not None:
                ti += f' ({nSensorPerNode[k]} sensors)'
            ax[k, ii].set(
                xlabel='STFT time frame index $i$',
                ylabel='Frequency bin index $\\nu$',
                title=ti,
                xticks=ax[k, ii].get_xticks(),
                xticklabels=np.round(np.array(dataIter[ii][k])[np.linspace(
                    start=0,
                    stop=len(dataIter[ii][k]) - 1,
                    num=len(ax[k, ii].get_xticks()),
                    dtype=int
                )], 2)
            )
            if ii != 0:
                ax[k, ii].set(ylabel='')
                ax[k, ii].set(yticklabels=[])
            if k != nNodes - 1:
                ax[k, ii].set(xticklabels=[])
                ax[k, ii].set(xlabel='')
            # Color bar
            cb = plt.colorbar(mapp, ax=ax[k, ii])
            cb.set_ticks(cb.get_ticks()[:-1])
            cb.set_ticklabels([f'$10^{int(tick)}$' for tick in cb.get_ticks()])
            if ii == nPlots - 1:
                cb.set_label('CN (log-scale)')
            ax[k, ii].autoscale(enable=True, axis='xy', tight=True)
    fig.tight_layout()
    
    return fig


def plot_filters(
        filters,
        filtersEXT,
        filtersCentr=None,
        nSensorsPerNode=None,
        refSensorIdx=0,
        figPrefix='filters',
        bestPerfData=None,
        tiDANSEflag=False
    ) -> dict:
    """
    Plot filters.

    Parameters
    ----------
    filters : [K x 1] list of [Nf x Nt x J] np.ndarray[real]
        DANSE filters per node.
        Can be the norm, real part, or imaginary part.
        Can also be the phase or magnitude of the filters.
        `K` : Number of nodes.
        `Nf` : Number of frequency bins.
        `Nt` : Number of time frames.
        `J` : Filter dimensions.
    filtersEXT : [K x 1] list of [Nf x Nt x J] np.ndarray[real]
        External DANSE filters per node, applied for fusion of
        local signals in the broadcasting stage.
        Can be the norm, real part, or imaginary part.
        Can also be the phase or magnitude of the filters.
    filtersCentr : [K x 1] list of [Nf x Nt x J] np.ndarray[real]
        Centralized filters per node.
        Can be the norm, real part, or imaginary part.
        Can also be the phase or magnitude of the filters.
    nSensorPerNode : [K x 1] list[int]
        Number of sensors per node.
    refSensorIdx : int
        Index of reference sensor (same for all nodes).
    figPrefix : str
        Prefix for figure name.
    bestPerfData : dict
        Dictionary containing best performance data.
    tiDANSEflag : bool
        If True, the inputs are related to TI-DANSE, not just DANSE.

    Returns
    ----------
    `figs` : dict of matplotlib.figure.Figure and strings
        Figure handle and export name.
    dataFigs : list[np.ndarray]
        List of data corresponding to each figure in `figs`.
    """
    # Useful sensor to node index reference
    sensorToNodeIndices = [[ii for _ in range(Mk)]
                        for ii, Mk in enumerate(nSensorsPerNode)]
    # Flatten
    sensorToNodeIndices = [i for s in sensorToNodeIndices for i in s]

    # Compute $\mathbf{w}_k^i$ (network-wide DANSE filters, for all `k`)
    nwDANSEfilts_allNodes, legNW_allNodes = compute_netwide_danse_filts(
        filters,
        filtersEXT,
        nSensorsPerNode,
        tiDANSEflag=tiDANSEflag
    )
    
    if tiDANSEflag:
        maxNorm, minNorm = None, None  # TODO: implement network-wide filter computation for TI-DANSE
        figs = []
    else:
        # Determine plot limits
        np.seterr(divide = 'ignore')    # avoid annoying warnings
        l = [np.log10(filt)\
            for filt in nwDANSEfilts_allNodes]
        maxNorm1 = np.nanmax([np.nanmax(ll[np.isfinite(ll)]) for ll in l])   # avoid NaNs and inf's
        minNorm1 = np.nanmin([np.nanmin(ll[np.isfinite(ll)]) for ll in l])   # avoid NaNs and inf's
        if filtersCentr is not None:
            l = [np.log10(np.mean(filt, axis=0))\
                for filt in filters + filtersCentr]  # concatenate `filters` and `filtersCentr`
        else:
            l = [np.log10(np.mean(filt, axis=0))\
                for filt in filters]
        np.seterr(divide = 'warn')      # reset warnings
        maxNorm2 = np.nanmax([np.nanmax(ll[np.isfinite(ll)]) for ll in l])   # avoid NaNs and inf's
        minNorm2 = np.nanmin([np.nanmin(ll[np.isfinite(ll)]) for ll in l])   # avoid NaNs and inf's
        maxNorm = np.amax([maxNorm1, maxNorm2])
        minNorm = np.amin([minNorm1, minNorm2])

        # Plot network-wide (TI-)DANSE filters
        figs = plot_netwide_danse_filts(
            nwDANSEfilts_allNodes,
            legNW_allNodes,
            [maxNorm, minNorm],
            figPrefix,
            bestPerfData
        )

    # Plot filter norms for regular (TI-)DANSE filters
    figs2, dataFigs = plot_danse_filts(
        filters,
        nSensorsPerNode,
        [maxNorm, minNorm],
        figPrefix,
        tiDANSEflag=tiDANSEflag
    )
    figs += figs2

    if filtersCentr is not None:
        # Plot filter norms for centralized filters
        figs3, dataFigs_c = plot_centr_filts(
            filtersCentr,
            nSensorsPerNode,
            [maxNorm, minNorm],
            refSensorIdx,
            figPrefix,
            bestPerfData,
            tiDANSEflag=tiDANSEflag
        )
        figs += figs3
        dataFigs += dataFigs_c

    # Transform to dict
    figs = dict(figs)
    
    return figs, dataFigs


def format_axes_filt_plot(ax, ti, maxNorm=None, minNorm=None):
    """Format axes for filter coefficients plots."""
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set(
        xlabel='STFT time frame index $i$',
        ylabel='$\\log_{{10}}(E_{{\\nu}}\\{|w_{{k,m}}[\\nu, i]|\\})$',
        title=ti,
    )
    if maxNorm is not None and minNorm is not None:
        ax.set_ylim([minNorm, maxNorm])
    ax.legend(loc='lower left')
    ax.grid(True)


def plot_netwide_danse_filts(
        nwDANSEfilts_allNodes,
        legNW_allNodes,
        nSensorsPerNode,
        maxminNorm,
        figPrefix='netwide_filters',
        bestPerfData=None
    ):
    """Plot network-wide (TI-)DANSE filters."""
    # Get number of nodes in WASN
    nNodes = len(nwDANSEfilts_allNodes)
    # Initialize main lists
    figs = []
    for k in range(nNodes):
        nodeCount = 0
        idxCurrNodeSensor = 0
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        for m in range(nwDANSEfilts_allNodes[k].shape[1]):
            ax.plot(
                np.log10(nwDANSEfilts_allNodes[k][:, m]),
                f'C{nodeCount}-',
                label=legNW_allNodes[k][m],
                alpha=1 / nSensorsPerNode[nodeCount] * (idxCurrNodeSensor + 1),
            )

            if bestPerfData is not None:
                # Add horizontal bar to show best perf coefficients
                if 'real' in figPrefix:
                    data = np.real(bestPerfData['wCentr'][k][:, 1, m])
                elif 'imag' in figPrefix:
                    data = np.imag(bestPerfData['wCentr'][k][:, 1, m])
                elif 'norm' in figPrefix:
                    data = np.abs(bestPerfData['wCentr'][k][:, 1, m])
                ax.hlines(
                    y=np.log10(np.mean(data, axis=0)),
                    xmin=0,
                    xmax=nwDANSEfilts_allNodes[k].shape[0] - 1,
                    colors=f'C{nodeCount}',
                    linestyles='dashed',
                    alpha=1 / nSensorsPerNode[nodeCount] * (idxCurrNodeSensor + 1),
                )
            # Increment node count
            if m == np.sum(nSensorsPerNode[:nodeCount + 1]) - 1:
                nodeCount += 1
                idxCurrNodeSensor = 0  # Reset sensor count
            else:
                idxCurrNodeSensor += 1  # Increment sensor count
        ti = f'Network-wide DANSE filters at node $k={k + 1}$'
        if nSensorsPerNode is not None:
            ti += f' ({nSensorsPerNode[k]} sensors)'
        format_axes_filt_plot(ax, ti, maxminNorm[0], maxminNorm[1])
        fig.tight_layout()
        figs.append((f'{figPrefix}_n{k + 1}_net', fig))
        plt.close(fig=fig)

    return figs


def plot_danse_filts(
        filters,
        nSensorsPerNode,
        maxminNorm,
        figPrefix='filters',
        tiDANSEflag=False
    ):
    """Plot regular (TI-)DANSE filters (i.e., not network-wide)."""
    # Get number of nodes in WASN
    nNodes = len(filters)
    # Initialize main lists
    figs = []
    dataFigs = []
    # Plot filter norms for regular (TI-)DANSE filters
    for k in range(nNodes):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        neighborIndices = [ii for ii in np.arange(nNodes) if ii != k]
        neighborCount = 0
        dataPlot = np.zeros_like(filters[k][0, :, :], dtype=float)
        for m in range(filters[k].shape[2]):
            # Get label for legend
            lab = f'$m={m + 1}$'
            if m < nSensorsPerNode[k]:
                lab += ' (local)'
            else:
                if tiDANSEflag:
                    lab += f' ($\\eta_{{-{k + 1}}}$)'
                else:
                    lab += f' (Node $k={neighborIndices[neighborCount] + 1}$)'
                    neighborCount += 1
            # Mean over frequency bins
            np.seterr(divide = 'ignore')   # avoid annoying warnings
            dataPlot[:, m] = np.log10(
                np.mean(filters[k][:, :, m], axis=0)
            )
            np.seterr(divide = 'warn')     # reset warnings
            ax.plot(
                dataPlot[:, m],
                label=lab,
                linestyle='dashed' if m < nSensorsPerNode[k] else 'solid'
            )
        # Set title
        ti = f'DANSE filters at node $k={k + 1}$'
        if nSensorsPerNode is not None:
            ti += f' ({nSensorsPerNode[k]} sensors)'
        # Format axes
        format_axes_filt_plot(ax, ti, maxminNorm[0], maxminNorm[1])
        fig.tight_layout()
        figs.append((f'{figPrefix}_n{k + 1}', fig))
        plt.close(fig=fig)
        dataFigs.append(dataPlot)  # Save data for later use

    return figs, dataFigs


def plot_centr_filts(
        filtersCentr,
        nSensorsPerNode,
        maxminNorm,
        refSensorIdx,
        figPrefix='filters',
        bestPerfData=None,
        tiDANSEflag=False
    ):
    # Get number of nodes in WASN
    nNodes = len(filtersCentr)
    # Initialize main lists
    figs = []
    dataFigs = []
    # Plot filter norms for ``centralized'' (== no-fusion DANSE) filters
    labelsCentr = [[] for _ in range(nNodes)]
    for k in range(nNodes):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        nodeCount = 0
        idxCurrNodeSensor = 0
        dataFig = np.zeros_like(filtersCentr[k][0, :, :], dtype=float)
        for m in range(filtersCentr[k].shape[2]):        
            # Get label for legend
            lab = f'$m={m + 1}$, Node {nodeCount + 1}'
            if m == np.sum(nSensorsPerNode[:k]) + refSensorIdx:
                lab += ' (reference)'
            # Mean over frequency bins
            np.seterr(divide = 'ignore')   # avoid annoying warnings
            dataFig[:, m] = np.log10(
                np.mean(filtersCentr[k][:, :, m], axis=0)
            )
            np.seterr(divide = 'warn')     # reset warnings
            ax.plot(
                dataFig[:, m],
                f'C{nodeCount}-',
                alpha=1 / nSensorsPerNode[nodeCount] * (idxCurrNodeSensor + 1),
                label=lab,
            )
            labelsCentr[k].append(lab)
            if bestPerfData is not None:
                # Add horizontal bar to show best perf coefficients
                if 'real' in figPrefix:
                    data = np.real(bestPerfData['wCentr'][k][:, 1, m])
                elif 'imag' in figPrefix:
                    data = np.imag(bestPerfData['wCentr'][k][:, 1, m])
                elif 'norm' in figPrefix:
                    data = np.abs(bestPerfData['wCentr'][k][:, 1, m])
                ax.hlines(
                    y=np.log10(np.mean(data, axis=0)),
                    xmin=0,
                    xmax=filtersCentr[k].shape[1] - 1,
                    colors=f'C{nodeCount}',
                    linestyles='dashed',
                    alpha=1 / nSensorsPerNode[nodeCount] * (idxCurrNodeSensor + 1),
                )
            # Increment node count
            if m == np.sum(nSensorsPerNode[:nodeCount + 1]) - 1:
                nodeCount += 1
                idxCurrNodeSensor = 0  # Reset sensor count
            else:
                idxCurrNodeSensor += 1  # Increment sensor count
        # Set title
        ti = f'Centralized filters at node $k={k + 1}$'
        # Format axes
        format_axes_filt_plot(ax, ti, maxminNorm[0], maxminNorm[1])
        fig.tight_layout()
        figs.append((f'{figPrefix}_c{k + 1}', fig))
        plt.close(fig=fig)
        dataFigs.append(dataFig)  # Save data for later use

    return figs, dataFigs


def compute_netwide_danse_filts(
        filters,
        filtersEXT,
        nSensorsPerNode,
        tiDANSEflag=False
    ):
    """Compute network-wide DANSE filters."""
    nNodes = len(filters)
    nwDANSEfilts_allNodes = []
    legNW_allNodes = []
    for k in range(nNodes):
        neighborCount = 0
        legendNetwide = []
        netwideDANSEfilts = np.zeros((filters[k].shape[1], 0))

        if tiDANSEflag:
            print('TI-DANSE network-wide filters computation not implemented yet.')
            return None, None 
        else:
            for q in range(nNodes):
                if q == k:
                    currVal = np.mean(
                        filters[k][:, :, :nSensorsPerNode[k]],
                        axis=0
                    )
                    for m in range(nSensorsPerNode[k]):
                        legendNetwide.append(f'$w_{{kk,{m + 1}}}$ (local)')
                else:
                    idxGkq = nSensorsPerNode[k] + neighborCount
                    currVal = np.mean(
                        filtersEXT[q][:, 1:, :] *\
                            filters[k][:, :-1, [idxGkq]],
                        axis=0
                    )
                    # ^^^ NB: we multiply g_kq^i by the (i-1)-th fusion vectors
                    currVal = np.concatenate(
                        (np.zeros((1, currVal.shape[1])), currVal),
                        axis=0
                    )
                    for m in range(nSensorsPerNode[q]):
                        legendNetwide.append(f'$w_{{qq,{m + 1}}}\\cdot g_{{kq}}$ ($q={q + 1}$)')
                    neighborCount += 1
                netwideDANSEfilts = np.concatenate(
                    (netwideDANSEfilts, currVal),
                    axis=1
                )
        
        nwDANSEfilts_allNodes.append(netwideDANSEfilts)
        legNW_allNodes.append(legendNetwide)
    
    return nwDANSEfilts_allNodes, legNW_allNodes


def export_danse_outputs(
        out: DANSEoutputs,
        wasnObj: WASN,
        p: TestParameters,
        room: pra.room.ShoeBox=None,
    ):
    """
    Post-processing after a DANSE run.

    Parameters
    ----------
    out : `danse.danse_toolbox.d_post.DANSEoutputs` object
        DANSE outputs (signals, etc.)
    wasnObj : `WASN` object
        WASN under consideration, after DANSE processing.
    room : `pyroomacoustics.room.ShoeBox` object
        Acoustic scenario under consideration.
    p : `TestParameters` object
        Test parameters.

    Returns
    -------
    out : `danse.danse_toolbox.d_post.DANSEoutputs` object
        DANSE outputs (signals, etc.), after post-processing.
    """
    
    if not p.exportParams.bypassAllExports:

        # If batch mode, export MMSE cost
        if p.danseParams.simType == 'batch':
            fig = out.plot_mmse_cost()
            fig.savefig(f'{p.exportParams.exportFolder}/mmse_cost.png', dpi=300)
            fig.savefig(f'{p.exportParams.exportFolder}/mmse_cost.pdf')
            plt.close(fig)
        elif p.danseParams.simType == 'online' and p.exportParams.mseBatchPerfPlot:
            fig = out.plot_batch_cost_at_each_update()
            fig.savefig(f'{p.exportParams.exportFolder}/batch_cost.png', dpi=300)
            fig.savefig(f'{p.exportParams.exportFolder}/batch_cost.pdf')

        # Export filter coefficients
        if p.exportParams.filters:
            fullPath = f'{p.exportParams.exportFolder}/filters.pkl.gz'
            pickle.dump(out.filters, gzip.open(fullPath, 'wb'))
            if p.danseParams.computeCentralised:
                fullPath = f'{p.exportParams.exportFolder}/filtersCentr.pkl.gz'
                pickle.dump(out.filtersCentr, gzip.open(fullPath, 'wb'))
            print('Exported complex filters.')
            
        # Export filter coefficients norm plot
        if p.exportParams.filterNormsPlot:
            print('Exporting filter norms plot...')
            out.plot_filter_evol(
                p.exportParams.exportFolder,
                exportNormsAsPickle=p.exportParams.filterNorms,  # boolean to export filter norms as pickle
                plots=['norm'],
                tiDANSEflag=not p.is_fully_connected_wasn()
            )
            print('Done.')

        # Export condition number plot
        if p.exportParams.conditionNumberPlot\
            and p.danseParams.simType == 'online':
            out.plot_cond(p.exportParams.exportFolder)

        # Export convergence plot
        if p.exportParams.convergencePlot and p.danseParams.computeCentralised:
            print('Not plotting convergence plot -- needs to be improved first [TODO]')
            # out.plot_convergence(p.exportParams.exportFolder)  # TODO:

        # Export .wav files
        if p.exportParams.wavFiles:
            out.export_sounds(
                wasnObj.wasn,
                p.exportParams.exportFolder,
                p.is_fully_connected_wasn()
            )

        # Plot (+ export) acoustic scenario (WASN)
        if p.exportParams.acousticScenarioPlot and room is not None:
            plot_asc(
                asc=room,
                p=p.wasnParams,
                folder=p.exportParams.exportFolder,
                usedAdjacencyMatrix=wasnObj.adjacencyMatrix,
                nodeTypes=[node.nodeType for node in wasnObj.wasn],
                originalAdjacencyMatrix=p.wasnParams.topologyParams.userDefinedTopo
            )

        # Plot SRO estimation performance
        if p.exportParams.sroEstimPerfPlot:
            fig = out.plot_sro_perf(
                Ns=p.danseParams.Ns,
                fs=p.wasnParams.fs,
                xaxistype='both'  # "both" == iterations & instants
            )
            fig.savefig(f'{p.exportParams.exportFolder}/sroEvolution.png', dpi=300)
            fig.savefig(f'{p.exportParams.exportFolder}/sroEvolution.pdf')

    # Compute performance metrics (+ export if needed)
    if p.exportParams.metricsPlot:
        if p.exportParams.bypassAllExports:
            exportFolder = None
        else:
            exportFolder = p.exportParams.exportFolder
            out.plot_perf(
                wasnObj.wasn,
                exportFolder,
                p.exportParams.metricsInPlots,
                snrYlimMax=p.snrYlimMax,
            )

    if not p.exportParams.bypassAllExports:
        # Plot signals at specific nodes (+ export)
        if p.exportParams.waveformsAndSpectrograms:
            out.plot_sigs(wasnObj.wasn, p.exportParams.exportFolder)

        # Save `DANSEoutputs` object after metrics computation
        if p.exportParams.danseOutputsFile:
            out.save(foldername=p.exportParams.exportFolder, light=True)
        if p.exportParams.metricsFile:
            # Save just metrics (for faster loading in post-processing scripts)
            out.save_metrics(foldername=p.exportParams.exportFolder)
        # Save `TestParameters` object
        if p.exportParams.parametersFile:
            if p.loadedFromYaml:
                p.save_yaml()   # save `TestParameters` object as YAML file
            else:
                p.save()    # save `TestParameters` object as Pickle archive

    return out