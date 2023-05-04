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
from danse_toolbox.d_base import DANSEparameters, get_stft
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
        self.TDfiltSpeech_c = None
        self.STFTfiltSpeech_c = None
        self.TDfiltNoise_c = None
        self.STFTfiltNoise_c = None
        self.TDfiltSpeech_l = None
        self.STFTfiltSpeech_l = None
        self.TDfiltNoise_l = None
        self.STFTfiltNoise_l = None

        # Original microphone signals
        self.micSignals = dv.yin
        # DANSE desired signal estimates
        self.TDdesiredSignals_est = dv.d
        self.STFTDdesiredSignals_est = dv.dhat
        self.TDfiltSpeech = dv.d_s
        self.STFTfiltSpeech = dv.dhat_s
        self.TDfiltNoise = dv.d_n
        self.STFTfiltNoise = dv.dhat_n
        if self.computeCentralised:
            # Centralised desired signal estimates
            self.TDdesiredSignals_est_c = dv.dCentr
            self.STFTDdesiredSignals_est_c = dv.dHatCentr
            self.TDfiltSpeech_c = dv.dCentr_s
            self.STFTfiltSpeech_c = dv.dHatCentr_s
            self.TDfiltNoise_c = dv.dCentr_n
            self.STFTfiltNoise_c = dv.dHatCentr_n
        if self.computeLocal:
            # Local desired signal estimates
            self.TDdesiredSignals_est_l = dv.dLocal
            self.STFTDdesiredSignals_est_l = dv.dHatLocal
            self.TDfiltSpeech_l = dv.dLocal_s
            self.STFTfiltSpeech_l = dv.dHatLocal_s
            self.TDfiltNoise_l = dv.dLocal_n
            self.STFTfiltNoise_l = dv.dHatLocal_n
        # SROs
        self.SROgroundTruth = dv.SROsppm
        self.SROsEstimates = dv.SROsEstimates
        self.SROsResiduals = dv.SROsResiduals
        self.flagIterations = dv.flagIterations
        self.firstUpRefSensor = dv.firstDANSEupdateRefSensor
        # Filters
        self.filters = dv.wTilde
        if self.computeCentralised:
            self.filtersCentr = dv.wCentr
        else:
            self.filtersCentr = None
        # Condition numbers
        if self.saveConditionNumber:
            self.condNumbers = dv.condNumbers
        # Other useful things
        self.beta = dv.expAvgBeta
        self.vadFrames = dv.oVADframes

        # Show initialised status
        self.initialised = True

        return self

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

    def load(self, foldername, dataType='pkl'):
        """Loads dataclass to Pickle archive in folder `foldername`."""
        return met.load(self, foldername, silent=True, dataType=dataType)

    def export_sounds(self, wasn, exportFolder):
        self.check_init()  # check if object is correctly initialised
        export_sounds(self, wasn, exportFolder)

    def plot_filter_norms(self, exportFolder=None, exportNormsAsPickle=False):
        """Plots a visualization of the evolution of filter norms in DANSE."""
        self.check_init()  # check if object is correctly initialised
        figs, dataFigs = plot_filter_norms(
            self.filters,
            self.filtersCentr,
            self.nSensorPerNode,
            refSensorIdx=self.referenceSensor
        )
        # Export figures
        if exportFolder is not None:
            for title, fig in figs.items():
                fullExportFolder = f'{exportFolder}/filtNorms'
                if not os.path.exists(fullExportFolder):
                    os.makedirs(fullExportFolder)
                fig.savefig(f'{fullExportFolder}/{title}.png', dpi=300)
                fig.savefig(f'{fullExportFolder}/{title}.pdf')
        else:
            plt.close(fig)
        # Export data
        if exportNormsAsPickle:
            fullExportFolder = f'{exportFolder}/filtNorms'
            if not os.path.exists(fullExportFolder):
                os.makedirs(fullExportFolder)
            with open(f'{fullExportFolder}/filtNorms.pkl', 'wb') as f:
                pickle.dump(dataFigs, f)

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
            onlySNRandESTOIinPlots=False,
            snrYlimMax=None
        ):
        """Plots DANSE performance."""
        self.check_init()  # check if object is correctly initialised
        self.metrics = compute_metrics(self, wasn)
        figStatic, figDynamic = plot_metrics(
            self,
            onlySNRandESTOIinPlots,
            snrYlimMax
        )
        if exportFolder is not None:
            figStatic.savefig(f'{exportFolder}/metrics.png')
            figStatic.savefig(f'{exportFolder}/metrics.pdf')
            if figDynamic is not None:
                figDynamic.savefig(f'{exportFolder}/metrics_dyn.png')
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
                fig.savefig(f'{fullExportFolder}/converg_node{k+1}.png')
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

        fig = plt.figure(figsize=(6,2))
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
                figs[k].savefig(f'{exportFolder}/sigs_node{k+1}.png')


@dataclass
class BatchDANSEoutputs(DANSEparameters):
    """
    Dataclass to assemble all useful outputs
    of the batch DANSE algorithm.
    """
    initialised : bool = False
    # ^^^ turns to True when `self.from_variables()` is called
    
    def import_params(self, p: DANSEparameters):
        self.__dict__.update(p.__dict__)
    
    def from_variables(self, bdv: BatchDANSEvariables):
        """
        Selects useful output values from `BatchDANSEvariables` object
        after DANSE processing.
        """

        self.filters: list[np.ndarray] = bdv.wTilde
        if self.computeCentralised:
            self.filtersCentr: list[np.ndarray] = bdv.wCentr
        if self.computeLocal:
            self.filtersLocal: list[np.ndarray] = bdv.wLocal

        self.TDdesiredSignals_est = np.array(bdv.dHat)
        self.TDdesiredSignals_est_c = np.array(bdv.dHatCentr)
        self.TDdesiredSignals_est_l = np.array(bdv.dHatLocal)
        self.TDdesiredSignals = np.array([x[:, np.newaxis] for x in bdv.d])
        self.noisySigs = np.array([
            bdv.yLocal[k][:, self.referenceSensor] for k in range(self.nNodes)
        ]).T

        self.fs: list[float] = bdv.fs

        # Show initialised status
        self.initialised = True

        return self

    def check_init(self):
        """Check if object is correctly initialised."""
        if not self.initialised:
            return ValueError('The DANSEoutputs object is empty.')
        
    def export_sounds(self, wasn, exportFolder):
        """Export sounds to .wav files."""
        self.check_init()  # check if object is correctly initialised
        export_sounds(self, wasn, exportFolder)
        
    def plot_filters_evol(self):
        """Plot filters evolution through time, at each node in the network."""
        self.check_init()
        
        def _plot_single_set_filters(ax, f, ls, lab):
            """
            Plots a single set of filters.

            Parameters
            ----------
            ax : matplotlib.axes.Axes
                Axes on which to plot.
            f : np.ndarray
                Filters to plot.
            ls : str
                Line style for plot.
            lab : str
                Label for plot legend.
            """
            for m in range(f.shape[0]):
                ax.plot(
                    f[m, :],
                    f'{ls}C{m}',
                    label=f'{lab} coeff. {m + 1}/{f.shape[0]}'
                )

        nNodes = len(self.filters)
        figs = []
        # Get yaxis limits
        ymin = np.amin([np.amin(f) for f in self.filters])
        ymax = np.amax([np.amax(f) for f in self.filters])
        ymin = ymin - 0.1 * (ymax - ymin)  # 10% margin
        ymax = ymax + 0.1 * (ymax - ymin)  # 10% margin
        for k in range(nNodes):
            fig, axes = plt.subplots(1,1)
            fig.set_size_inches(8.5, 3.5)
            _plot_single_set_filters(axes, self.filters[k], '.-', 'DANSE')
            if self.computeCentralised:
                _plot_single_set_filters(axes, self.filtersCentr[k], '--', 'Centr.')
            # _plot_single_set_filters(axes, self.filtersLocal[k], ':', 'Local')
            axes.set_title(f'Filters at node {k + 1}')
            axes.set_xlabel('DANSE iteration $i$')
            axes.set_ylabel('Coefficients')
            axes.set_xticks(axes.get_xticks())
            axes.set_xticklabels([int(i) for i in axes.get_xticks()])
            axes.set_xlim([0, len(self.filters[k][0, :])])
            axes.set_ylim([ymin, ymax])
            axes.legend()
            axes.grid()
            fig.tight_layout()
            figs.append(fig)
        return figs
    
    def plot_mmse_perf(self):
        """
        Plot the MMSE cost as a function of DANSE iterations for the batch
        DANSE algorithm, at each node in the network.
        """
        # Compute MMSE cost
        mmse = np.mean(
            np.abs(self.TDdesiredSignals - self.TDdesiredSignals_est) ** 2, axis=1
        )
        mmseCentr = np.mean(
            np.abs(self.TDdesiredSignals - self.TDdesiredSignals_est_c) ** 2, axis=1
        )

        fig, axes = plt.subplots(1,1)
        fig.set_size_inches(8.5, 3.5)
        for m in range(mmse.shape[0]):
            axes.plot(mmse[m, :], f'.-C{m}', label=f'Node {m + 1}')
            axes.plot(mmseCentr[m, :], f'.--C{m}', label=f'Node {m + 1} (centr.)')
        axes.set_xlabel('DANSE iteration $i$')
        axes.set_ylabel('MMSE cost')
        axes.set_xticks(axes.get_xticks())
        axes.set_xticklabels([int(i) for i in axes.get_xticks()])
        axes.set_xlim([0, mmse.shape[1]])
        axes.legend()
        axes.grid()
        plt.tight_layout()

        return fig
    
    def plot_sigs(self, win=np.sqrt(np.hanning(1024)), ovlp=0.5):
        """
        Plot time-domain signals, at each node in the network.

        Parameters
        ----------
        win : np.ndarray
            Windowing function to apply to the signals before plotting.
        ovlp : float
            Overlap between windows.

        Returns
        -------
        figs : list[matplotlib.figure.Figure]
            List of figures, one per node.
        """

        nNodes = len(self.filters)  # Number of nodes in the network
        
        figs = []
        # Plot per node
        for k in range(nNodes):

            # Common parameters (to be unpacked)
            com = (self.fs[k], win, ovlp)
            
            # Get STFTs
            d_k, f, t = get_stft(self.TDdesiredSignals[k, :, -1], *com)
            dHat_k, _, _ = get_stft(self.TDdesiredSignals_est[k, :, -1], *com)
            dHatCentr_k, _, _ = get_stft(self.TDdesiredSignals_est_c[k, :, -1], *com)
            y_k, _, _ = get_stft(self.noisySigs[:, k], *com)

            # Get color bar limits
            limLow = 20 * np.log10(np.amin([
                np.amin(np.abs(d_k)),
                np.amin(np.abs(dHat_k)),
                np.amin(np.abs(dHatCentr_k)),
                np.amin(np.abs(y_k))
            ]))
            limHigh = 20 * np.log10(np.amax([
                np.amax(np.abs(d_k)),
                np.amax(np.abs(dHat_k)),
                np.amax(np.abs(dHatCentr_k)),
                np.amax(np.abs(y_k))
            ]))

            fig, axes = plt.subplots(2, 2)
            fig.set_size_inches(8.5, 3.5)
            stft_subplot(
                axes[0, 0], t, f, 20 * np.log10(np.abs(np.squeeze(y_k))),
                [limLow, limHigh], 'Noisy signal'
            )
            stft_subplot(
                axes[0, 1], t, f, 20 * np.log10(np.abs(np.squeeze(d_k))),
                [limLow, limHigh], 'Desired signal'
            )
            stft_subplot(
                axes[1, 0], t, f, 20 * np.log10(np.abs(np.squeeze(dHat_k))),
                [limLow, limHigh], 'DANSE estimate'
            )
            stft_subplot(
                axes[1, 1], t, f, 20 * np.log10(np.abs(np.squeeze(dHatCentr_k))),
                [limLow, limHigh], 'Centralized estimate'
            )
            fig.tight_layout()
            figs.append(fig) 

        return figs


def compute_metrics(
        out: DANSEoutputs,
        wasn: list[Node]
    ) -> EnhancementMeasures:
    """
    Compute and store evaluation metrics after signal enhancement.

    Parameters
    ----------
    out : `DANSEoutputs` object
        DANSE run outputs.
    wasn : list of `Node` objects
        WASN under consideration.
    folder : str
        Folder where to create the "wav" folder where to export files.
    
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
    snr = _ndict(out.nNodes)  # Unweighted SNR
    fwSNRseg = _ndict(out.nNodes)  # Frequency-weighted segmental SNR
    stoi = _ndict(out.nNodes)  # (extended) Short-Time Objective Intelligibility
    pesq = _ndict(out.nNodes)  # Perceptual Evaluation of Speech Quality
    tStart = time.perf_counter()
    for k in range(out.nNodes):
        # Derive starting sample for metrics computations
        startIdx[k] = int(np.floor(wasn[k].metricStartTime * wasn[k].fs))
        print(f"Node {k+1}: computing speech enhancement metrics from the {startIdx[k] + 1}'th sample on (t_start = {wasn[k].metricStartTime} s --> avoid bias due to initial filters guesses in first iterations)...")
        print(f'Computing signal enhancement evaluation metrics for node {k + 1}/{out.nNodes} (sensor {out.referenceSensor + 1}/{wasn[k].nSensors})...')

        # Compute starting indices for centralised and local estimates
        TDdesiredSignals_est_c, TDdesiredSignals_est_l = None, None
        TDfilteredSpeech_c, TDfilteredSpeech_l = None, None
        TDfilteredNoise_c, TDfilteredNoise_l = None, None
        if out.computeCentralised:
            TDdesiredSignals_est_c = out.TDdesiredSignals_est_c[:, k]
            TDfilteredSpeech_c = out.TDfiltSpeech_c[:, k]
            TDfilteredNoise_c = out.TDfiltNoise_c[:, k]
            print(f"Node {k+1}: computing speech enhancement metrics for CENTRALISED PROCESSING from the {startIdx[k] + 1}'th sample on (t_start = {wasn[k].metricStartTime} s).")
        if out.computeLocal:
            TDdesiredSignals_est_l = out.TDdesiredSignals_est_l[:, k]
            TDfilteredSpeech_l = out.TDfiltSpeech_l[:, k]
            TDfilteredNoise_l = out.TDfiltNoise_l[:, k]
            print(f"Node {k+1}: computing speech enhancement metrics for LOCAL PROCESSING from the {startIdx[k] + 1}'th sample on (t_start = {wasn[k].metricStartTime} s).")

        out0, out1, out2, out3 = get_metrics(
            # Clean speech mixture (desired signal)
            clean=np.squeeze(wasn[k].cleanspeechRefSensor),
            noiseOnly=np.squeeze(wasn[k].cleannoiseRefSensor),
            # Microphone signals
            noisy=wasn[k].data[:, out.referenceSensor],
            filteredSpeech=out.TDfiltSpeech[:, k],
            filteredNoise=out.TDfiltNoise[:, k],
            filteredSpeech_c=TDfilteredSpeech_c,
            filteredNoise_c=TDfilteredNoise_c,
            filteredSpeech_l=TDfilteredSpeech_l,
            filteredNoise_l=TDfilteredNoise_l,
            # DANSE outputs (desired signal estimates)
            enhan=out.TDdesiredSignals_est[:, k],
            enhan_c=TDdesiredSignals_est_c,
            enhan_l=TDdesiredSignals_est_l,
            # Start indices
            startIdx=startIdx[k],
            # Other parameters
            fs=wasn[k].fs,
            vad=wasn[k].vadCombined,
            dynamic=out.dynMetrics,
            gamma=out.gammafwSNRseg,
            fLen=out.frameLenfwSNRseg
        )

        snr[f'Node{k + 1}'] = out0
        fwSNRseg[f'Node{k + 1}'] = out1
        stoi[f'Node{k + 1}'] = out2
        pesq[f'Node{k + 1}'] = out3

    print(f'All signal enhancement evaluation metrics computed in {np.round(time.perf_counter() - tStart, 3)} s.')

    # Group measures into EnhancementMeasures object
    metrics = EnhancementMeasures(
        fwSNRseg=fwSNRseg,
        stoi=stoi,
        pesq=pesq,
        snr=snr,
        startIdx=startIdx
    )

    return metrics

def plot_metrics(
        out: DANSEoutputs,
        onlySNRandESTOIinPlots=False,
        snrYlimMax=None
    ):
    """
    Visualize evaluation metrics.

    Parameters
    ----------
    out : `DANSEoutputs` object
        DANSE outputs.
    onlySNRandESTOIinPlots : bool
        If True, only include the SNR and the eSTOI.
    snrYlimMax : float or int
        If not None, set a particular y-axis limit for the SNR plot.
    """

    # Hard-coded variables
    barWidth = 1

    if onlySNRandESTOIinPlots:
        nCols = 2
        fig1 = plt.figure(figsize=(8,3))
    else:
        nCols = 4
        fig1 = plt.figure(figsize=(12,3))
    
    ax = fig1.add_subplot(1, nCols, 1)   # Unweighted SNR
    metrics_subplot(out.nNodes, ax, barWidth, out.metrics.snr)
    ax.set(title='SNR', ylabel='[dB]')
    if snrYlimMax is not None:
        ax.set_ylim([0, snrYlimMax])
    #
    ax = fig1.add_subplot(1, nCols, 2)   # STOI
    metrics_subplot(out.nNodes, ax, barWidth, out.metrics.stoi)
    ax.set(title='eSTOI')
    ax.set_ylim([0, 1])
    #
    if not onlySNRandESTOIinPlots:
        ax = fig1.add_subplot(1, nCols, 3)   # fwSNRseg
        metrics_subplot(out.nNodes, ax, barWidth, out.metrics.fwSNRseg)
        ax.set(title='fwSNRseg', ylabel='[dB]')
        #
        ax = fig1.add_subplot(1, nCols, 4)   # PESQ
        metrics_subplot(out.nNodes, ax, barWidth, out.metrics.pesq)
        ax.set(title='PESQ')
    #
    ax.legend(bbox_to_anchor=(1, 0), loc="lower left")

    # fig1.suptitle("Speech enhancement metrics")
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
            for nodeRef, value in dynMetric.items():        # loop over nodes
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
        # fig2.suptitle("Dynamic speech enhancement metrics")
    else:
        fig2 = None

    return fig1, fig2


def metrics_subplot(numNodes, ax, barWidth, data):
    """Helper function for <Results.plot_enhancement_metrics()>.
    
    Parameters
    ----------
    numNodes : int
        Number of nodes in network.
    ax : Axes handle
        Axes handle to plot on.
    barWidth : float
        Width of bars for bar plot.
    data : dict of np.ndarrays of floats /or/ dict 
            of np.ndarrays of [3 x 1] lists of floats
        Speech enhancement metric(s) per node.
    """

    flagZeroBar = False  # flag for plotting a horizontal line at `metric = 0`

    # Columns count
    baseCount = 2
    if data['Node1'].afterCentr != 0.:
        baseCount += 1
    if data['Node1'].afterLocal != 0.:
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
                    color='C3',
                    edgecolor='k',
                    label='Local est.'
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
                color='C1',
                edgecolor='k',
                label='DANSE est.'
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
                    color='C3',
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
                color='C1',
                edgecolor='k'
            )

            # Consider case where the metrics was not computed (e.g., PESQ with
            # SRO-affected sampling frequency).
            if data[f'Node{idxNode + 1}'].after == 0 and\
                data[f'Node{idxNode + 1}'].before == 0 and\
                data[f'Node{idxNode + 1}'].afterCentr == 0 and\
                data[f'Node{idxNode + 1}'].afterLocal == 0:
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
            data[f'Node{idxNode + 1}'].afterLocal < 0:
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


def export_sounds(out: DANSEoutputs, wasn: list[Node], folder):
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
    """

    folderShort = met.shorten_path(folder)
    # Check path validity
    if not Path(f'{folder}/wav').is_dir():
        Path(f'{folder}/wav').mkdir()
        print(f'Created .wav export folder ".../{folderShort}/wav".')
    for k in range(len(wasn)):
        data = normalize_toint16(wasn[k].data)
        wavfile.write(
            f'{folder}/wav/noisy_N{k + 1}_Sref{out.referenceSensor + 1}.wav',
            int(wasn[k].fs), data
        )
        #
        data = normalize_toint16(wasn[k].cleanspeech[:, out.referenceSensor])
        wavfile.write(
            f'{folder}/wav/desired_N{k + 1}_Sref{out.referenceSensor + 1}.wav',
            int(wasn[k].fs), data
        )
        # vvv if enhancement has been performed
        if len(out.TDdesiredSignals_est[:, k]) > 0:
            data = normalize_toint16(out.TDdesiredSignals_est[:, k])
            wavfile.write(
                f'{folder}/wav/enhanced_N{k + 1}.wav',
                int(wasn[k].fs), data
            )
        # vvv if enhancement has been performed and centralised estimate computed
        if out.computeCentralised:
            if len(out.TDdesiredSignals_est_c[:, k]) > 0:
                data = normalize_toint16(out.TDdesiredSignals_est_c[:, k])
                wavfile.write(
                    f'{folder}/wav/enhancedCentr_N{k + 1}.wav',
                    int(wasn[k].fs), data
                )
        # vvv if enhancement has been performed and local estimate computed
        if out.computeLocal:
            if len(out.TDdesiredSignals_est_l[:, k]) > 0:
                data = normalize_toint16(out.TDdesiredSignals_est_l[:, k])
                wavfile.write(
                    f'{folder}/wav/enhancedLocal_N{k + 1}.wav',
                    int(wasn[k].fs), data
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
    Plots an acoustic scenario nicely.

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
        sensorToNodeIndices = p.sensorToNodeIndicesASC
    else:
        sensorToNodeIndices = p.sensorToNodeIndices

    # Determine appropriate node radius for ASC subplots
    nodeRadius = 0
    for k in range(p.nNodes):
        allIndices = np.arange(sum(nSensorPerNode))
        sensorIndices = allIndices[sensorToNodeIndices == k]
        if len(sensorIndices) > 1:
            meanpos = np.mean(asc.mic_array.R[:, sensorIndices], axis=1)
            curr = np.amax(asc.mic_array.R[:, sensorIndices] - \
                meanpos[:, np.newaxis])
        else:
            curr = 0.1
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
    plot_side_room(
        ax,
        p.rd[:2],
        np.array([ii.position[:2] for ii in asc.sources[:p.nDesiredSources]]), 
        np.array([ii.position[:2] for ii in asc.sources[-p.nNoiseSources:]]), 
        asc.mic_array.R[:2, :].T,
        sensorToNodeIndices,
        dotted=p.t60==0,
        showLegend=False,
        nodeRadius=nodeRadius,
        nodeTypes=nodeTypes
    )
    if usedAdjacencyMatrix != np.array([]):
        # Add topology lines
        _plot_connections(
            sensorCoords=asc.mic_array.R[:2, :],
            stnIdx=sensorToNodeIndices
        )
    ax.set(xlabel='$x$ [m]', ylabel='$y$ [m]', title='Top view')
    #
    ax = fig.add_subplot(1, nCols, 2)
    plot_side_room(
        ax,
        p.rd[-2:],
        np.array([ii.position[-2:] for ii in asc.sources[:p.nDesiredSources]]), 
        np.array([ii.position[-2:] for ii in asc.sources[-p.nNoiseSources:]]), 
        asc.mic_array.R[-2:, :].T,
        sensorToNodeIndices,
        dotted=p.t60==0,
        showLegend=False,
        nodeRadius=nodeRadius,
        nodeTypes=nodeTypes
    )
    if usedAdjacencyMatrix != np.array([]):
        # Add topology lines
        _plot_connections(
            sensorCoords=asc.mic_array.R[-2:, :],
            stnIdx=sensorToNodeIndices
        )
    ax.set(xlabel='$y$ [m]', ylabel='$z$ [m]', title='Side view')
    #
    if plot3Dview:
        ax = fig.add_subplot(1, nCols, 3, projection='3d')
        plot_asc_3d(ax, asc, p)  # plot room in 3d

    plt.tight_layout()

    if folder != '':
        # Make sure folder exists
        if not Path(folder).exists():
            Path(folder).mkdir(parents=True, exist_ok=True)
        # Export
        fig.savefig(f'{folder}/asc.png', dpi=300)
        fig.savefig(f'{folder}/asc.pdf')
        
    return fig


def plot_side_room(
        ax, rd2D, rs, rn, r,
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


def plot_signals_all_nodes(out: DANSEoutputs, wasn: list[Node]):
    """
    Plot DANSE output signals, comparing with inputs.

    Parameters
    ----------
    out : `DANSEoutputs` object
        DANSE run outputs.

    Returns
    -------
    figs : list of `matplotlib.figure.Figure` objects
        Figure handle for each node.
    """

    figs = []
    # Plot per node
    for k in range(out.nNodes):
        fig, axForTitle = plot_signals(
            node=wasn[k],
            win=out.winWOLAanalysis,
            ovlp=out.WOLAovlp
        )
        axForTitle.set_title(
            f'Node {k + 1}, {out.nSensorPerNode[k]} sensor(s) ($\\beta$={np.round(out.beta[k], 4)})'
        )
        plt.tight_layout()
        figs.append(fig)

    return figs


def plot_signals(node: Node, win, ovlp):
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
        label='Noisy (ref. sensor)'
    )
    # Desired signal estimate waveform 
    ax.plot(
        node.timeStamps,
        node.enhancedData - 4*delta,
        label='Enhanced (global)'
    )
    currDelta = 4*delta
    if len(node.enhancedData_l) > 0:
        ax.plot(
            node.timeStamps,
            node.enhancedData_l - currDelta - 2*delta,
            label='Enhanced (local)'
        )
        currDelta += 2*delta
    if len(node.enhancedData_c) > 0:
        ax.plot(
            node.timeStamps,
            node.enhancedData_c - currDelta - 2*delta,
            label='Enhanced (centr.)'
        )
    # Plot start of enhancement metrics computations
    ax.vlines(
        x=node.metricStartTime,
        ymin=np.amin(ax.get_ylim()),
        ymax=np.amax(ax.get_ylim()),
        colors='0.5',
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


def plot_filter_norms(
        filters,
        filtersCentre=None,
        nSensorsPerNode=None,
        refSensorIdx=0
    ) -> dict:
    """
    Plot filter norms.

    Parameters
    ----------
    filters : [K x 1] list of [Nf x Nt x J] np.ndarray[complex]
        DANSE filters per node.
        `K` : Number of nodes.
        `Nf` : Number of frequency bins.
        `Nt` : Number of time frames.
        `J` : Filter dimensions.
    filtersCentre : [K x 1] list of [Nf x Nt x J] np.ndarray[complex]
        Centralized DANSE filters per node.
    nSensorPerNode : [K x 1] list[int]
        Number of sensors per node.
    refSensorIdx : int
        Index of reference sensor (same for all nodes).

    Returns
    ----------
    `figs` : dict of matplotlib.figure.Figure and strings
        Figure handle and export name.
    dataFigs : list[np.ndarray]
        List of data corresponding to each figure in `figs`.
    """

    # Get number of nodes in WASN
    nNodes = len(filters)
    
    # Compute y-axis limits
    l = [np.log10(np.mean(np.abs(filt), axis=0)) for filt in filters]
    maxNorm = np.nanmax([np.nanmax(ll[np.isfinite(ll)]) for ll in l])   # avoid NaNs and inf's
    minNorm = np.nanmin([np.nanmin(ll[np.isfinite(ll)]) for ll in l])   # avoid NaNs and inf's

    def _format_axes(myAx, ti, maxNorm, minNorm):
        """
        Format axes.
        
        Parameters
        ----------
        myAx : matplotlib.axes._subplots.AxesSubplot
            Axes handle.
        ti : str
            Axes title.
        maxNorm : float
            Maximum filter norm.
        minNorm : float
            Minimum filter norm.
        """
        myAx.autoscale(enable=True, axis='x', tight=True)
        myAx.set(
            xlabel='STFT time frame index $i$',
            ylabel='$\\log_{{10}}(E_{{\\nu}}\\{|w_{{k,m}}[\\nu, i]|\\})$',
            title=ti,
            xticks=myAx.get_xticks(),
            xticklabels=np.round(np.linspace(
                start=0,
                stop=filters[0].shape[1] - 1,
                num=len(myAx.get_xticks()),
                dtype=int
            ), 2),
            ylim=[minNorm, maxNorm]
        )
        myAx.legend(loc='upper right')
        myAx.grid(True)

    # Useful sensor to node index reference
    sensorToNodeIndices = [
        [ii for _ in range(Mk)] for ii, Mk in enumerate(nSensorsPerNode)
    ]
    # Flatten list
    sensorToNodeIndices = [
        item for sublist in sensorToNodeIndices for item in sublist
    ]

    # Plot filter norms for regular DANSE filters
    figs = []
    dataFigs = []
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
                lab += f' (Node $k={neighborIndices[neighborCount] + 1}$)'
                neighborCount += 1
            # Mean over frequency bins
            dataPlot[:, m] = np.log10(
                np.mean(np.abs(filters[k][:, :, m]), axis=0)
            )
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
        _format_axes(ax, ti, maxNorm, minNorm)
        fig.tight_layout()
        figs.append((f'filtnorms_n{k + 1}', fig))
        dataFigs.append(dataPlot)  # Save data for later use

    
    if filtersCentre is not None:
        # Plot filter norms for ``centralized'' (== no-fusion DANSE) filters
        for k in range(nNodes):
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            nodeCount = 0
            idxCurrNodeSensor = 0
            dataFig = np.zeros_like(filtersCentre[k][0, :, :], dtype=float)
            for m in range(filtersCentre[k].shape[2]):        
                # Get label for legend
                lab = f'$m={m + 1}$, Node {nodeCount + 1}'
                if m == np.sum(nSensorsPerNode[:k]) + refSensorIdx:
                    lab += ' (reference)'
                # Mean over frequency bins
                dataFig[:, m] = np.log10(
                    np.mean(np.abs(filtersCentre[k][:, :, m]), axis=0)
                )
                ax.plot(
                    dataFig[:, m],
                    f'C{nodeCount}-',
                    alpha=1 / nSensorsPerNode[nodeCount] * (idxCurrNodeSensor + 1),
                    label=lab,
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
            _format_axes(ax, ti, maxNorm, minNorm)
            fig.tight_layout()
            figs.append((f'filtnorms_c{k + 1}', fig))
            dataFigs.append(dataFig)  # Save data for later use

    # Transform to dict
    figs = dict(figs)
    
    return figs, dataFigs


def export_danse_outputs(
        out: DANSEoutputs,
        wasnObj: WASN,
        room: pra.room.ShoeBox,
        p: TestParameters
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
        # Export filter coefficients norm plot
        if p.exportParams.filterNormsPlot:
            out.plot_filter_norms(
                p.exportParams.exportFolder,
                exportNormsAsPickle=p.exportParams.filterNorms  # boolean to export filter norms as pickle  
            )

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
            out.export_sounds(wasnObj.wasn, p.exportParams.exportFolder)

        # Plot (+ export) acoustic scenario (WASN)
        if p.exportParams.acousticScenarioPlot:
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
            fig.savefig(f'{p.exportParams.exportFolder}/sroEvolution.png')
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
                p.exportParams.onlySNRandESTOIinPlots,
                snrYlimMax=p.snrYlimMax,
            )

    if not p.exportParams.bypassAllExports:
        # Plot signals at specific nodes (+ export)
        if p.exportParams.waveformsAndSpectrograms:
            out.plot_sigs(wasnObj.wasn, p.exportParams.exportFolder)

        # Save `DANSEoutputs` object after metrics computation
        if p.exportParams.danseOutputsFile:
            out.save(foldername=p.exportParams.exportFolder, light=True)
        # Save `TestParameters` object
        if p.exportParams.parametersFile:
            if p.loadedFromYaml:
                p.save_yaml()   # save `TestParameters` object as YAML file
            else:
                p.save()    # save `TestParameters` object as Pickle archive

    return out


def export_batch_danse_outputs(
        out: BatchDANSEoutputs,
        wasnObj: WASN,
        p: TestParameters
    ):
    """
    Post-processing after a batch DANSE run.

    Parameters
    ----------
    out : `danse.danse_toolbox.d_post.BatchDANSEoutputs` object
        Batch DANSE outputs.
    wasnObj : `WASN` object
        WASN under consideration, after DANSE processing.
    p : `TestParameters` object
        Test parameters.

    Returns
    -------
    out : `danse.danse_toolbox.d_post.BatchDANSEoutputs` object
        Batch DANSE outputs after post-processing.
    """
        
    if not p.exportParams.bypassAllExports:
        # out.plot_filters_evol()  # TODO: figure out whether this is useful or not

        if p.exportParams.mmsePerfPlot:
            fig = out.plot_mmse_perf()
            fig.savefig(f'{p.exportParams.exportFolder}/mmseEvolution.png', dpi=300)
            fig.savefig(f'{p.exportParams.exportFolder}/mmseEvolution.pdf')

        # Plot waveforms and spectrograms
        if p.exportParams.waveformsAndSpectrograms:
            figs = out.plot_sigs()
            for k, fig in enumerate(figs):
                fig.savefig(f'{p.exportParams.exportFolder}/waveforms_node{k + 1}.png', dpi=300)

        # Export .wav files
        if p.exportParams.wavFiles:
            out.export_sounds(wasnObj.wasn, p.exportParams.exportFolder)

    return out