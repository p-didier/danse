
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import danse_toolbox.dataclass_methods as met

@dataclass
class AcousticScenarioParameters:
    rd: np.ndarray = np.array([5, 5, 5])  # room dimensions [m]
    fs: float = 16000.     # base sampling frequency [Hz]
    t60: float = 0.        # reverberation time [s]
    minDistToWalls: float = 0.33   # minimum distance between elements and room walls [m]
    #    
    referenceSensor: int = 0    # Index of the reference sensor at each node
    interSensorDist: float = 0.1   # distance separating microphones
    arrayGeometry: str = 'grid3d'   # microphone array geometry (only used if numSensorPerNode > 1)
    # ^^^ possible values: 'linear', 'radius', 'grid3d'
    #
    lenRIR: int = 2**10    # length of RIRs [samples]
    sigDur: float = 5.     # signals duration [s]
    #
    nDesiredSources: int = 1   # number of desired sources
    nNoiseSources: int = 1     # number of undesired (noise) sources
    desiredSignalFile: list[str] = field(default_factory=list)  # list of paths to desired signal file(s)
    noiseSignalFile: list[str] = field(default_factory=list)    # list of paths to noise signal file(s)
    baseSNR: int = 5                        # [dB] SNR between dry desired signals and dry noise
    #
    VADenergyFactor: float = 4000           # VAD energy factor (VAD threshold = max(energy signal)/VADenergyFactor)
    VADwinLength: float = 40e-3             # [s] VAD window length
    #
    nNodes: int = 1        # number of nodes in scenario
    nSensorPerNode: list[int] = field(default_factory=list)    # number of sensors per node
    #
    loadFrom: str = ''  # if provided, tries and load an ASC from .pkl.gz archives (used to load older scenarios (<= year 2022))


@dataclass
class WASNparameters(AcousticScenarioParameters):
    SROperNode: np.ndarray = np.array([0])
    topologyType: str = 'fully-connected'       # type of WASN topology
        # ^^^ valid values: "fully-connected"; TODO: add some
    selfnoiseSNR: float = 50   # self-noise SNR
        # [signal: noise-free signal; noise: self-noise]
    
    def __post_init__(self):
        # Dimensionality checks
        if type(self.SROperNode) is not list:
            # ^^^ required check for JSON export/import
            if len(self.SROperNode) != self.nNodes:
                if all(self.SROperNode == self.SROperNode[0]):
                    print(f'Automatically setting all SROs to the only value provided ({self.SROperNode[0]} PPM).')
                    self.SROperNode = np.full(
                        self.nNodes,
                        fill_value=self.SROperNode[0]
                    )
                else:
                    raise ValueError(f'The number of SRO values ({len(self.SROperNode)}) does not correspond to the number of nodes in the WASN ({self.nNodes}).')
        # Explicitly derive sensor-to-node indices
        self.sensorToNodeIndices = np.array(
            list(itertools.chain(*[[ii] * self.nSensorPerNode[ii]\
            for ii in range(len(self.nSensorPerNode))])))  
            # ^^^ itertools trick from https://stackoverflow.com/a/953097
        # Sampling rate per node
        if type(self.SROperNode) is not list:
            self.fsPerNode = self.fs * (1 + self.SROperNode * 1e-6)
        else:  # < --required check for JSON export/import
            self.fsPerNode = self.fs *\
                (1 + np.array(self.SROperNode[:-1]) * 1e-6)

@dataclass
class Node:
    nSensors: int = 1
    sro: float = 0.
    fs: float = 16000.
    cleanspeech: np.ndarray = np.array([])  # mic. signals if no noise present
    data: np.ndarray = np.array([])  # mic. signals
    enhancedData: np.ndarray = np.array([]) # signals after enhancement
    enhancedData_c: np.ndarray = np.array([]) # after CENTRALISED enhancement
    enhancedData_l: np.ndarray = np.array([]) # after LOCAL enhancement
    timeStamps: np.ndarray = np.array([])
    neighborsIdx: list[int] = field(default_factory=list)
    vad: np.ndarray = np.array([])
    beta: float = 1.    # exponential averaging (forgetting factor)
                        # for Ryy and Rnn updates at this node

    def __post_init__(self):
        # Combined VAD
        self.vadCombined = np.array(
            [1 if sum(self.vad[ii, :]) > 0 else 0\
                for ii in range(self.vad.shape[0])]
        )
        # Combined wet clean speeches
        self.cleanspeechCombined = np.sum(self.cleanspeech, axis=1)


# @dataclass
# class PlottingOptions:
#     """
#     Copy-pasted on 19.01.2023 from
#     "01_algorithms/03_signal_gen/01_acoustic_scenes/utilsASC/classes.py".
#     """
#     nodeCircleRadius: float = None      # radius of circle to be plotted around each node (if None, compute radius dependent on nodes coordinates)
#     nodesColors: str = 'multi'          # color used for each node. If "multi", use a different color for each node
#     plot3D: bool = False
#     texts: bool = True      # if True, show the desired and noise sources references on the graph itself
#     nodesNr: bool = True    # if True, show the node numbers on the graph itself


# @dataclass
# class AcousticScenario:
#     """
#     Class for keeping track of acoustic scenario parameters
#     Copy-pasted on 19.01.2023 from
#     "01_algorithms/03_signal_gen/01_acoustic_scenes/utilsASC/classes.py".
#     """
#     rirDesiredToSensors: np.ndarray = np.array([1])     # RIRs between desired sources and sensors
#     rirNoiseToSensors: np.ndarray = np.array([1])       # RIRs between noise sources and sensors
#     desiredSourceCoords: np.ndarray = np.array([1])     # Coordinates of desired sources
#     sensorCoords: np.ndarray = np.array([1])            # Coordinates of sensors
#     sensorToNodeTags: np.ndarray = np.array([1])        # Tags relating each sensor to its node
#     noiseSourceCoords: np.ndarray = np.array([1])       # Coordinates of noise sources
#     roomDimensions: np.ndarray = np.array([1])          # Room dimensions   
#     absCoeff: float = 1.                                # Absorption coefficient
#     samplingFreq: float = 16000.                        # Sampling frequency
#     numNodes: int = 2                                   # Number of nodes in network
#     distBtwSensors: float = 0.05                        # Distance btw. sensors at one node
#     topology: str = 'fully_connected'                   # WASN topology type ("fully_connected" or ...TODO)

#     def __post_init__(self):
#         """Post object initialization function.
        
#         Parameters
#         ----------
#         rng : np.random.default_range() random generator
#             Random generator.
#         seed : int
#             Seed to create a random generator (only used if `rng is None`).
#         """
#         self.numDesiredSources = len(self.desiredSourceCoords)      # number of desired sources
#         self.numSensors = len(self.sensorCoords)                    # number of sensors
#         self.numNoiseSources = len(self.noiseSourceCoords)          # number of noise sources
#         self.numSensorPerNode = np.unique(self.sensorToNodeTags, return_counts=True)[-1]    # number of sensors per node
#         return self
    
#     # Save and load
#     def load(self, foldername: str):
#         a: AcousticScenario = met.load(self, foldername)
#         return a
#     def save(self, filename: str):
#         met.save(self, filename, exportType='pkl')

#     def plot(self, options: PlottingOptions = PlottingOptions()):

#         # Determine appropriate node radius for ASC subplots
#         nodeRadius = 0
#         for k in range(self.numNodes):
#             allIndices = np.arange(self.numSensors)
#             sensorIndices = allIndices[self.sensorToNodeTags == k + 1]
#             curr = np.amax(self.sensorCoords[sensorIndices, :] - np.mean(self.sensorCoords[sensorIndices, :], axis=0))
#             if curr > nodeRadius:
#                 nodeRadius = copy.copy(curr)

#         # Detect noiseless scenarios
#         noiselessFlag = self.rirNoiseToSensors.shape[-1] == 0

#         fig, (a0, a1) = plt.subplots(2, 2, gridspec_kw={'height_ratios': [3, 1]})
#         plot_side_room(a0[0], self.roomDimensions[0:2], 
#                     self.desiredSourceCoords[:, [0,1]], 
#                     self.noiseSourceCoords[:, [0,1]], 
#                     self.sensorCoords[:, [0,1]],
#                     self.sensorToNodeTags,
#                     dotted=self.absCoeff==1,
#                     options=options,
#                     nodeRadius=nodeRadius)
#         a0[0].set(xlabel='$x$ [m]', ylabel='$y$ [m]', title='Top view')
#         #
#         plot_side_room(a0[1], self.roomDimensions[1:], 
#                     self.desiredSourceCoords[:, [1,2]], 
#                     self.noiseSourceCoords[:, [1,2]],
#                     self.sensorCoords[:, [1,2]],
#                     self.sensorToNodeTags,
#                     dotted=self.absCoeff==1,
#                     options=options,
#                     showLegend=False,
#                     nodeRadius=nodeRadius)
#         a0[1].set(xlabel='$y$ [m]', ylabel='$z$ [m]', title='Side view')
            
#         # Add distance info
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#         boxText = 'Node distances\n\n'
#         for ii in range(self.numNodes):
#             for jj in range(self.desiredSourceCoords.shape[0]):
#                 d = np.mean(np.linalg.norm(self.sensorCoords[self.sensorToNodeTags == ii + 1,:] - self.desiredSourceCoords[jj,:]))
#                 boxText += f'{ii + 1}$\\to$D{jj + 1}={np.round(d, 2)}m\n'
#             for jj in range(self.noiseSourceCoords.shape[0]):
#                 d = np.mean(np.linalg.norm(self.sensorCoords[self.sensorToNodeTags == ii + 1,:] - self.noiseSourceCoords[jj,:]))
#                 boxText += f'{ii + 1}$\\to$N{jj + 1}={np.round(d, 2)}m\n'
#             boxText += '\n'
#         boxText = boxText[:-1]
#         # Plot RIRs
#         t = np.arange(self.rirDesiredToSensors.shape[0]) / self.samplingFreq

#         # Set RIRs plots y-axes bounds
#         if noiselessFlag:
#             ymax = np.amax(self.rirDesiredToSensors[:, 0, 0])
#             ymin = np.amin(self.rirDesiredToSensors[:, 0, 0])
#         else:
#             ymax = np.amax([np.amax(self.rirDesiredToSensors[:, 0, 0]), np.amax(self.rirNoiseToSensors[:, 0, 0])])
#             ymin = np.amin([np.amin(self.rirDesiredToSensors[:, 0, 0]), np.amin(self.rirNoiseToSensors[:, 0, 0])])

#         # Plot RIRs
#         a1[0].plot(t, self.rirDesiredToSensors[:, 0, 0], 'k')
#         a1[0].grid()
#         a1[0].set(xlabel='$t$ [s]', title=f'RIR node 1 - D1')
#         a1[0].set_ylim([ymin, ymax])
#         if noiselessFlag:
#             a1[1].set_xlim([0, 1])
#             a1[1].set_ylim([0, 1])
#             a1[1].text(0.5, 0.5, 'Noiseless',ha='center', va='center')
#             a1[1].set_xticks([])
#             a1[1].set_yticks([])
#         else:
#             a1[1].plot(t, self.rirNoiseToSensors[:, 0, 0], 'k')
#             a1[1].grid()
#             a1[1].set(xlabel='$t$ [s]', title=f'RIR node 1 - N1')
#             a1[1].set_ylim([ymin, ymax])
#         # Add text boxes
#         a0[1].text(1.1, 0.9, boxText, transform=a0[1].transAxes, fontsize=10,
#                 verticalalignment='top', bbox=props)
#         a1[1].text(1.1, 0.1, f'Abs. coeff.:\n$\\alpha$ = {np.round(self.absCoeff, 2)}', transform=a1[1].transAxes, fontsize=10,
#                 verticalalignment='top', bbox=props)
#         fig.tight_layout()
#         return fig


# def plot_side_room(
#     ax, rd2D, rs, rn, r, sensorToNodeTags,
#     options: PlottingOptions,
#     scatsize=20,
#     dotted=False,
#     showLegend=True,
#     nodeRadius=None
#     ):
#     """Plots a 2-D room side, showing the positions of
#     sources and nodes inside of it.
#     Parameters
#     ----------
#     ax : Axes handle
#         Axes handle to plot on.
#     rd2D : [2 x 1] list
#         2-D room dimensions [m].
#     rs : [Ns x 2] np.ndarray (real)
#         Desired (speech) source(s) coordinates [m]. 
#     rn : [Nn x 2] np.ndarray (real)
#         Noise source(s) coordinates [m]. 
#     r : [N x 2] np.ndarray (real)
#         Sensor(s) coordinates [m].
#     TODO: options
#         ...
#     sensorToNodeTags : [N x 1] np.ndarray (int)
#         Tags relating each sensor to a node number (>=1).
#     scatsize : float
#         Scatter plot marker size.
#     dotted : bool
#         If true, use dotted lines. Else, use solid lines (default).
#     """

#     numNodes = len(np.unique(sensorToNodeTags))
#     numSensors = len(sensorToNodeTags)
    
#     plot_room2D(ax, rd2D, dotted)
#     # Desired sources
#     for idxSensor in range(rs.shape[0]):
#         ax.scatter(rs[idxSensor,0], rs[idxSensor,1], s=2*scatsize,c='lime',marker='d', edgecolor='k')
#         if options.texts:
#             ax.text(rs[idxSensor,0], rs[idxSensor,1], "D%i" % (idxSensor+1))
#     # Noise sources
#     for idxSensor in range(rn.shape[0]):
#         ax.scatter(rn[idxSensor,0], rn[idxSensor,1], s=2*scatsize,c='red',marker='P', edgecolor='k')
#         if options.texts:
#             ax.text(rn[idxSensor,0], rn[idxSensor,1], "N%i" % (idxSensor+1))
#     # Nodes and sensors
#     if options.nodesColors == 'multi':
#         circHandles = []
#         leg = []
#     for idxNode in range(numNodes):
#         allIndices = np.arange(numSensors)
#         sensorIndices = allIndices[sensorToNodeTags == idxNode + 1]
#         for idxSensor in sensorIndices:
#             if options.nodesColors == 'multi':
#                 ax.scatter(r[idxSensor,0], r[idxSensor,1], s=scatsize,c=f'C{idxNode}',edgecolors='black',marker='o')
#             else:
#                 ax.scatter(r[idxSensor,0], r[idxSensor,1], s=scatsize,c=options.nodesColors,edgecolors='black',marker='o')
#         # Draw circle around node
#         if nodeRadius is not None:
#             radius = nodeRadius
#         else:
#             radius = np.amax(r[sensorIndices, :] - np.mean(r[sensorIndices, :], axis=0))
#         if options.nodesColors == 'multi':
#             circ = plt.Circle((np.mean(r[sensorIndices,0]), np.mean(r[sensorIndices,1])),
#                                 radius * 2, color=f'C{idxNode}', fill=False)
#             circHandles.append(circ)
#             leg.append(f'Node {idxNode + 1}')
#         else:
#             circ = plt.Circle((np.mean(r[sensorIndices,0]), np.mean(r[sensorIndices,1])),
#                                 radius * 2, color=options.nodesColors, fill=False)
#         # Add label
#         if options.nodesNr:
#             ax.text(np.mean(r[sensorIndices,0]) + 1.5*radius,
#                     np.mean(r[sensorIndices,1]) + 1.5*radius,
#                     f'$\\mathbf{{{idxNode+1}}}$', c=f'C{idxNode}')
#         ax.add_patch(circ)
#     ax.grid()
#     ax.set_axisbelow(True)
#     ax.axis('equal')
#     if showLegend and options.nodesColors == 'multi':
#         nc = 1  # number of columbs in legend object
#         if len(circHandles) >= 4:
#             nc = 2
#         ax.legend(circHandles, leg, loc='lower right', ncol=nc, mode='expand')
#     return None


# def plot_room2D(ax, rd, dotted=False):
#     """Plots the edges of a rectangle in 2D on the axes <ax>
    
#     Parameters
#     ----------
#     ax : matplotlib Axes object
#         Axes object onto which the rectangle should be plotted.
#     rd : [3 x 1] (or [1 x 3], or [2 x 1], or [1 x 2]) np.ndarray or list of float
#         Room dimensions [m].
#     dotted : bool
#         If true, use dotted lines. Else, use solid lines (default).
#     """

#     fmt = 'k'
#     if dotted:
#         fmt += '--'
    
#     ax.plot([rd[0],0], [0,0], fmt)
#     ax.plot([0,0], [0,rd[1]], fmt)
#     ax.plot([rd[0],rd[0]], [0,rd[1]], fmt)
#     ax.plot([0,rd[0]], [rd[1],rd[1]], fmt)

#     return None
