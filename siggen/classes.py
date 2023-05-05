
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import danse_toolbox.dataclass_methods as met
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D


@dataclass
class AcousticScenarioParameters:
    rd: np.ndarray = np.array([5, 5, 5])  # room dimensions [m]
    fs: float = 16000.     # base sampling frequency [Hz]
    t60: float = 0.        # reverberation time [s]
    minDistToWalls: float = 0.5   # minimum distance between elements and room walls [m]
    layoutType: str = 'random'
        # type of acoustic scenario layout
        # ^^^ valid values:
        #  - "random": random layout, no constraints on source/nodes placement.
        #  - "random_spinning_top": random spinning top layout
        #       Description: the spinning top layout is a 3D layout where the
        #       nodes are arranged in a circle around the line along which
        #       the source are placed. In this case, the line's orientation
        #       is random.
        #  - "vert_spinning_top": vertical spinning top layout
        #       Description: same as above, but with a vertical source line.
    #
    spinTop_randomWiggleAmount: float = 0.0  # [m] amount of random wiggle
        # in the node positions (only for the spinning top layout).
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
    # vvv VAD parameters vvv
    VADenergyDecrease_dB: float = 30   # The threshold is `VADenergyDecrease_dB` below the peak signal energy
    VADwinLength: float = 20e-3     # [s] VAD window length
    #
    nNodes: int = 0        # number of nodes in scenario
    nSensorPerNode: list[int] = field(default_factory=list)    # number of sensors per node
    #
    loadFrom: str = ''  # if provided, tries and load an ASC from .pkl.gz archives (used to load older scenarios (<= year 2022))')

@dataclass
class TopologyParameters:
    """
    Class for simulation parameters related to the WASN's topology.
    """
    topologyType: str = 'fully-connected'       # type of WASN topology
        # ^^^ valid values:
        #  - "fully-connected": Fully connected WASN;
        #  - "ad-hoc": Ad-hoc topology WASN, based on `commDistance` field;
        #  - "user-defined": User-defined topology.
    #
    # vvv Only used if `topologyType == 'ad-hoc'`:
    commDistance: float = 0.    # maximum inter-node communication distance [m]
    seed: int = 12345     # random-generator seed
    plotTopo: bool = False  # if True, plots a visualization of the topology,
                            # once created in `siggen.utils.get_topo()`.
    userDefinedTopo: np.ndarray = np.array([])  # connectivity matrix
    # ^^^ used only iff `topologyType == 'user-defined'`.

    def __post_init__(self):
        """Post-initialization checks (automatically conducted when invoking
        a class instance)."""
        if self.topologyType == 'user-defined':
            # Check that the WASN is connected
            if not nx.is_connected(nx.from_numpy_array(self.userDefinedTopo)):
                raise ValueError('The provided "user-defined" adjacency matrix corresponds to an unconnected graph.')
            # If fully connected, adapt fields
            if (self.userDefinedTopo == 1).all():
                inp = input('User-defined topology is fully connected. Change field "topologyType" to "fully-connected"? [y/[n]]  ')
                if inp in ['y', 'Y']:
                    print('Setting field "topologyType" to "fully-connected" -> will compute DANSE (not TI-DANSE)')
                    self.topologyType = 'fully-connected'
                else:
                    print(f'Keeping field "topologyType" as is ("{self.topologyType}").')

@dataclass
class WASNparameters(AcousticScenarioParameters):
    generateRandomWASNwithSeed: int = 0     # if > 0: ignore all other
        # parameters and generate a completely random WASN.
    SROperNode: np.ndarray = np.array([0])
    topologyParams: TopologyParameters = TopologyParameters()
    selfnoiseSNR: float = 50.   # self-noise SNR
        # [signal: noise-free signal; noise: self-noise]
    # vvv Experimental parameters
    addedNoiseSignalsPerNode: list = field(default_factory=list)
        # Number of random-noise (unusable) signals to be added to each node.
    sensorToNodeIndicesASC: list = field(default_factory=list)
        # List of sensor-to-node indices, for ASC plotting.
    nSensorPerNodeASC: list = field(default_factory=list)
        # List of sensor-to-node indices, for ASC plotting.
    
    def __post_init__(self):
        """Post-initialization commands, automatically conducted when invoking
        a class instance."""
        if int(self.generateRandomWASNwithSeed) > 0:
            rng = np.random.default_rng(int(self.generateRandomWASNwithSeed))
            # Generate random WASN
            self.nNodes = int((10 - 5) * rng.random() + 5)
            self.nSensorPerNode = np.array([
                int((5 - 1) * rng.random() + 1) for _ in range(self.nNodes)
            ])
            self.topologyParams.topologyType = 'ad-hoc'
            # Inform user
            print(f"""
            RANDOMLY GENERATED WASN (`p.WASNparameters.generateRandomWASNwithSeed > 0`):
            >> {self.nNodes} nodes;
            >> # sensor(s) per node: {self.nSensorPerNode};
            >> Topology: ad-hoc.
            """)

        def _dim_check(var, nNodes, printoutRef):
            """Helper function -- checks the dimensionality of a variable."""
            if isinstance(var, list):
                var = np.array(var)
            elif isinstance(var, float) or isinstance(var, int):
                var = np.array([var])
            if len(var) != nNodes:
                if len(var) > 0:
                    if all(var == var[0]):
                        print(f'Automatically setting all {printoutRef} to the only value provided ({var[0]}).')
                        var = np.full(
                            nNodes,
                            fill_value=var[0]
                        )
                    else:
                        raise ValueError(f'The number of {printoutRef} values ({len(var)}) does not correspond to the number of nodes in the WASN ({nNodes}).')
                else:
                    # If no value provided, use 0
                    var = np.full(nNodes, fill_value=0)
            return var
        
        # Dimensionality checks
        self.addedNoiseSignalsPerNode = _dim_check(
            self.addedNoiseSignalsPerNode,
            self.nNodes,
            'added random-noise (unusable) signals per node'
        )
        self.SROperNode = _dim_check(
            self.SROperNode,
            self.nNodes,
            'SRO per node'
        )

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
        if self.nNodes != len(self.nSensorPerNode):
            raise ValueError(f'The length of the list containing the numbers of sensor per node ({len(self.nSensorPerNode)}) does not match the number of nodes ({self.nNodes}).')
        # VAD energy factor (VAD threshold = max(energy signal)/VADenergyFactor)
        self.VADenergyFactor = 10 ** (self.VADenergyDecrease_dB / 10)

@dataclass
class Node:
    index: int = 0
    nSensors: int = 1
    refSensorIdx: int = 0
    sro: float = 0.
    fs: float = 16000.
    cleanspeech: np.ndarray = np.array([])  # mic. signals if no noise present
    cleannoise: np.ndarray = np.array([])  # mic. signals if no speech present
    data: np.ndarray = np.array([])  # mic. signals
    enhancedData: np.ndarray = np.array([]) # signals after enhancement
    enhancedData_c: np.ndarray = np.array([]) # after CENTRALISED enhancement
    enhancedData_l: np.ndarray = np.array([]) # after LOCAL enhancement
    timeStamps: np.ndarray = np.array([])
    neighborsIdx: list[int] = field(default_factory=list)
    downstreamNeighborsIdx: list[int] = field(default_factory=list)
    upstreamNeighborsIdx: list[int] = field(default_factory=list)
    vad: np.ndarray = np.array([])          # VAD
    vadPerFrame: np.ndarray = np.array([])  # VAD per frame
    beta: float = 1.    # exponential averaging (forgetting factor)
                        # for Ryy and Rnn updates at this node.
    betaWext: float = 1.    # exponential averaging (forgetting factor)
                            # for wEXT updates from wEXTtarget at this node.
    # Geometrical parameters
    sensorPositions: np.ndarray = np.array([])  # coordinates of each sensor
    nodePosition: np.ndarray = np.array([])     # global node coordinates
    nodeType: str = 'default'   # type of node:
        # -'default': no particular type. Used, e.g., in fully connected WASNs;
        # -'leaf': leaf node, with only one neighbor.
        # -'root': root node, root of a tree-topology WASN.\
    # Other
    metricStartTime : float = 0.    # start time instant to compute speech
        # enhancement metrics at that node [s].

    def __post_init__(self):
        # Combined VAD
        self.vadCombined = np.array(
            [1 if sum(self.vad[ii, :]) > 0 else 0\
                for ii in range(self.vad.shape[0])]
        )
        # Wet clean speech at reference sensor
        self.cleanspeechRefSensor = self.cleanspeech[:, self.refSensorIdx]
        # Wet clean noise at reference sensor
        self.cleannoiseRefSensor = self.cleannoise[:, self.refSensorIdx]


@dataclass
class WASN:
    wasn: list[Node] = field(default_factory=list)  # list of `Node` objects
    adjacencyMatrix: np.ndarray = np.array([])  # adjacency matrix of WASN
    rootSelectionMeasure: str = 'snr'       
        # ^ measure to select the tree root
        # -'snr': signal-to-noise ratio estimate.
        # -'user-defined': user-defined estimate.
        # -anything else: invalid.
    userDefinedRoot: int = None    # index of user-defined root node
        # ^ used iff `rootSelectionMeasure == 'user-defined'`.
    rootIdx: int = userDefinedRoot  # effective root node index
    leafToRootOrdering: list = field(default_factory=list)
        # ^ node ordering from leaf to root (elements can be `int`
        #   or `list[int]`).
    vadPerFrameCentralized: np.ndarray = np.array([]) # centralized VAD
        # ^ boolean array.

    def set_tree_root(self):
        """Sets the root of a tree-topology WASN."""
        # Base check
        if (self.adjacencyMatrix == 1).all():
            print('/!\ The WASN is fully connected: cannot select a root!')
            return None

        def _get_snr(data, vad, refSensor=0):
            """Helper function: get SNR from microphone signals and VAD."""
            snr = 10 * np.log10(np.mean(np.abs(data[:, refSensor])**2) /\
                np.mean(np.abs(data[vad[:, refSensor] == 1, refSensor])**2))
            return snr

        if self.rootSelectionMeasure == 'snr':
            # Estimate SNRs
            snrs = np.array([
                _get_snr(node.data, node.vad) for node in self.wasn
            ])
            rootIdx = np.argmax(snrs)
            print(f'Node {rootIdx+1} was set as the root based on SNR estimates.')
        elif self.rootSelectionMeasure == 'user-defined':
            if self.userDefinedRoot is not None:
                rootIdx = self.userDefinedRoot
                print(f'Node {rootIdx+1} was set as the root (user-defined).')
            else:
                raise ValueError('The user-defined root was not provided.')
        else:
            raise ValueError(f"The measure used to select the tree's root ('{self.rootSelectionMeasure}') is invalid.")

        # Set root
        self.wasn[rootIdx].nodeType = 'root'
        self.rootIdx = rootIdx
    
    def get_metrics_start_time(
            self,
            startComputeMetricsAt: str,
            minNoSpeechDurEndUtterance: float
        ):
        """Infers a good start time for the computation of speech enhancement
        metrics based on the speech signal used (after 1 speech utterance -->
        whenever the VAD has gone up and down).
        
        Parameters
        ----------
        startComputeMetricsAt : str
            Time at which to start computing metrics.
                Valid values:
                -- 'beginning_2nd_utterance': start computing metrics at the
                beginning of the 2nd utterance.
                -- 'beginning_1st_utterance': start computing metrics at the
                beginning of the 1st utterance.
                -- 'end_1st_utterance': start computing metrics at the end of the
                1st utterance.
        minNoSpeechDurEndUtterance : float
            Minimum duration of no speech at the end of a speech utterance [s].
        """
        # Get VADs
        VADs = [node.vad for node in self.wasn]

        # Check that the VADs are for single-sources only
        if VADs[0].shape[-1] > 1:
            raise ValueError('NYI: multiple-sources VAD case.')  # TODO:
        
        def _jump_over_short_changes(idx):
            """Helper function: jump over short VAD changes."""
            idxVadToNoSpeech = vadChangesIndices[idx]
            # Next VAD change to speech
            nextIdxVADtoSpeech = vadChangesIndices[idx + 1]
            while nextIdxVADtoSpeech - idxVadToNoSpeech <\
                minNoSpeechDurEndUtterance * self.wasn[k].fs:
                idx += 2  # go to next VAD change to no-speech
                # Update indices
                idxVadToNoSpeech = vadChangesIndices[idx]  
                nextIdxVADtoSpeech = vadChangesIndices[idx + 1]
            return idx

        nNodes = len(VADs)
        for k in range(nNodes):
            # Compute indices of VAD changes
            allIndices = np.arange(1, len(VADs[k]))
            vadDiffVect = np.diff(np.squeeze(VADs[k]), axis=0)
            vadChangesIndices = allIndices[vadDiffVect != 0]

            # Get the initial VAD state
            initialVADstate = VADs[k][0][0]  # 0 or 1 (no speech or speech)
            
            # Start at the first VAD change to no-speech
            if initialVADstate == 0:  # if we start with no speech
                # The first VAD change is to speech --> select the second.
                idxVADstartMetrics = 1
            elif initialVADstate == 1:  # if we start with speech
                # The first VAD change is to no-speech --> select it.
                idxVADstartMetrics = 0
            
            if startComputeMetricsAt == 'end_1st_utterance':
                # Jump over short (less than `minNoSpeechDurEndUtterance`) 
                # no-speech (VAD == 0) segments --> until reaching the 
                # end of an actual utterance.
                idxVADstartMetrics = _jump_over_short_changes(idxVADstartMetrics)
                idxVadToNoSpeech = vadChangesIndices[idxVADstartMetrics]
                
                # Set the metrics computation start time [s]
                self.wasn[k].metricStartTime = idxVadToNoSpeech / self.wasn[k].fs

            elif startComputeMetricsAt == 'beginning_1st_utterance':
                # Set the metrics computation start time [s]
                if idxVADstartMetrics - 1 < 0:
                    self.wasn[k].metricStartTime = 0
                else:
                    self.wasn[k].metricStartTime =\
                        vadChangesIndices[idxVADstartMetrics - 1] / self.wasn[k].fs
                
            elif 'beginning_2nd_utterance' in startComputeMetricsAt:
                # Jump over short (less than `minNoSpeechDurEndUtterance`) 
                # no-speech (VAD == 0) segments --> until reaching the 
                # end of an actual utterance.
                idxVADstartMetrics = _jump_over_short_changes(idxVADstartMetrics)
                idxVadToSpeech = vadChangesIndices[idxVADstartMetrics + 1]

                # Set the metrics computation start time [s]
                self.wasn[k].metricStartTime = idxVadToSpeech / self.wasn[k].fs

                if 'after' in startComputeMetricsAt:
                    # Add the specified duration [s] after the beginning of the
                    # 2nd utterance
                    durAfterMs = float(startComputeMetricsAt.split('_')[-1])
                    self.wasn[k].metricStartTime += durAfterMs / 1e3
                
    def orientate(self):
        """Orientate the tree-topology from leaves towards root."""
        # Base check
        if 'root' not in [node.nodeType for node in self.wasn]:
            raise ValueError('The WASN cannot be orientated: missing root node.')
        
        def identify_upstream_downstream(nodeIdx: int, wasn: list[Node], passedNodes):
            """Recursive helper function to orientate WASN."""
            nextNodesIndices = []
            for n in wasn[nodeIdx].neighborsIdx:
                # Identify downstream neighbors
                if nodeIdx not in wasn[n].downstreamNeighborsIdx and\
                    nodeIdx not in wasn[n].upstreamNeighborsIdx:
                    wasn[n].downstreamNeighborsIdx.append(nodeIdx)
                    if n not in passedNodes:
                        nextNodesIndices.append(n)
                        if n not in wasn[nodeIdx].upstreamNeighborsIdx:
                            wasn[nodeIdx].upstreamNeighborsIdx.append(n)
                # Identify upstream neighbors
                for ii in wasn[n].neighborsIdx:
                    if ii not in wasn[n].downstreamNeighborsIdx and\
                        ii not in wasn[n].upstreamNeighborsIdx:
                        wasn[n].upstreamNeighborsIdx.append(ii)
            return nextNodesIndices, wasn

        nextRootIndices = [self.rootIdx]
        passedNodes = []
        # Iteratively orientate the WASN
        self.leafToRootOrdering.append([self.rootIdx])
        while nextRootIndices != []:
            nextNextRootIndices = []
            for k in nextRootIndices:
                foo, self.wasn = identify_upstream_downstream(
                    k, self.wasn, passedNodes
                )
                passedNodes.append(k)
                nextNextRootIndices += foo
            nextRootIndices = nextNextRootIndices
            # Save branch ordering (from root to leaves, for now)
            if nextRootIndices != []:
                self.leafToRootOrdering.append(nextRootIndices)
        self.leafToRootOrdering.reverse()  # reverse to go from leaves to root
            
    def plot_me(self, ax=None, scatterSize=300):
        """
        Plots the WASN in 3D.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes3D object
            3D axes where to plot.
        scatterSize : int /or/ float
            Size of the markers for scatter plots.
        """

        # Convert to NetworkX `Graph` object
        Gnx = nx.from_numpy_array(self.adjacencyMatrix)
        nodesPos = dict(
            [(k, self.wasn[k].nodePosition) for k in range(len(self.wasn))]
        )
        # Extract node and edge positions from the layout
        node_xyz = np.array([nodesPos[v] for v in sorted(Gnx)])
        edge_xyz = np.array(
            [(nodesPos[u], nodesPos[v]) for u, v in Gnx.edges()]
        )
        edge_xyz_idx = np.array(
            [(u, v) for u, v in Gnx.edges()]
        )
        # Plot the nodes - alpha is scaled by "depth" automatically
        if ax is None:
            fig = plt.figure()
            fig.set_size_inches(8.5, 3.5)
            ax = fig.add_subplot(projection='3d')
        ax.scatter(*node_xyz.T, s=scatterSize, ec="w")
        # Add node numbers
        for ii in range(len(nodesPos)):
            # Reference for node type (single letter)
            nodeTypeRef = self.wasn[ii].nodeType[0].upper()
            ax.text(
                node_xyz[ii][0],
                node_xyz[ii][1],
                node_xyz[ii][2],
                f'$\\mathbf{{{ii+1}}}$ ({nodeTypeRef}.)'
            )
            if nodeTypeRef == 'R':
                # Root node: highlight it
                ax.scatter(
                    node_xyz[ii][0],
                    node_xyz[ii][1],
                    node_xyz[ii][2],
                    s=scatterSize * 1.5,
                    c='r',
                    alpha=0.5
                )

        # Plot the upstream edges
        colorUpstream = 'black'
        for ii, vizedge in enumerate(edge_xyz):
            if (vizedge[0, :] != vizedge[1, :]).any():
                if 'root' in [node.nodeType for node in self.wasn]:
                    arrowOrientation = "-|>"
                    if edge_xyz_idx[ii][0] in\
                        self.wasn[edge_xyz_idx[ii][1]].upstreamNeighborsIdx:
                        arrowOrientation = "<|-"
                    draw_3d_arrow(
                        ax, vizedge, arrowOrientation, color=colorUpstream
                    )  # draw arrows
                else:
                    ax.plot(*vizedge.T, color=colorUpstream)
        
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        ax.set_zlabel('$z$ [m]')
    
    def get_vad_per_frame(self, frameLen, frameShift):
        """
        Computes the VAD per frame for each node in the WASN.

        Parameters
        ----------
        frameLen : int
            Frame length [samples].
        frameShift : int
            Frame shift [samples].
        """
        for k in range(len(self.wasn)):  # for each node
            vadCurrNode = self.wasn[k].vad
            if len(vadCurrNode) == 0:
                raise ValueError(f"Node {k} has no VAD.")
            # Compute VAD per frame
            vadPerFrame = np.zeros(len(vadCurrNode) // frameShift)
            for ii in range(len(vadPerFrame)):
                idxBeg = ii*frameShift
                idxEnd = ii*frameShift+frameLen
                if idxEnd > len(vadCurrNode):
                    vadPerFrame = vadPerFrame[:ii + 1]  # crop (no VAD if frame is incomplete)
                    break
                vadChunk = vadCurrNode[idxBeg:idxEnd]
                # VAD is 1 if more than half of the frame is active
                vadPerFrame[ii] = float(sum(vadChunk) >= len(vadChunk) // 2)
            # Save VAD per frame into `Node` object in WASN
            self.wasn[k].vadPerFrame = vadPerFrame.astype(bool)


def draw_3d_arrow(ax, coords, arrowOrientation, color="tab:gray"):
    """
    Inspired by
    https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c.
    
    Parameters
    ----------
    ax : matplotlib.pyplot.Axes3D object
        3D axes to plot on.
    coords : [2 x 3] np.ndarray (float)
        Coordinates [[xA, yA, zA], [xB, yB, zB]].
    """
    class Arrow3D(FancyArrowPatch):

        def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._xyz = (x, y, z)
            self._dxdydz = (dx, dy, dz)

        def draw(self, renderer):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform(
                (x1, x2), (y1, y2), (z1, z2), self.axes.M
            )
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            super().draw(renderer)
            
        def do_3d_projection(self, renderer=None):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform(
                (x1, x2), (y1, y2), (z1, z2), self.axes.M
            )
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

            return np.min(zs)
    
    def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
        '''Add an 3d arrow to an `Axes3D` instance.'''
        arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
        ax.add_artist(arrow)
    
    setattr(Axes3D, 'arrow3D', _arrow3D)

    ax.arrow3D(
        coords[0, 0],
        coords[0, 1],
        coords[0, 2],
        coords[1, 0] - coords[0, 0],
        coords[1, 1] - coords[0, 1],
        coords[1, 2] - coords[0, 2],
        mutation_scale=20,
        arrowstyle=arrowOrientation,
        color=color
    )