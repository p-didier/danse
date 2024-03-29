
import time
import itertools
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D


@dataclass
class RandomIRParameters:
    distribution: str = 'uniform'  # statistic distribution of randomly generated IR
        # ^^^ valid values:
        #  - "uniform": uniform distribution;
        #  - "normal": normal distribution.
    minValue: float = -.5   # minimum value of randomly generated IR
    maxValue: float = .5    # maximum value of randomly generated IR
    duration: float = 0.2   # [s] duration of randomly generated IR
    decay: str = 'none'   # decay of randomly generated IR
        # ^^^ valid values:
        #  - "none" (or `None`): no decay;
        #  - "exponential": exponential decay with time constant `decayTimeConstant`.
        #  - "immediate": IR is non-zero at t=0, 0 elsewhere.
    decayTimeConstant: float = 0.1  # [s] time constant of exponential decay

@dataclass
class RandomSignalsParameters:
    distribution: str = 'uniform'  # distribution of randomly generated signals
        # ^^^ valid values:
        #  - "uniform": uniform distribution;
        #  - "normal": normal distribution.
    minValue: float = -1.   # minimum value of randomly generated signals
    maxValue: float = 1.    # maximum value of randomly generated signals
    pauseType: str = 'none'   # distribution of pauses between randomly
        # generated signals fragments. Only applied to the "target" signals
        # to create "noise-only" periods. NB: the "interferer" signals are
        # always played continuously.
        # ^^^ valid values:
        #  - "none" (or `None`): no pauses;
        #  - "random": random distribution of pauses;
        #  - "predefined": uniform distribution of pauses (i.e., all pauses have
        #       the same duration `pauseDuration` and are separated
        #       by the same duration `pauseSpacing`).
    pauseDuration: float = 0.5   # [s] duration of pauses between
        # randomly generated desired signals fragments.
    pauseSpacing: float = 0.5    # [s] spacing between pauses between
        # randomly generated desired signals fragments.
    randPauseDuration_max: float = 0.5  # [s] maximum duration of pauses
        # between randomly generated desired signals fragments.
    randPauseDuration_min: float = 0.1  # [s] minimum duration of pauses
        # between randomly generated desired signals fragments.
    randPauseSpacing_max: float = 0.5   # [s] maximum spacing between pauses
        # between randomly generated desired signals fragments.
    randPauseSpacing_min: float = 0.1   # [s] minimum spacing between pauses
        # between randomly generated desired signals fragments.
    startWithPause: bool = False    # if True, starts the randomly generated
        # desired signals with a pause.


@dataclass
class AcousticScenarioParameters:
    trueRoom: bool = True  # if True, simulate an actual room. Otherwise,
        # randomly generate impulse responses from each sensor to each source.
    randIRsParams: RandomIRParameters = RandomIRParameters()
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
        #  - "predefined": predefined layout from YAML file `predefinedLayoutFile`.
        #  - "all_nodes_in_center": all nodes at the center of the room
        #       (useful for debugging).
    predefinedLayoutFile: str = ''  # used iff `layoutType == "predefined"`.
    #
    spinTop_randomWiggleAmount: float = 0.0  # [m] amount of random wiggle
        # in the node positions (only for the spinning top layout).
    spinTop_minInterNodeDist: float = None  # [m] minimum distance between nodes
        # (only for the spinning top layout).
    spinTop_minSourceSpacing: float = None  # [m] minimum distance between
        # sound sources (only for the spinning top layout).
    #    
    referenceSensor: int = 0    # Index of the reference sensor at each node
    interSensorDist: float = 0.1   # distance separating microphones
    arrayGeometry: str = 'grid3d'   # microphone array geometry (only used if numSensorPerNode > 1)
    # ^^^ possible values: 'linear', 'radius', 'grid3d'
    #
    lenRIR: int = 2**10    # length of RIRs [samples]
    sigDur: float = 5.     # signals duration [s]
    #
    # vvv Diffuse sources vvv
    diffuseNoise: bool = False  # if True, adds diffuse noise to the scenario
        # using submodule `pyANFgen`.
    diffuseNoisePowerFactor: float = 0. # [dB] power factor to apply to diffuse noise
        # ^^^ `diffuseNoise_k = get_diffuse_noise() * 10 ** (p.diffuseNoisePowerFactor / 20)`
    typeDiffuseNoise: str = 'noise' # type of diffuse noise
        # ^^^ valid values:
        #  - "noise": noise generated by `np.random.randn()`;
        #  - "babble": babble noise loaded via `fileDiffuseBabble`.
    fileDiffuseBabble: str = '' # path to diffuse babble noise file
    #
    # vvv Localized sources vvv
    nDesiredSources: int = 1   # number of desired sources
    nNoiseSources: int = 1     # number of undesired (noise) sources
    signalType: str = 'from_file'   # type of signals
        # ^^^ valid values:
        #  - "from_file": signals loaded from files
        #               (see `desiredSignalFile` and `noiseSignalFile` below);
        #  - "random": signals generated randomly.
    desiredSignalFile: list[str] = field(default_factory=list)
        # ^^^ list of paths to desired signal file(s)
    noiseSignalFile: list[str] = field(default_factory=list)
        # ^^^ list of paths to noise signal file(s)
        # NB: if `noiseSignalFile` is a string and
        # `noiseSignalFile == loadfrom <path>`, loads `nNoiseSources`
        # noise signals randomly selected from the audio files in the
        # folder at <path>.
    randSignalsParams: RandomSignalsParameters = RandomSignalsParameters()
    noiseSignalFilesLoadedFromFolder: str = None
        # folder from which noise signals were loaded. `None` by default.
    snrBasis: str = 'dry_signals'   # basis for SNR in acoustic scenario
        # ^^^ valid values:
        #  - "dry_signals": SNR computed between dry desired signals and dry noise;
        #  - "at_mic_<x>": SNR computed at microphone <x> (e.g., "at_mic_0") - 
        #       for each noise separately (NB: the microphone index is 0-based).
    snr: int = 5   # [dB] SNR between dry desired signals and dry noise
    # vvv VAD parameters vvv
    VADenergyDecrease_dB: float = 30   # The threshold is `VADenergyDecrease_dB` below the peak signal energy
    VADwinLength: float = 20e-3     # [s] VAD window length
    vadMinProportionActive: float = 0.5  # for computation of frame-based VAD:
        # the VAD at a frame is considered active if at least
        # `vadMinProportionActive` of the per-sample VAD values within the
        # frame are active.
    enableVADloadFromFile: bool = True  # if True, loads VAD from file
    vadFilesFolder: str = ''    # folder containing VAD files
    #
    nSensorPerNode: list[int] = field(default_factory=list)    # number of sensors per node
    #
    loadFrom: str = ''  # if provided, tries and load an ASC from .pkl.gz archives (used to load older scenarios (<= year 2022))')

    def __post_init__(self):
        if self.interSensorDist >= np.amin(self.rd) / 3:
            raise ValueError('`interSensorDist` must be smaller than the minimum room dimension divided by 3.')        


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
                print('WARNING: User-defined topology is fully connected but field is "topologyType".')
                time.sleep(0.2)  # let user read...
                # inp = input('User-defined topology is fully connected. Change field "topologyType" to "fully-connected"? [y/[n]]  ')
                # while inp not in ['y', 'n', 'Y', 'N']:
                #     inp = input('User-defined topology is fully connected. Change field "topologyType" to "fully-connected"? [y/[n]]  ')
                # if inp in ['y', 'Y']:
                #     print('Setting field "topologyType" to "fully-connected" -> will compute DANSE (not TI-DANSE)')
                #     self.topologyType = 'fully-connected'
                # else:
                #     print(f'Keeping field "topologyType" as is ("{self.topologyType}").')

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
        self.topologyParams.__post_init__()
        # Define `self.nNodes`
        self.nNodes = len(self.nSensorPerNode)
        #
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
        # Basic checks
        if self.minDistToWalls >= np.amin(self.rd) / 3:
            raise ValueError('`minDistToWalls` must be smaller than the minimum room dimension divided by 3.')

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

        def _select_random_files_from_folder(folder, nFiles, fs=None):
            """Helper function -- selects `nFiles` files randomly from a folder."""
            # Get all files in folder
            files = list(Path(folder).glob('**/*'))
            
            if self.fileDiffuseBabble != '':
                # Remove diffuse babble file, if any
                files = [x for x in files if x.is_file() and\
                    not Path.samefile(x, self.fileDiffuseBabble)]
            else:
                # Remove folders
                files = [x for x in files if x.is_file()]
            
            if fs is not None:
                # For each file, check if there is a corresponding file with
                # the same name but different sampling rate
                for file in files:
                    if Path(f'{str(file)[:-4]}_{int(fs)}Hz.wav') in files:
                        # If so, remove the file with the wrong sampling rate
                        files.remove(file)
            # Select `nFiles` files randomly
            return [
                str(files[ii]) for ii in np.random.choice(
                    np.arange(len(files)),
                    size=nFiles,
                    replace=False
                )
            ]
        
        # Loading noises from folder, if asked
        if self.noiseSignalFile[:len('loadfrom ')] == 'loadfrom ':
            # Get folder
            noiseFolder = self.noiseSignalFile[len('loadfrom '):]
            self.noiseSignalFile = _select_random_files_from_folder(
                folder=noiseFolder,
                nFiles=self.nNoiseSources,
                fs=self.fs
            )
            self.noiseSignalFilesLoadedFromFolder = noiseFolder
        elif self.noiseSignalFilesLoadedFromFolder is not None and\
            Path(self.noiseSignalFilesLoadedFromFolder).is_dir():
            # `elif` statement for when `__post_init__` is called for the N>1th
            # time (e.g., when loading a WASN layout from a YAML file).
            self.noiseSignalFile = _select_random_files_from_folder(
                folder=self.noiseSignalFilesLoadedFromFolder,
                nFiles=self.nNoiseSources,
                fs=self.fs
            )
        
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
        # Check validity of acoustic scenario
        if self.layoutType == 'predefined':
            if not Path(self.predefinedLayoutFile).is_file():
                raise ValueError(f'The file "{self.predefinedLayoutFile}" does not exist.')
            
        if self.topologyParams.topologyType == 'fully-connected':
            # For consistency and to avoid conflicts, make sure the user-
            # defined topology is ones.
            self.topologyParams.userDefinedTopo =\
                np.ones((self.nNodes, self.nNodes))
            
        # Adapt fields for random IRs scenarios
        if not self.trueRoom:
            if self.diffuseNoise:
                print('WARNING: `diffuseNoise` is set to True, but `trueRoom` is set to False. Setting `diffuseNoise` to False.')
                self.diffuseNoise = False

    def align_with_loaded_yaml_layout(self, layoutDict):
        """Ensures the WASN parameters are consistent with the layout loaded
        from a YAML file.
        
        Parameters
        ----------
        layoutDict : dict
            Dictionary containing the layout loaded from a YAML file (see
            `siggen.utils.load_layout_from_yaml()`).
        """
        self.rd = np.array(layoutDict['rd'])
        self.nSensorPerNode = np.array(layoutDict['Mk'])
        self.nNodes = len(self.nSensorPerNode)
        self.nDesiredSources = len(layoutDict['targetCoords'])
        self.nNoiseSources = len(layoutDict['interfererCoords'])
        self.__post_init__()  # re-run post-initialization commands

        # Check that enough signals were provided
        if self.nDesiredSources > len(self.desiredSignalFile):
            raise ValueError(f'Not enough desired signals provided ({len(self.desiredSignalFile)} provided, {self.nDesiredSources} required).')
        if self.nNoiseSources > len(self.noiseSignalFile):                
            raise ValueError(f'Not enough noise signals provided ({len(self.noiseSignalFile)} provided, {self.nNoiseSources} required).')

@dataclass
class Node:
    index: int = 0
    nSensors: int = 1
    refSensorIdx: int = 0
    sro: float = 0.
    fs: float = 16000.
    cleanspeech: np.ndarray = np.array([])  # mic. signals if no noise present
    cleanspeech_noSRO: np.ndarray = np.array([])  # mic. signals if no noise present and no SROs
    cleannoise: np.ndarray = np.array([])  # mic. signals if no speech present
    cleannoise_noSRO: np.ndarray = np.array([])  # mic. signals if no speech present and no SROs
    data: np.ndarray = np.array([])  # mic. signals
    data_noSRO: np.ndarray = np.array([])  # mic. signals with no SROs
    enhancedData: np.ndarray = np.array([]) # signals after enhancement
    enhancedData_c: np.ndarray = np.array([]) # after CENTRALISED enhancement
    enhancedData_l: np.ndarray = np.array([]) # after LOCAL enhancement
    enhancedData_ssbc: np.ndarray = np.array([]) # after single-sensor broadcast enhancement
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
    metricEndTime : float = 0.    # end time instant to compute speech
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

    def all_nodes_at_same_position(self):
        """Returns True if all nodes of the WASN are located at the
        same position in the environment."""
        return all([
            np.allclose(self.wasn[0].nodePosition, node.nodePosition)\
                for node in self.wasn
        ])

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
    
    def get_metrics_key_time(
            self,
            ref: str,
            minNoSpeechDurEndUtterance: float,
            timeType: str = 'start'
        ):
        """
        Infers a good start (or end) time for the computation of speech
        enhancement metrics based on the speech signal used (e.g., after 1
        speech utterance --> whenever the VAD has gone up and down).
        
        Parameters
        ----------
        ref : str
            Reference string for key time at which to start (or end) computing metrics.
                Valid values:
                -- 'beginning_2nd_utterance': start (or end) computing metrics at the
                beginning of the 2nd utterance.
                -- 'beginning_1st_utterance': start (or end) computing metrics at the
                beginning of the 1st utterance.
                -- 'end_1st_utterance': start (or end) computing metrics at the end of the
                1st utterance.
        minNoSpeechDurEndUtterance : float
            Minimum duration of no speech at the end of a speech utterance [s].
        timeType : str, optional
            Type of key time to return. Valid values:
                -- 'start': return the start time [s] of the metrics computation.
                -- 'end': return the end time [s] of the metrics computation.
        """
        # Get VADs
        VADs = [node.vad for node in self.wasn]

        # Check that the VADs are for single-sources only
        if VADs[0].shape[-1] > 1:
            raise ValueError('NYI: multiple-sources VAD case.')  # TODO:

        nNodes = len(VADs)
        times = np.zeros(nNodes)
        for k in range(nNodes):
            # Compute the key time
            if ref is None and timeType == 'start':
                times[k] = 0
            elif ref is None and timeType == 'end':
                times[k] = self.wasn[k].data.shape[0] / self.wasn[k].fs
            else:
                times[k] = get_key_time(
                    ref=ref,
                    vad=VADs[k],
                    fs=self.wasn[k].fs,
                    minNoSpeechDurEndUtterance=minNoSpeechDurEndUtterance
                )
            
            if timeType == 'start':
                self.wasn[k].metricStartTime = times[k]
            elif timeType == 'end':
                self.wasn[k].metricEndTime = times[k]

    def orientate(self):
        """Orientate the tree-topology from leaves towards root."""
        # Base check
        if 'root' not in [node.nodeType for node in self.wasn]:
            raise ValueError('The WASN cannot be orientated: missing root node.')
        
        def identify_upstream_downstream(
                nodeIdx: int,
                wasn: list[Node],
                passedNodes: list[int]
            ):
            """Recursive helper function to orientate WASN."""
            nextNodesIndices = []
            for q in wasn[nodeIdx].neighborsIdx:
                # Identify downstream neighbors
                if nodeIdx not in wasn[q].downstreamNeighborsIdx and\
                    nodeIdx not in wasn[q].upstreamNeighborsIdx:
                    wasn[q].downstreamNeighborsIdx.append(nodeIdx)
                    if q not in passedNodes:
                        nextNodesIndices.append(q)
                        if q not in wasn[nodeIdx].upstreamNeighborsIdx:
                            wasn[nodeIdx].upstreamNeighborsIdx.append(q)
                # Identify upstream neighbors
                for ii in wasn[q].neighborsIdx:
                    if ii not in wasn[q].downstreamNeighborsIdx and\
                        ii not in wasn[q].upstreamNeighborsIdx:
                        wasn[q].upstreamNeighborsIdx.append(ii)
            return nextNodesIndices, wasn

        nextRootIndices = [self.rootIdx]
        passedNodes = []
        # Reset object before iteratively orientating the WASN
        self.leafToRootOrdering = [self.rootIdx]
        for k in range(len(self.wasn)):
            self.wasn[k].upstreamNeighborsIdx = []
            self.wasn[k].downstreamNeighborsIdx = []
        
        # Iteratively orientate the WASN
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
    
    def get_vad_per_frame(self, frameLen, frameShift, minProportionActive=0.5):
        """
        Computes the VAD per frame for each node in the WASN.

        Parameters
        ----------
        frameLen : int
            Frame length [samples].
        frameShift : int
            Frame shift [samples].
        minProportionActive : float
            Minimum proportion of active samples in a frame for the frame to be
            considered active.
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
                # VAD is 1 if more than `minProportionActive` of the frame
                # is active.
                vadPerFrame[ii] = float(
                    sum(vadChunk) >= len(vadChunk) * minProportionActive
                )
            # Save VAD per frame into `Node` object in WASN* 
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

def get_key_time(ref, vad, fs, minNoSpeechDurEndUtterance):
    """
    Returns the key start (or end) time [s] for the computation
    of speech enhancement metrics based on the speech signal.
    """
    
    def _jump_over_short_changes(idx):
        """Helper function: jump over short VAD changes."""
        idxVadToNoSpeech = vadChangesIndices[idx]
        # Next VAD change to speech
        nextIdxVADtoSpeech = vadChangesIndices[idx + 1]
        while nextIdxVADtoSpeech - idxVadToNoSpeech <\
            minNoSpeechDurEndUtterance * fs:
            idx += 2  # go to next VAD change to no-speech
            # Update indices
            idxVadToNoSpeech = vadChangesIndices[idx]  
            nextIdxVADtoSpeech = vadChangesIndices[idx + 1]
        return idx
    
    # Compute indices of VAD changes
    allIndices = np.arange(1, len(vad))
    vadDiffVect = np.diff(np.squeeze(vad), axis=0)
    vadChangesIndices = allIndices[vadDiffVect != 0]

    # Get the initial VAD state
    initialVADstate = vad[0][0]  # 0 or 1 (no speech or speech)
    
    # Start at the first VAD change to no-speech
    if initialVADstate == 0:  # if we start with no speech
        # The first VAD change is to speech --> select the second.
        idxVADstartMetrics = 1
    elif initialVADstate == 1:  # if we start with speech
        # The first VAD change is to no-speech --> select it.
        idxVADstartMetrics = 0
    
    if ref == 'end_1st_utterance':
        # Jump over short (less than `minNoSpeechDurEndUtterance`) 
        # no-speech (VAD == 0) segments --> until reaching the 
        # end of an actual utterance.
        idxVADstartMetrics = _jump_over_short_changes(idxVADstartMetrics)
        idxVadToNoSpeech = vadChangesIndices[idxVADstartMetrics]
        
        # Set the metrics computation start time [s]
        return idxVadToNoSpeech / fs

    elif ref == 'beginning_1st_utterance':
        # Set the metrics computation start time [s]
        if idxVADstartMetrics - 1 < 0:
            return 0
        else:
            return vadChangesIndices[idxVADstartMetrics - 1] / fs
        
    elif 'beginning_2nd_utterance' in ref:
        # Jump over short (less than `minNoSpeechDurEndUtterance`) 
        # no-speech (VAD == 0) segments --> until reaching the 
        # end of an actual utterance.
        idxVADstartMetrics = _jump_over_short_changes(idxVADstartMetrics)
        idxVadToSpeech = vadChangesIndices[idxVADstartMetrics + 1]

        # Set the metrics computation start time [s]
        t = idxVadToSpeech / fs

        if 'after' in ref:
            # Add the specified duration [s] after the beginning of the
            # 2nd utterance
            durAfterMs = float(ref.split('_')[-1])
            t += durAfterMs / 1e3
        
        return t
    
    elif ref[:len('after')] == 'after':
        if 'ms' in ref:
            unitFactor = 1e-3
            durAfterBeg = float(ref[len('after_'):-2])
        elif 's' in ref:
            unitFactor = 1
            durAfterBeg = float(ref[len('after_'):-1])
        else:
            raise ValueError(f'Invalid unit for `startComputeMetricsAt` ({ref}).')
        # Set the metrics computation start time [s]
        return durAfterBeg * unitFactor