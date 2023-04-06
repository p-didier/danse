import os
import copy
import librosa
import resampy
import numpy as np
import pickle, gzip
from numba import njit
from pathlib import Path
from siggen import classes
import scipy.signal as sig
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rot


def build_room(p: classes.WASNparameters):
    """
    Builds room, adds nodes and sources, simulates RIRs
    and computes microphone signals.

    Parameters
    ----------
    p : `WASNparameters` object
        Parameters.

    Returns
    -------
    room : `pyroomacoustics.room.ShoeBox` object
        Room (acoustic scenario) object.
    vad : [N x Nnodes x Nsources] np.ndarray (bool or int [0 or 1])
        VAD per sample, per node, and per speech source.
    wetSpeeches : [Nnodes x 1] list of [Nm[k] x N] np.ndarray (float)
        Wet (RIR-affected) speech signal at each sensor of each node.
        `Nm[k]` is the number of microphones at node `k`.
    wetNoises : [Nnodes x 1] list of [Nm[k] x N] np.ndarray (float)
        Wet (RIR-affected) noise signal at each sensor of each node.
    """

    # Invert Sabine's formula to obtain the parameters for the ISM simulator
    if p.t60 == 0:
        max_order = 0
        e_absorption = 0.5  # <-- arbitrary
    else:
        e_absorption, max_order = pra.inverse_sabine(p.t60, p.rd)
    
    # Create room
    room = pra.ShoeBox(
        p=p.rd,
        fs=p.fs,
        max_order=max_order,
        air_absorption=False,
        materials=pra.Material(e_absorption),
    )

    # Desired number of samples
    desiredNumSamples = int(p.sigDur * p.fs)

    def _get_source_signal(file):
        """Helper function to load and process source signal."""
        # Load
        y, fsOriginal = librosa.load(file)
        # Resample
        y = resampy.resample(y, fsOriginal, p.fs)
        # Adjust length
        if len(y) > desiredNumSamples:
            y = y[:desiredNumSamples]
        elif len(y) < desiredNumSamples:
            while len(y) < desiredNumSamples:
                y = np.concatenate((y, y))  # loop
                y = y[:desiredNumSamples]
        # Whiten
        y = (y - np.mean(y)) / np.std(y)  # whiten
        return y
    
    if p.layoutType == 'random':
        for k in range(p.nNodes):
            # Generate node and sensors
            r = np.random.uniform(size=(3,)) * (p.rd - 2 * p.minDistToWalls)\
                + p.minDistToWalls # node centre coordinates
            sensorsCoords = generate_array_pos(
                r,
                p.nSensorPerNode[k],
                p.arrayGeometry,
                p.interSensorDist,
                applyRandomRot=True
            )
            room.add_microphone_array(sensorsCoords.T)

        # Add desired sources
        desiredSignalsRaw = np.zeros((desiredNumSamples, p.nDesiredSources))
        for ii in range(p.nDesiredSources):
            # Load sound file
            y = _get_source_signal(p.desiredSignalFile[ii])
            desiredSignalsRaw[:, ii] = y  # save (for VAD computation)
            ssrc = pra.soundsource.SoundSource(
                position=np.random.uniform(size=(3,)) *\
                    (p.rd - 2 * p.minDistToWalls) + p.minDistToWalls, # coordinates
                signal=desiredSignalsRaw[:, ii]
            )
            room.add_soundsource(ssrc)

        # Add noise sources
        noiseSignalsRaw = np.zeros((desiredNumSamples, p.nNoiseSources))
        for ii in range(p.nNoiseSources):
            # Load sound file
            y = _get_source_signal(p.noiseSignalFile[ii])
            y *= 10 ** (-p.baseSNR / 20)        # gain to set SNR
            noiseSignalsRaw[:, ii] = y  # save (for use in metrics computation)
            ssrc = pra.soundsource.SoundSource(
                position=np.random.uniform(size=(3,)) *\
                    (p.rd - 2 * p.minDistToWalls) + p.minDistToWalls, # coordinates
                signal=noiseSignalsRaw[:, ii]
            )
            room.add_soundsource(ssrc)
        #
    elif p.layoutType == 'random_spinning_top':  # Spinning top layout
        # Generate a random line for the sources
        azimuthLine = np.random.uniform(np.pi / 8, np.pi / 2 - np.pi / 8)
        elevationLine = np.random.uniform(np.pi / 8, np.pi / 2 - np.pi / 8)
        maxR = np.sqrt(np.sum(p.rd ** 2))  # room diagonal length

        def _random_point_on_line(az, el):
            """Helper function to generate a random point on a line."""
            # Ensure the points are in the room
            cond1, cond2, cond3 = True, True, True
            x, y, z = -1, -1, -1
            while cond1 or cond2 or cond3:
                r = np.random.uniform(
                    p.minDistToWalls,
                    maxR - p.minDistToWalls
                )
                x = r * np.cos(az) * np.sin(el)
                y = r * np.sin(az) * np.sin(el)
                z = r * np.cos(el)
                # Update conditions
                cond1 = x > p.rd[0] - p.minDistToWalls or x < p.minDistToWalls
                cond2 = y > p.rd[1] - p.minDistToWalls or y < p.minDistToWalls
                cond3 = z > p.rd[2] - p.minDistToWalls or z < p.minDistToWalls
            return x, y, z

        # Generate speech sources along the line
        desiredSignalsRaw = np.zeros((desiredNumSamples, p.nDesiredSources))
        for ii in range(p.nDesiredSources):
            # Generate random point on the line
            x, y, z = _random_point_on_line(azimuthLine, elevationLine)
            # Add source
            sourceSig = _get_source_signal(p.desiredSignalFile[ii])
            desiredSignalsRaw[:, ii] = sourceSig  # save (for VAD computation)
            ssrc = pra.soundsource.SoundSource(
                position=np.array([x, y, z]), # coordinates
                signal=desiredSignalsRaw[:, ii]
            )
            room.add_soundsource(ssrc)

        # Generate noise sources along the line
        noiseSignalsRaw = np.zeros((desiredNumSamples, p.nNoiseSources))
        for ii in range(p.nNoiseSources):
            # Generate random point on the line
            x, y, z = _random_point_on_line(azimuthLine, elevationLine)
            # Add source
            sourceSig = _get_source_signal(p.noiseSignalFile[ii])
            sourceSig *= 10 ** (-p.baseSNR / 20)        # gain to set SNR
            noiseSignalsRaw[:, ii] = sourceSig
            # ^^^ (for use in metrics computation)
            ssrc = pra.soundsource.SoundSource(
                position=np.array([x, y, z]), # coordinates
                signal=noiseSignalsRaw[:, ii]
            )
            room.add_soundsource(ssrc)
        
        # Generate node and sensor positions
        validNodePos = False
        attemptsCount = 0
        maxNumAttempts = 99
        while not validNodePos:
            # Generate random circle center
            cx, cy, cz = _random_point_on_line(azimuthLine, elevationLine)
            # Compute maximal circle radius based on room dimension
            refDistCenterCircle = np.sqrt(np.sum(np.array([cx, cy, cz]) ** 2))
            if refDistCenterCircle < maxR / 2:
                circRmax = (refDistCenterCircle - p.minDistToWalls) / 3  # <-- a little arbitrary... [PD06.04.2023]
            else:
                circRmax = (maxR - refDistCenterCircle - p.minDistToWalls) / 3
            # Generate random circle radius
            circR = np.random.uniform(
                2 * p.interSensorDist * np.amax(p.nSensorPerNode),
                circRmax
            )

            micArray = np.zeros((np.sum(p.nSensorPerNode), 3))
            # Generate array centers equally spaced on the circle
            angleArrayCenters = np.arange(0, 2 * np.pi, 2 * np.pi / p.nNodes)
            for k in range(p.nNodes):
                # Generate node and sensors on the perimeter of the circle
                r = angleArrayCenters[k]
                x = cx + circR * np.cos(r)
                y = cy + circR * np.sin(r)
                z = cz
                idxSensor = int(np.sum(p.nSensorPerNode[:k]))
                micArray[idxSensor:(idxSensor + p.nSensorPerNode[k]), :] =\
                    generate_array_pos(
                    np.array([x, y, z]),
                    p.nSensorPerNode[k],
                    p.arrayGeometry,
                    p.interSensorDist,
                    applyRandomRot=True
                )
            # Rotate coordinates so that the circle is perpendicular
            # to the source line.
            micArray = rotate_array_yx(
                micArray,
                targetVector=np.array([cx, cy, cz])
            )
            # Check that all microphones are in the room
            if np.any(micArray > p.rd - p.minDistToWalls) or\
                np.any(micArray < p.minDistToWalls):
                validNodePos = False
                attemptsCount += 1
                if attemptsCount > maxNumAttempts:
                    raise ValueError("Could not find valid node positions.")
            else:
                validNodePos = True
        
        room.add_microphone_array(micArray.T)

    if 0:  # debug
        fig = plt.figure()
        fig.set_size_inches(3.5, 3.5)
        ax = fig.add_subplot(projection='3d')
        plot_asc_3d(ax, room, p, cx, cy, cz, plotSourcesLine=True)

    room.compute_rir()
    room.simulate()

    # Truncate signals (no need for reverb tail)
    room.mic_array.signals = room.mic_array.signals[:, :int(p.fs * p.sigDur)]

    # Extract desired source RIRs per node
    rirsDesiredSources = []
    for k in range(p.nNodes):
        rirsCurr = [room.rir[ii][:p.nDesiredSources]\
            for ii in range(len(room.rir)) if p.sensorToNodeIndices[ii] == k]
        rirsDesiredSources.append(rirsCurr)
    # Extract noise source RIRs per node
    rirsNoiseSources = []
    for k in range(p.nNodes):
        rirsCurr = [room.rir[ii][-p.nNoiseSources:]\
            for ii in range(len(room.rir)) if p.sensorToNodeIndices[ii] == k]
        rirsNoiseSources.append(rirsCurr)
    # Get wet speech and compute VAD
    vad, wetSpeeches = get_vad(
        rirsDesiredSources,
        desiredSignalsRaw,
        p
    )
    # Get wet noise
    _, wetNoises = get_vad(
        rirsNoiseSources,
        noiseSignalsRaw,
        p
    )

    return room, vad, wetSpeeches, wetNoises

def plot_asc_3d(
        ax,
        room,
        p: classes.WASNparameters,
        cx=None,
        cy=None,
        cz=None,
        plotSourcesLine=False
    ):
    """Plot room, nodes, and sources in 3D.
    
    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        3D axes object.
    room : pyroomacoustics.Room
        Room object.
    p : Parameters
        Parameters object.
    cx, cy, cz : float
        Circle center coordinates.
    plotSourcesLine : bool
        If True, plot the line along which the sources are generated.
    """
    # Plot nodes
    for k in range(p.nNodes):
        ax.scatter(
            room.mic_array.R[0, p.sensorToNodeIndices == k],
            room.mic_array.R[1, p.sensorToNodeIndices == k],
            room.mic_array.R[2, p.sensorToNodeIndices == k],
            color='black'
        )
        # Plot node labels
        ax.text(
            room.mic_array.R[0, p.sensorToNodeIndices == k][0],
            room.mic_array.R[1, p.sensorToNodeIndices == k][0],
            room.mic_array.R[2, p.sensorToNodeIndices == k][0],
            str(k + 1),
            color='black'
        )
    # Plot desired sources
    for ii in range(p.nDesiredSources):
        ax.scatter(
            room.sources[ii].position[0],
            room.sources[ii].position[1],
            room.sources[ii].position[2],
            color='green',
            marker='d'  # thin diamond marker
        )
    # Plot noise sources
    for ii in range(p.nNoiseSources):
        ax.scatter(
            room.sources[p.nDesiredSources + ii].position[0],
            room.sources[p.nDesiredSources + ii].position[1],
            room.sources[p.nDesiredSources + ii].position[2],
            color='red',
            marker='P'  # big plus-sign marker
        )
    if p.layoutType == 'random_spinning_top' and plotSourcesLine:
        # Plot sources line
        ax.plot(
            [room.sources[0].position[0], room.sources[1].position[0]],
            [room.sources[0].position[1], room.sources[1].position[1]],
            [room.sources[0].position[2], room.sources[1].position[2]],
            color='black'
        )
    if cx is not None and cy is not None and cz is not None and plotSourcesLine:
        # Plot circle center
        ax.scatter(
            cx, cy, cz,
            color='yellow'
        )
    
    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')
    ax.set_zlabel('$z$ [m]')
    ax.set_xlim([0, p.rd[0]])
    ax.set_ylim([0, p.rd[1]])
    ax.set_zlim([0, p.rd[2]])
    ax.set_title('Room layout (3D)')
    plt.show()


def get_vad(rirs, xdry, p: classes.AcousticScenarioParameters):
    """
    Compute all node- and desired-source-specific VADs.

    Parameters
    ----------
    rirs : [Nnodes x 1] list of [Nsources x 1] lists of [variable length x 1]
            np.ndarrays (float)
        Individual RIRs between the reference sensor of each node
        and each source.
    xdry : [N x Nsources] np.ndarray (float)
        Dry (latent) source signals.
    p : AcousticScenarioParameters object
        Acoustic scenario parameters.

    Returns
    -------
    vad : [N x Nnodes x Nsources] np.ndarray (bool or int [0 or 1])
        VAD per sample, per node, and per speech source.
    wetsigs : [K x 1] list of [Nsensor[k] x N] np.ndarray (float)
        Wet (RIR-affected) speech (or noise) signal at each sensor of each node.
    """

    vad = np.zeros((xdry.shape[0], len(rirs), len(rirs[0][0])))
    wetsigs = [np.zeros((len(rirs[k]), xdry.shape[0], len(rirs[k][0]))) for k in range(len(rirs))]
    for k in range(len(rirs)):  # for each node
        for m in range(len(rirs[k])):  # for each microphone
            for ii in range(len(rirs[k][m])):  # for each desired source
                # Compute wet desired-only signal
                wetsig = sig.fftconvolve(xdry[:, ii], rirs[k][m][ii], axes=0)
                wetsigs[k][m, :, ii] = wetsig[:xdry.shape[0]]  # truncate

        for ii in range(len(rirs[k][p.referenceSensor])):  # for each desired source
            thrsVAD = np.amax(wetsigs[k][p.referenceSensor, :, ii] ** 2) /\
                p.VADenergyFactor
            vad[:, k, ii], _ = oracleVAD(
                wetsigs[k][p.referenceSensor, :, ii],
                tw=p.VADwinLength,
                thrs=thrsVAD,
                Fs=p.fs
            )

    # Sum wet signals over sources
    wetsigs = [np.sum(wetsig, axis=-1) for wetsig in wetsigs]

    return vad, wetsigs


def plot_mic_sigs(room: pra.room.ShoeBox, vad=None):
    """
    Quick plot of the microphone signals
    """

    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    axes.plot(room.mic_array.signals.T)
    if vad is not None:
        axes.plot(
            vad[:, 0, 0] * np.amax(abs(room.mic_array.signals)),
            'k',
            label='Oracle VAD'
        )
    axes.grid()
    axes.legend()
    plt.tight_layout()
    plt.show()


def generate_array_pos(
        nodeCoords,
        Mk,
        arrayGeom,
        micSep,
        force2D=False,
        applyRandomRot=False
    ):
    """
    Define node positions based on node position, number of nodes,
    and array type.

    Parameters
    ----------
    nodeCoords : [J x 3] array of real floats
        Nodes coordinates in 3-D space [m].
    TODO:
    force2D : bool
        If true, projects the sensor coordinates on the z=0 plane.
    applyRandomRot : bool
        If true, applies a random rotation to the sensors coordinates.

    Returns
    -------
    sensorCoords : [(J*Mk) x 3] array of real floats.
        Sensor coordinates in 3-D space [m].
    """

    if arrayGeom == 'linear':
        # 1D local geometry
        x = np.linspace(start=0, stop=Mk * micSep, num=Mk)
        # Center
        x -= np.mean(x)
        # Make 3D
        sensorCoordsBeforeRot = np.zeros((Mk,3))
        sensorCoordsBeforeRot[:,0] = x
        
        # Rotate in 3D through randomized rotation vector 
        rotvec = np.random.uniform(low=0, high=1, size=(3,))
        if force2D:
            rotvec[1:2] = 0
        sensorCoords = np.zeros_like(sensorCoordsBeforeRot)
        for ii in range(Mk):
            myrot = rot.from_rotvec(np.pi/2 * rotvec)
            sensorCoords[ii,:] =\
                myrot.apply(sensorCoordsBeforeRot[ii, :]) + nodeCoords
    elif arrayGeom == 'radius':
        radius = micSep 
        sensorCoords = np.zeros((Mk,3))
        for ii in range(Mk):
            flag = False
            while not flag:
                r = np.random.uniform(low=0, high=radius, size=(3,))
                if np.sqrt(r[0]**2 + r[1]**2 + r[2]**2) <= radius:
                    sensorCoords[ii, :] = r + nodeCoords - radius/2
                    flag = True
    elif arrayGeom == 'grid3d':
        sensorCoords = np.zeros((Mk, 3))
        # Create grid
        d = micSep
        x_ = np.linspace(nodeCoords[0] - d, nodeCoords[0] + d, 3)
        y_ = np.linspace(nodeCoords[1] - d, nodeCoords[1] + d, 3)
        z_ = np.linspace(nodeCoords[2] - d, nodeCoords[2] + d, 3)
        x, y, z = np.meshgrid(x_, y_, z_)
        # Flatten coordinates
        coordFlat = np.zeros((np.prod(x.shape), 3))
        counter = 0
        for ii in range(3):
            for jj in range(3):
                for kk in range(3):
                    coordFlat[counter, :] = [
                        x[ii,jj,kk],
                        y[ii,jj,kk],
                        z[ii,jj,kk]
                    ]
                    counter += 1
        # Base configuration ("atomic" -- see Word journal week 39, MON)
        idx = [13,4,22,10,16,14,12]
        sensorCoords[:np.amin([len(idx), Mk]), :] =\
            coordFlat[idx[:np.amin([len(idx), Mk])], :]
        if len(idx) < Mk:
            allIdx = np.arange(coordFlat.shape[0])
            idxValid = [ii for ii in allIdx if ii not in idx]
            for ii in range(Mk - len(idx)):
                sensorCoords[np.amin([len(idx), Mk]) + ii, :] =\
                    coordFlat[idxValid[ii], :]
    else:
        raise ValueError('No sensor array geometry defined for \
            array type "%s"' % arrayGeom)
    
    if applyRandomRot:
        # Rotate in 3D through randomized rotation vector, while keeping
        # the center of the array fixed.
        rotvec = np.random.uniform(low=0, high=1, size=(3,))
        if force2D:
            rotvec[1:2] = 0
        arrayCenter = np.mean(sensorCoords, axis=0)
        sensorCoordsAfterRot = np.zeros_like(sensorCoords)
        for ii in range(Mk):
            myrot = rot.from_rotvec(np.pi/2 * rotvec)
            sensorCoordsAfterRot[ii,:] =\
                myrot.apply(sensorCoords[ii, :] - arrayCenter)
        sensorCoords = sensorCoordsAfterRot + arrayCenter

    return sensorCoords


def export_asc(room: pra.room.ShoeBox, path):
    """
    Export acoustic scenario (pra.room.ShoeBox object)
    as compressed archive.
    """
    pickle.dump(room, gzip.open(path, 'wb'))



def oracleVAD(x,tw,thrs,Fs):
    """
    Oracle Voice Activity Detection (VAD) function. Returns the
    oracle VAD for a given speech (+ background noise) signal <x>.
    Based on the computation of the short-time signal energy.
    
    Parameters
    ----------
    -x [N*1 float vector, -] - Time-domain signal.
    -tw [float, s] - VAD window length.
    -thrs [float, [<x>]^2] - Energy threshold.
    -Fs [int, samples/s] - Sampling frequency.
    
    Returns
    -------
    -oVAD [N*1 binary vector] - Oracle VAD corresponding to <x>.

    (c) Paul Didier - 13-Sept-2021
    SOUNDS ETN - KU Leuven ESAT STADIUS
    ------------------------------------
    """

    # Check input format
    x = np.array(x)     # Ensure it is an array
    if len(x.shape) > 1:
        print('<oracleVAD>: input signal is multidimensional: \
            using 1st row as reference')
        dimsidx = range(len(x.shape))
        # Rearrange x dimensions in increasing order of size
        x = np.transpose(x, tuple(np.take(dimsidx,np.argsort(x.shape))))
        for ii in range(x.ndim-1):
            x = x[0]    # extract 1 "row" along the largest dimension

    # Number of samples
    n = len(x)

    # VAD window length
    if tw > 0:
        Nw = tw*Fs
    else:
        Nw = 1

    # Compute VAD
    oVAD = np.zeros(n)
    for ii in range(n):
        chunk_x = np.zeros(int(Nw))
        if Nw == 1:
            chunk_x[0] = x[ii]
        else:
            chunk_x = x[np.arange(ii,int(min(ii+Nw, len(x))))]
        oVAD[ii] = compute_VAD(chunk_x,thrs)

    # Time vector
    t = np.arange(n)/Fs

    return oVAD,t


@njit
def compute_VAD(chunk_x,thrs):
    """
    JIT-ed time-domain VAD computation
    
    (c) Paul Didier - 6-Oct-2021
    SOUNDS ETN - KU Leuven ESAT STADIUS
    ------------------------------------
    """
    # Compute short-term signal energy
    E = np.mean(np.abs(chunk_x)**2)
    # Assign VAD value
    if E > thrs:
        VADout = 1
    else:
        VADout = 0
    return VADout


def build_wasn(
        room: pra.room.ShoeBox,
        vad,
        wetSpeeches,
        wetNoises,
        p: classes.WASNparameters,
        minNoSpeechDurEndUterrance: float
    ):
    """
    Build WASN from parameters (including asynchronicities and topology).
    
    Parameters
    ----------
    room : [augmented] `pyroomacoustics.room.ShoeBox` object
        Room (acoustic scenario) object. Augmented with VAD and 
        wet speech signals at each node's reference sensor (output of
        `build_room()`).
    vad : [N x Nnodes x Nspeechsources] np.ndarray (bool or int [0 or 1])
        VAD per sample, per node, and per speech source.
    wetSpeeches : [Nnodes x 1] list of [Nm[k] x N] np.ndarray (float)
        Wet (RIR-affected) speech signal at each sensor of each node.
        `Nm[k]` is the number of microphones at node `k`.
    wetNoises : [Nnodes x 1] list of [Nm[k] x N] np.ndarray (float)
        Wet (RIR-affected) noise signal at each sensor of each node.
    p : `WASNparameters` object
        WASN parameters
    minNoSpeechDurEndUterrance : float
        Minimum duration of silence at the end of an utterance [s].

    Returns
    -------
    myWASN : `WASN` object
        WASN object, including all necessary values.
    """

    # Create network topological map (inter-node links)
    adjacencyMatrix, neighbors = get_topo(
        p.topologyParams,
        sensorToNodeIndices=p.sensorToNodeIndices,
        sensorCoords=room.mic_array.R  # microphones positions
    )

    myWASN = classes.WASN()
    for k in range(p.nNodes):
        # Apply asynchronicities
        sigs, t, fsSRO = apply_asynchronicity_at_node(
            y=room.mic_array.signals[p.sensorToNodeIndices == k, :].T,
            fs=p.fs,
            sro=p.SROperNode[k],
            sto=0.
        )
        # Apply microphone self-noise
        selfNoise = np.zeros_like(sigs)
        for m in range(sigs.shape[-1]):
            sigs[:, m], selfNoise[:, m] = apply_self_noise(
                sigs[:, m], p.selfnoiseSNR
            )
        selfNoiseRefSensor = selfNoise[:, 0]

        # Apply asynchronicities also to speech-only signal
        speechOnly, _, _ = apply_asynchronicity_at_node(
            y=wetSpeeches[k].T,
            fs=p.fs,
            sro=p.SROperNode[k],
            sto=0.
        )
        # Add same microphone self-noise
        speechOnly += selfNoiseRefSensor[:, np.newaxis]
        # Apply asynchronicities also to noise-only signal
        noiseOnly, _, _ = apply_asynchronicity_at_node(
            y=wetNoises[k].T,
            fs=p.fs,
            sro=p.SROperNode[k],
            sto=0.
        )
        # Add same microphone self-noise
        noiseOnly += selfNoiseRefSensor[:, np.newaxis]

        # Get geometrical parameters
        sensorPositions = room.mic_array.R[:, p.sensorToNodeIndices == k]
        nodePosition = np.mean(sensorPositions, axis=1)
        
        # Create node
        node = classes.Node(
            index=k,
            nSensors=p.nSensorPerNode[k],
            refSensorIdx=p.referenceSensor,
            sro=p.SROperNode[k],
            fs=fsSRO,
            data=sigs,
            cleanspeech=speechOnly,
            cleannoise=noiseOnly,
            timeStamps=t,
            neighborsIdx=neighbors[k],
            vad=vad[:, k, :],
            sensorPositions=sensorPositions,
            nodePosition=nodePosition
        )
        
        # Add to WASN
        myWASN.wasn.append(node)

    # Include adjacency matrix
    myWASN.adjacencyMatrix = adjacencyMatrix
    
    # Infer a good start time for the computation of speech enhancement
    # metrics based on the speech signal used (after 1 speech utterance -->
    # whenever the VAD has gone up and down).
    myWASN.get_metrics_start_time(
        minNoSpeechDurEndUterrance=minNoSpeechDurEndUterrance
    )

    return myWASN


def apply_self_noise(sig, snr):
    """Apply random self-noise to sensor signal `sig`."""
    sn = np.random.uniform(-1, 1, size=sig.shape)
    Pn = np.mean(np.abs(sn) ** 2)
    Ps = np.mean(np.abs(sig) ** 2)
    currSNR = 10 * np.log10(Ps / Pn)
    sn *= 10 ** (-(snr - currSNR) / 20)
    sig += sn
    return sig, sn


def apply_asynchronicity_at_node(y, fs, sro=0., sto=0.):
    """
    Apply asynchronicities (SROs and STOs) to the signals
    at the current node.

    Parameters
    ----------
    y : [N x M] np.ndarray (float)
        Time-domain signal(s) at node's `M` sensor(s).
    fs : float or int
        Nominal sampling frequency of node [Hz].
    sro : float or int
        Sampling rate offset with respect to
        the reference node in the WASN [PPM].
    sto : float
        Sampling time offset with respect to
        the reference node in the WASN [s].

    Returns
    -------
    xResamp : [N x M] np.ndarray (float)
        Resampled (and truncated /or/ padded) signals
    t : [N x 1] np.ndarray (float)
        Corresponding resampled time stamp vector.
    fsSRO : float
        True (SRO-affected nominal) sampling frequency [Hz].
    """

    yAsync = np.zeros_like(y)
    nSensors = y.shape[-1]
    for ii in range(nSensors):
        ycurr = y[:, ii]
        # Apply SRO
        yAsync[:, ii], t, fsSRO = resample_for_sro(ycurr, fs, sro)
        # Apply STO
        #TODO

    return yAsync, t, fsSRO


def get_topo(
    topoParams: classes.TopologyParameters,
    sensorToNodeIndices,
    sensorCoords,
    roomDim=[5, 5, 5]
    ):
    """
    Create inter-node connections matrix 
    depending on the type of WASN topology,
    and prepare corresponding lists of node-
    specific neighbors.

    Parameters
    ----------
    topoParams : TopologyParameters class instance
        Topology parameters (type, max. comm. distance, etc.).
    sensorToNodeIndices : list (int)
        Sensor-to-node indices for each sensor in the WASN.
    sensorCoords : [3 x Nsensors] np.ndarray (float)
        3-D coordinates of all sensors [m].
    roomDim : [3 x 1] list or np.ndarray (float)
        Room dimensions [x, y, z] (for plotting if 
        `topoParams.plotTopo == True`).

    Returns
    -------
    topo : [K x K] np.ndarray (int [or float]: 0 [0.] or 1 [1.])
        Connectivity matrix.
    neighbors : [K x 1] list of [variable-dim x 1] lists (int)
        Node-specific lists of neighbor nodes indices.
    """

    # Infer useful variables
    numNodes = np.amax(sensorToNodeIndices) + 1

    # Get geometrical central coordinates of each node
    geomCentreCoords = np.zeros((3, numNodes))
    for k in range(numNodes):
        geomCentreCoords[:, k] = np.mean(
            sensorCoords[:, sensorToNodeIndices == k],
            axis=1
        )

    # Potential TODO : oriented graph vvv
    # If `{topo}_{i,j} == 2`: `i` can send data to `j` but not vice-versa.
    # If `{topo}_{i,j} == 3`: `j` can send data to `i` but not vice-versa.
    
    # ------- FULLY CONNECTED -------
    if topoParams.topologyType == 'fully-connected':
        topo = np.ones((numNodes, numNodes), dtype=int)
    # ------- AD HOC -------
    elif topoParams.topologyType == 'ad-hoc':
        topo = np.zeros((numNodes, numNodes), dtype=int)
        for k in range(numNodes):
            currCoords = geomCentreCoords[:, k]
            # Compute inter-node distances
            distWrtOthers = np.linalg.norm(
                geomCentreCoords - currCoords[:, np.newaxis],
                axis=0
            )
            topo[k, :] = distWrtOthers < topoParams.commDistance
    # ------- USER DEFINED -------
    elif topoParams.topologyType == 'user-defined':
        # User-defined topology
        if topoParams.userDefinedTopo.shape != (numNodes, numNodes):
            raise ValueError(f'The user-defined WASN topology is wrongly dimensioned ([{topoParams.userDefinedTopo.shape[0]} x {topoParams.userDefinedTopo.shape[1]}], should be [{numNodes} x {numNodes}]). Aborting...')
        validConnMatValues = np.array([0, 1])
        if not all([val in validConnMatValues\
            for val in topoParams.userDefinedTopo.flatten()]):
            raise ValueError(f'The user-defined WASN connectivity matrix contains invalid values (valid values: {validConnMatValues}).')
        if not np.allclose(
            topoParams.userDefinedTopo,
            topoParams.userDefinedTopo.T
        ):
            raise ValueError('The user-defined WASN connectivity matrix must be symmetrical.')
        topo = topoParams.userDefinedTopo
    # ------- INVALID -------
    else:
        raise ValueError(f'Invalid topology type: "{topoParams.topologyType}".')

    # Plot if asked
    if topoParams.plotTopo:
        plot_topology(topo, geomCentreCoords, roomDim)
        plt.show()

    # Get node-specific lists of neighbor nodes indices
    allNodes = np.arange(numNodes)
    neighbors = [list(allNodes[(topo[:, k] > 0) &\
        (allNodes != k)]) for k in range(numNodes)]

    return topo, neighbors


def resample_for_sro(x, baseFs, SROppm):
    """Resamples a vector given an SRO and a base sampling frequency.

    Parameters
    ----------
    x : [N x 1] np.ndarray (float)
        Time-domain signal to be resampled.
    baseFs : float or int
        Base sampling frequency [samples/s].
    SROppm : float
        SRO [ppm].

    Returns
    -------
    xResamp : [N x 1] np.ndarray (float)
        Resampled signal
    t : [N x 1] np.ndarray (float)
        Corresponding resampled time stamp vector.
    fsSRO : float
        Re-sampled signal's sampling frequency [Hz].
    """

    fsSRO = baseFs * (1 + SROppm / 1e6)
    if baseFs != fsSRO:
        xResamp = resampy.core.resample(x, baseFs, fsSRO)
    else:
        xResamp = copy.copy(x)

    t = np.arange(len(xResamp)) / fsSRO

    if len(xResamp) >= len(x):
        xResamp = xResamp[:len(x)]
        t = t[:len(x)]
    else:
        # Extend time stamps vector
        dt = t[1] - t[0]
        tadd = np.linspace(t[-1]+dt, t[-1]+dt*(len(x) - len(xResamp)),\
            len(x) - len(xResamp))
        t = np.concatenate((t, tadd))
        # Append zeros
        xResamp = np.concatenate((xResamp, np.zeros(len(x) - len(xResamp))))

    return xResamp, t, fsSRO


def plot_topology(connectivityMatrix, nodeCoords, rd):
    """
    Plots a visualization of the WASN topology as a 3-D graph.
    
    Parameters
    ----------
    connectivityMatrix : [K x K] np.ndarray (int [or float]: 0 [0.] or 1 [1.])
        Connectivity matrix.
    nodeCoords : [3 x K] np.ndarray (float)
        3-D coordinates of the geometrical centre of each node in the WASN.
    rd : [3 x 1] list (or np.ndarray) (float)
        Room dimensions ([x, y, z]) [m].

    Returns
    -------
    fig : matplotlib.pyplot figure handle
        Figure handle (for, e.g., subsequent export of the figure).
    """

    def _plot_connection(ax, coordsNode1, coordsNode2):
        """Helper function to plot a single connection between two nodes."""
        ax.plot(
            [coordsNode1[0], coordsNode2[0]],
            [coordsNode1[1], coordsNode2[1]],
            [coordsNode1[2], coordsNode2[2]],
            'k'
        )

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    fig.set_size_inches(8.5, 3.5)
    for k in range(nodeCoords.shape[1]):
        ax.scatter(
            nodeCoords[0, k],
            nodeCoords[1, k],
            nodeCoords[2, k],
            marker='o',
            label=f'Node $k={k+1}$'
        )
    # Add topology connectivity lines
    for k in range(connectivityMatrix.shape[0]):
        for q in range(connectivityMatrix.shape[1]):
            # Only consider upper triangular matrix without diagonal
            # (redundant, otherwise)
            if k > q and connectivityMatrix[k, q] == 1:
                _plot_connection(ax, nodeCoords[:, k], nodeCoords[:, q])
    # Format axes
    ax.set_xlim([0, rd[0]])
    ax.set_ylim([0, rd[1]])
    ax.set_zlim([0, rd[2]])
    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')
    ax.set_zlabel('$z$ [m]')
    ax.legend()
    ax.grid()
    plt.tight_layout()	
    
    return fig


def rotate_array_yx(array, targetVector=None):
    """
    Rotates an array of 3-D coordinates so that its normal vector is aligned
    with a target vector.

    Inspired by https://math.stackexchange.com/a/476311.

    Parameters
    ----------
    array : [N x 3] np.ndarray (float)
        Array of 3-D coordinates to be rotated.
    targetVector : [3 x 1] np.ndarray (float)
        Target vector to which the array's normal vector should be aligned.

    Returns
    -------
    array : [N x 3] np.ndarray (float)
        Rotated array of 3-D coordinates.
    """
    if targetVector is None:
        # Compute array center
        targetVector = np.mean(array, axis=0)
    # Reset array center
    array = array - targetVector
    
    # Compute rotation matrix
    a = np.array([0, 0, 1])  # vector normal to my current plane
    b = targetVector / np.linalg.norm(targetVector)  # vector normal to my target plane
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    identityMat = np.eye(3)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    rotMat = identityMat + vx + np.dot(vx, vx) * (1 - c) / s**2

    # Apply rotation matrix
    array = np.dot(rotMat, array.T).T

    # Recentre array
    array = array + targetVector
    return array