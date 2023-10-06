import os
import re
import copy
import yaml
import librosa
import resampy
import numpy as np
import pickle, gzip
from . import classes
from numba import njit
import scipy.signal as sig
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rot
from pyANFgen.pyanfgen.utils import pyanfgen, ANFgenConfig


def build_scenario(p: classes.WASNparameters):
    """
    Interprets parameters to decide whether to simulate an actual room,
    or to generate random impulse responses.

    
    vad : [N x Nnodes x Nsources] np.ndarray (bool or int [0 or 1])
        VAD per sample, per node, and per speech source.
    wetSpeeches : [Nnodes x 1] list of [Nm[k] x N] np.ndarray (float)
        Wet (RIR-affected) speech signal at each sensor of each node.
        `Nm[k]` is the number of microphones at node `k`.
    wetNoises : [Nnodes x 1] list of [Nm[k] x N] np.ndarray (float)
        Wet (RIR-affected) noise signal at each sensor of each node.
    """

    # Get impulse responses
    if p.trueRoom:
        room, irs_d, irs_n = build_room(p)
    else:
        irs_d, irs_n = generate_random_impulse_responses(p)
        room = None
    
    # Get raw source signals
    dRaw, nRaw = get_raw_source_signals(p, irs_d, irs_n)
    
    # Get wet speech and compute VAD
    vad, wetSpeeches = get_vad(
        irs_d,
        dRaw,
        p
    )
    if p.nNoiseSources > 0:
        # Get wet noise
        _, wetNoises = get_vad(
            irs_n,
            nRaw,
            p,
            bypassVADcomputation=True  # save computation time
        )
        if 'mic' in p.snrBasis:
            # Verify that correct SNR is obtained at reference mic
            #  -- NB: computation only valid for a single noise source.  # TODO: generalize
            # Extract reference (network-wide) microphone index
            refMicIdx = int(re.findall("\d+", p.snrBasis)[0])
            # Find corresponding node index
            refNodeIdx = np.where(p.sensorToNodeIndices == refMicIdx)[0][0]
            # Find corresponding local sensor index (at node `refNodeIdx`)
            refMicIdxLocal = int(
                refMicIdx -np.sum(p.nSensorPerNode[:refNodeIdx])
            )
            # Compute SNR based on wet speech and wet noise
            currSNR = 10 * np.log10(
                np.sum(wetSpeeches[refNodeIdx][refMicIdxLocal, :] ** 2) / \
                    np.sum(wetNoises[refNodeIdx][refMicIdxLocal, :] ** 2)
            )
            if np.round(currSNR) == p.snr:
                print(f'Correct SNR ({currSNR} dB ~= {int(np.round(currSNR))} dB) obtained at reference microphone.')
            else:
                print(f'Incorrect SNR ({currSNR} dB ~= {int(np.round(currSNR))} dB) obtained at reference microphone.')
    else:
        wetNoises = None

    return room, vad, wetSpeeches, wetNoises


# def apply_rirs(signals: np.ndarray, rirs: list) -> np.ndarray:
#     """
#     Apply RIRs to signals.

#     Parameters
#     ----------
#     signals : [N x Nsources] np.ndarray (float)
#         Signals to which to apply RIRs.
#     rirs : [Nnodes x 1] list of [Nm[k] x 1] lists of [Nsource x N] np.ndarray (float)
#         RIRs at each sensor of each node, for each source.
#         `Nm[k]` is the number of microphones at node `k`.

#     Returns
#     -------
#     signalsOut : [Nnodes x 1] list of [N x Nm[k]] np.ndarray (float)
#     """

#     # Apply RIRs
#     signalsOut = []
#     for k in range(len(rirs)):  # loop over nodes
#         # Loop over sensors
#         signalsOut.append(np.zeros((signals.shape[0], len(rirs[k]))))
#         for ii in range(len(rirs[k])):  # loop over sensors
#             # Loop over sources
#             for jj in range(len(rirs[k][ii])):  # loop over sources
#                 # Apply RIR
#                 currSourceContribution = sig.lfilter(rirs[k][ii][jj], 1, signals[:, jj])
#                 # Add to output
#                 if jj == 0:
#                     signalsOut[k][:, ii] = currSourceContribution
#                 else:
#                     signalsOut[k][:, ii] += currSourceContribution

#     return signalsOut


def get_raw_source_signals(
        p: classes.WASNparameters,
        irs_d: list=None,
        irs_n: list=None
    ):
    """
    Obtain raw (unprocessed) source signals.

    Parameters
    ----------
    p : `WASNparameters` object
        Parameters.
    irs_d : [Nnodes x 1] list of [Nm[k] x 1] lists of [Ndesired x 1] lists of [N x 1] np.ndarray (float)
        RIRs from desired sources at each sensor of each node.
        `Nm[k]` is the number of microphones at node `k`.
    irs_n : [Nnodes x 1] list of [Nm[k] x 1] lists of [Nnoise x 1] lists of [N x 1] np.ndarray (float)
        RIRs from noise sources at each sensor of each node.

    Returns
    -------
    desiredSignalsRaw : [N x Ndesired] np.ndarray (float)
        Raw (unprocessed) desired source signals.
    noiseSignalsRaw : [N x Nnoise] np.ndarray (float)
        Raw (unprocessed) noise source signals.
    """

    def _get_source_signal(file):
        """Helper function to load and process source signal."""
        # Load
        y, fsOriginal = librosa.load(file, sr=None)
        # Resample
        if fsOriginal != p.fs:
            print(f'Resampling {file} from {fsOriginal} Hz to {p.fs} Hz...')
            y = resampy.resample(y, fsOriginal, p.fs)
        # Adjust length
        if len(y) > desiredNumSamples:
            y = y[:desiredNumSamples]
        elif len(y) < desiredNumSamples:
            while len(y) < desiredNumSamples:
                y = np.concatenate((y, y))  # loop
                y = y[:desiredNumSamples]
        # Whiten
        # y = (y - np.mean(y)) / np.std(y)  # whiten [commented out by PD on 2023.09.07 (see journal week36 THU)]
        y = y / np.std(y)  # normalie
        return y

    # Desired number of samples
    desiredNumSamples = int(p.sigDur * p.fs)

    # Add desired sources
    desiredSignalsRaw = np.zeros((desiredNumSamples, p.nDesiredSources))
    for ii in range(p.nDesiredSources):
        if p.signalType == 'from_file':
            # Load sound file
            y = _get_source_signal(p.desiredSignalFile[ii])
        elif p.signalType == 'random':
            # Generate random signal
            y = generate_rand_signal(
                p.randSignalsParams,
                desiredNumSamples,
                p.fs
            )
        # Remove DC offset
        desiredSignalsRaw[:, ii] = y

    # Add noise sources
    noiseSignalsRaw = np.zeros((desiredNumSamples, p.nNoiseSources))
    for ii in range(p.nNoiseSources):
        if p.signalType == 'from_file':
            # Load sound file
            n = _get_source_signal(p.noiseSignalFile[ii])
        elif p.signalType == 'random':
            # Generate random signal
            n = generate_rand_signal(
                p.randSignalsParams,
                desiredNumSamples,
                p.fs,
                noPauses=True  # no pauses for noise signals
            )
        if p.snrBasis == 'dry_signals':
            # Set SNR based on dry signals
            n *= 10 ** (-p.snr / 20)    # gain to set SNR
        elif 'mic' in p.snrBasis:
            # Extract reference (network-wide) microphone index
            refMicIdx = int(re.findall("\d+", p.snrBasis)[0])
            # Find corresponding node index
            refNodeIdx = np.where(p.sensorToNodeIndices == refMicIdx)[0][0]
            # Find corresponding local sensor index (at node `refNodeIdx`)
            refMicIdxLocal = int(
                refMicIdx -np.sum(p.nSensorPerNode[:refNodeIdx])
            )
            # Extract relevant RIRs
            irs_d_refMic = irs_d[refNodeIdx][refMicIdxLocal]  # [Ndesired x 1] list of [N x 1] np.ndarray (float)
            ir_n_refMic = irs_n[refNodeIdx][refMicIdxLocal][ii]  # [N x 1] np.ndarray (float)
            # Compute wet signals (target and noise) at reference microphone
            y_refMic = np.zeros((desiredNumSamples,))
            for jj, ir in enumerate(irs_d_refMic):  # loop over desired sources
                y_refMic += sig.lfilter(ir, 1, desiredSignalsRaw[:, jj])
            n_refMic = sig.lfilter(ir_n_refMic, 1, n)
            # Compute current SNR, just due to RIRs
            currSNR = 10 * np.log10(
                np.sum(y_refMic ** 2) / np.sum(n_refMic ** 2)
            )
            # Set dry sources SNR based on current wet signals SNR
            n *= 10 ** ((currSNR - p.snr) / 20)
        noiseSignalsRaw[:, ii] = n

    return desiredSignalsRaw, noiseSignalsRaw


def generate_random_impulse_responses(p: classes.WASNparameters):
    """
    Generate random impulse responses.

    Parameters
    ----------
    p : `WASNparameters` object
        Parameters.

    Returns
    -------
    rirsDesiredSources : [Nnodes x 1] list of [Nm[k] x 1] lists of
                        [Ndesired x 1] lists of [N x 1] np.ndarray (float)
        RIRs from desired sources at each sensor of each node.
        `Nm[k]` is the number of microphones at node `k`.
    rirsNoiseSources : [Nnodes x 1] list of [Nm[k] x 1] lists of
                        [Nnoise x 1] lists of [N x 1] np.ndarray (float)
        RIRs from noise sources at each sensor of each node.
    """

    # Generate random impulse responses
    rirsDesiredSources = []
    rirsNoiseSources = []
    for k in range(p.nNodes):
        currNodeRirsDesiredSources = []
        currNodeRirsNoiseSources = []
        for _ in range(p.nSensorPerNode[k]):
            # Generate desired source RIRs
            currNodeRirsDesiredSources.append(
                generate_random_rir(
                    p.randIRsParams,
                    p.nDesiredSources,
                    p.fs
                )
            )
            # Generate noise source RIRs
            currNodeRirsNoiseSources.append(
                generate_random_rir(
                    p.randIRsParams,
                    p.nNoiseSources,
                    p.fs
                )
            )
        rirsDesiredSources.append(currNodeRirsDesiredSources)
        rirsNoiseSources.append(currNodeRirsNoiseSources)
    
    return rirsDesiredSources, rirsNoiseSources


def generate_random_rir(
        prir: classes.RandomIRParameters,
        n: int,
        fs: float
    ) -> np.ndarray:
    # Generate random RIRs
    rirs = []
    for _ in range(n):
        # Generate random RIR
        if prir.distribution == 'uniform':
            rir = np.random.uniform(
                prir.minValue,
                prir.maxValue,
                (int(prir.duration * fs),)
            )
        elif prir.distribution == 'normal':
            rir = np.random.normal(
                0,
                np.amax([np.abs(prir.maxValue), np.abs(prir.minValue)]),
                (int(prir.duration * fs),)
            )
        # Add exponential decay if asked
        if prir.decay == 'exponential':
            rir *= np.exp(-np.arange(len(rir)) /\
                (prir.decayTimeConstant * fs))
        elif prir.decay == 'immediate':
            rir[1:] = 0  # turns into a Dirac
            rir[0] = np.abs(rir[0])  # make sure the Dirac is positive
        #
        rirs.append(rir)
    return rirs


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
    rirsDesiredSources : [Nnodes x 1] list of [Nm[k] x 1] lists of
                    [Ndesired x 1] lists of [N x 1] np.ndarray (float)
        RIRs from desired sources at each sensor of each node.
        `Nm[k]` is the number of microphones at node `k`.
    rirsNoiseSources : [Nnodes x 1] list of [Nm[k] x 1] lists of
                    [Nnoise x 1] lists of [N x 1] np.ndarray (float)
        RIRs from noise sources at each sensor of each node.
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
    
    if p.layoutType == 'predefined':
        layoutInfo = load_layout_from_yaml(p.predefinedLayoutFile)
        p.align_with_loaded_yaml_layout(layoutInfo)

        if any(np.array(layoutInfo['rd']) != p.rd.astype(float)):
            raise ValueError('Room dimensions do not match.')
        
        # Extract sensor coordinates
        sensorsCoords = []
        for k in range(len(layoutInfo['Mk'])):
            # Generate node and sensors
            sensorsCoords.append(layoutInfo['sensorCoords'][
                int(np.sum(layoutInfo['Mk'][:k])):int(np.sum(layoutInfo['Mk'][:k + 1]))
            ])
        # Flatten list
        sensorsCoords = np.concatenate(sensorsCoords, axis=0).T

        # Extract desired sources coordinates
        desiredSourceCoords = np.array(layoutInfo['targetCoords']).T

        # Extract noise sources coordinates
        noiseSourceCoords = np.array(layoutInfo['interfererCoords']).T

    elif p.layoutType == 'all_nodes_in_center':
        # Compute sensor coordinates
        sensorsCoords = []
        for k in range(p.nNodes):
            # Generate node and sensors
            r = np.array([p.rd[0] / 2, p.rd[1] / 2, p.rd[2] / 2])
            sensorsCoords.append(generate_array_pos(
                r,
                p.nSensorPerNode[k],
                p.arrayGeometry,
                p.interSensorDist,
                applyRandomRot=False
            ))
        # Flatten list
        sensorsCoords = np.concatenate(sensorsCoords, axis=0).T
            
        # Compute desired sources coordinates
        desiredSourceCoords = np.random.uniform(size=(3, p.nDesiredSources)) *\
            (p.rd[:, np.newaxis] - 2 * p.minDistToWalls) + p.minDistToWalls

        # Compute noise sources coordinates
        noiseSourceCoords = np.random.uniform(size=(3, p.nNoiseSources)) *\
            (p.rd[:, np.newaxis] - 2 * p.minDistToWalls) + p.minDistToWalls\
        
    elif p.layoutType == 'random':
        # Compute sensor coordinates
        sensorsCoords = []
        for k in range(p.nNodes):
            # Generate node and sensors
            r = np.random.uniform(size=(3,)) * (p.rd - 2 * p.minDistToWalls)\
                + p.minDistToWalls # node centre coordinates
            sensorsCoords.append(generate_array_pos(
                r,
                p.nSensorPerNode[k],
                p.arrayGeometry,
                p.interSensorDist,
                applyRandomRot=False
            ))
        # Flatten list
        sensorsCoords = np.concatenate(sensorsCoords, axis=0).T
            
        # Compute desired sources coordinates
        desiredSourceCoords = np.random.uniform(size=(3, p.nDesiredSources)) *\
            (p.rd[:, np.newaxis] - 2 * p.minDistToWalls) + p.minDistToWalls

        # Compute noise sources coordinates
        noiseSourceCoords = np.random.uniform(size=(3, p.nNoiseSources)) *\
            (p.rd[:, np.newaxis] - 2 * p.minDistToWalls) + p.minDistToWalls
        #
    elif 'spinning_top' in p.layoutType:  # Spinning top layout

        if 'random' in p.layoutType:
            # Generate a random line for the sources
            azimuthLine = np.random.uniform(np.pi / 8, np.pi / 2 - np.pi / 8)
            elevationLine = np.random.uniform(np.pi / 8, np.pi / 2 - np.pi / 8)
            maxR = np.sqrt(np.sum(p.rd ** 2))  # room diagonal length
        elif 'vert' in p.layoutType:
            # Set a vertical line for the sources
            azimuthLine = np.pi / 2
            elevationLine = 0
            maxR = p.rd[0]  # room width

        # Generate a random line offset with respect to the room floor
        if p.spinTop_minInterNodeDist is not None:
            circRmin = copy.deepcopy(p.spinTop_minInterNodeDist)
        else:
            circRmin = 2 * p.interSensorDist * np.amax(p.nSensorPerNode)

        # Generate node and sensor positions
        validNodePos = False
        attemptsCount = 0
        maxNumAttempts = 999  # <-- arbitrary... [PD06.04.2023]
        while not validNodePos:
            if attemptsCount > maxNumAttempts:
                raise ValueError("Could not find valid node positions.")
            print('Spinning-top layout creation -- attempt #' + str(attemptsCount + 1))
            
            # Set a random spinning top axis center   
            xOffset = np.random.normal(
                p.minDistToWalls + circRmin, p.rd[0] - circRmin - p.minDistToWalls
            )
            yOffset = np.random.normal(
                p.minDistToWalls + circRmin, p.rd[1] - circRmin - p.minDistToWalls
            )

            # Generate a random line for the sources
            azimuthLine = np.random.uniform(np.pi / 8, np.pi / 2 - np.pi / 8)
            elevationLine = np.random.uniform(np.pi / 8, np.pi / 2 - np.pi / 8)
            
            # Generate speech sources along the line
            desiredSourceCoords = random_points_on_line(
                azimuthLine,
                elevationLine,
                xOffset,
                yOffset,
                p.rd,
                n=p.nDesiredSources,
                minSpacing=p.spinTop_minSourceSpacing
            )
            if desiredSourceCoords is None:
                attemptsCount += 1
                continue
            desiredSourceCoords = desiredSourceCoords.T  # adapt for pyroomacoustics

            # Generate noise sources along the line
            noiseSourceCoords = random_points_on_line(
                azimuthLine,
                elevationLine,
                xOffset,
                yOffset,
                p.rd,
                n=p.nNoiseSources,
                minSpacing=p.spinTop_minSourceSpacing
            )
            if noiseSourceCoords is None:
                attemptsCount += 1
                continue
            noiseSourceCoords = noiseSourceCoords.T  # adapt for pyroomacoustics
            
            # Generate random circle center
            circCenter = random_points_on_line(
                azimuthLine,
                elevationLine,
                xOffset,
                yOffset,
                p.rd
            )
            circCenter = np.squeeze(circCenter)
            if circCenter is None:
                attemptsCount += 1
                continue
            # Compute maximal circle radius based on room dimension
            if 'random' in p.layoutType:
                refDistCenterCircle = np.sqrt(
                    np.sum(np.array(circCenter) ** 2)
                )  # distance from circle center to room origin
            elif 'vert' in p.layoutType:
                circCenter[2] = 0
                refDistCenterCircle = np.sqrt(
                    np.sum(np.array(circCenter) ** 2)
                )  # distance from circle center to room origin
            
            if refDistCenterCircle < maxR / 2:
                circRmax = (refDistCenterCircle - p.minDistToWalls) / 3  # <-- a little arbitrary... [PD06.04.2023]
            else:
                circRmax = (maxR - refDistCenterCircle - p.minDistToWalls) / 3
            # Generate random circle radius
            if circRmax < circRmin:
                circR = circRmin
            else:
                circR = np.random.uniform(circRmin, circRmax)

            sensorsCoords = np.zeros((np.sum(p.nSensorPerNode), 3))
            # Generate array centers equally spaced on the circle
            angleArrayCenters = np.arange(0, 2 * np.pi, 2 * np.pi / p.nNodes)
            for k in range(p.nNodes):
                # Generate node and sensors on the perimeter of the circle
                r = angleArrayCenters[k]
                x = circCenter[0] + circR * np.cos(r)
                y = circCenter[1] + circR * np.sin(r)
                z = circCenter[2]
                # If asked, add random wiggle to the array centers
                if p.spinTop_randomWiggleAmount != 0:
                    print('Adding random wiggle to array centers... (amount = +/-' + str(p.spinTop_randomWiggleAmount) + ' m)')
                    x += np.random.uniform(
                        -p.spinTop_randomWiggleAmount,
                        p.spinTop_randomWiggleAmount
                    )
                    y += np.random.uniform(
                        -p.spinTop_randomWiggleAmount,
                        p.spinTop_randomWiggleAmount
                    )
                    z += np.random.uniform(
                        -p.spinTop_randomWiggleAmount,
                        p.spinTop_randomWiggleAmount
                    )
                idxSensor = int(np.sum(p.nSensorPerNode[:k]))
                sensorsCoords[idxSensor:(idxSensor + p.nSensorPerNode[k]), :] =\
                    generate_array_pos(
                    np.array([x, y, z]),
                    p.nSensorPerNode[k],
                    p.arrayGeometry,
                    p.interSensorDist,
                    applyRandomRot=True,
                    )
            # Rotate coordinates so that the circle is perpendicular
            # to the source line.
            if 'random' in p.layoutType:
                sensorsCoords = rotate_array_yx(
                    sensorsCoords,
                    targetVector=np.array(circCenter)
                )
            
            # Check that all microphones and sources are in the room
            if np.any(sensorsCoords > p.rd - p.minDistToWalls) or\
                np.any(sensorsCoords < p.minDistToWalls) or\
                np.any(desiredSourceCoords > p.rd - p.minDistToWalls) or\
                np.any(desiredSourceCoords < p.minDistToWalls) or\
                np.any(noiseSourceCoords > p.rd - p.minDistToWalls) or\
                np.any(noiseSourceCoords < p.minDistToWalls):
                validNodePos = False
                attemptsCount += 1
            else:
                validNodePos = True
        
        sensorsCoords = sensorsCoords.T  # adapt for `add_sensors_and_sources_to_room`
        
    # Add sensors and sources to room
    # room, desiredSignalsRaw, noiseSignalsRaw = add_sensors_and_sources_to_room(
    room = add_sensors_and_sources_to_room(
        sensorsCoords=sensorsCoords,
        desiredSourceCoords=desiredSourceCoords,
        noiseSourceCoords=noiseSourceCoords,
        p=p,
        room=room,
    )

    # Compute RIRs
    room.compute_rir()
    # room.simulate()

    # # Truncate signals (no need for reverb tail)
    # room.mic_array.signals = room.mic_array.signals[:, :int(p.fs * p.sigDur)]

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

    return room, rirsDesiredSources, rirsNoiseSources


def add_sensors_and_sources_to_room(
        sensorsCoords: np.ndarray,
        desiredSourceCoords: np.ndarray,
        noiseSourceCoords: np.ndarray,
        p: classes.WASNparameters,
        room: pra.room.ShoeBox
    ) -> pra.room.ShoeBox:
    """
    Add sensors and sources (desired and noise) to Pyroomacoustics room
    object from previously computed coordinates.

    Parameters
    ----------
    sensorsCoords : [3 x Nsensors] np.ndarray (float)
        Sensor coordinates [m].
    desiredSourceCoords : [3 x Ndesired] np.ndarray (float)
        Desired source coordinates [m].
    noiseSourceCoords : [3 x Nnoise] np.ndarray (float)
        Noise source coordinates [m].
    p : WASNparameters object
        Parameters.
    room : pyroomacoustics.room.ShoeBox object
        Room object.

    Returns
    -------
    room : pyroomacoustics.room.ShoeBox object
        Room object.
    desiredSignalsRaw : [N x Ndesired] np.ndarray (float)
        Raw (unprocessed) desired source signals.
    noiseSignalsRaw : [N x Nnoise] np.ndarray (float)
        Raw (unprocessed) noise source signals.
    """
    
    # Add nodes
    room.add_microphone_array(sensorsCoords)

    # Desired number of samples
    desiredNumSamples = int(p.sigDur * p.fs)

    # Add desired sources
    for ii in range(p.nDesiredSources):
        ssrc = pra.soundsource.SoundSource(
            position=desiredSourceCoords[:, ii], # coordinates
            signal=np.zeros(desiredNumSamples)  # placeholder
        )
        room.add_soundsource(ssrc)

    # Add noise sources
    for ii in range(p.nNoiseSources):
        ssrc = pra.soundsource.SoundSource(
            position=noiseSourceCoords[:, ii], # coordinates
            signal=np.zeros(desiredNumSamples)  # placeholder
        )
        room.add_soundsource(ssrc)

    return room


def generate_rand_signal(
        prand: classes.RandomSignalsParameters,
        n: int,
        fs: float,
        noPauses=False
    ) -> np.ndarray:
    """
    Generate random signal.

    Parameters
    ----------
    prand : RandomSignalsParameters object
        Parameters.
    n : int
        Desired number of samples.
    fs : float
        Sampling frequency [Hz].
    noPauses : bool
        If True, do not add pauses.

    Returns
    -------
    y : [N x 1] np.ndarray (float)
        Random signal.
    """
    if prand.distribution == 'uniform':
        y = np.random.uniform(
            prand.minValue,
            prand.maxValue,
            (n,)
        )
    elif prand.distribution == 'normal':
        y = np.random.normal(
            0,
            np.amax([np.abs(prand.maxValue), np.abs(prand.minValue)]),
            (n,)
        )        

    # Add pauses
    if not noPauses:
        if prand.pauseType == 'random':
            # Compute the number of pauses based on the desired
            # min/max pause length
            nPauses = int(
                n / ((prand.randPauseDuration_min +\
                    prand.randPauseDuration_max) * fs) / 2
            )
            # Add random pauses
            for ii in range(nPauses):
                # Generate random pause length
                leng = np.random.uniform(
                    prand.randPauseDuration_min,
                    prand.randPauseDuration_max
                )
                # Generate random pause position
                pausePos = np.random.uniform(0, n - leng * fs)
                # Add pause
                y[int(pausePos):int(pausePos + leng * fs)] = 0

        elif prand.pauseType == 'predefined':
            # Add regular pauses based on the desired
            # pause length and pause spacing
            leng = prand.pauseDuration * fs
            space = prand.pauseSpacing * fs
            if prand.startWithPause:
                startIdx = 0
            else:
                startIdx = space
            pausePos = np.arange(startIdx, n, space + leng)
            for ii in range(len(pausePos)):
                y[int(pausePos[ii]):int(pausePos[ii] + leng)] = 0

    return y


def plot_asc_3d(
        ax,
        room,
        p: classes.WASNparameters,
        cx=None,
        cy=None,
        cz=None,
        plotSourcesLine=False
    ):
    """
    Plot room, nodes, and sources in 3D.
    
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


def get_vad(
        rirs,
        xdry: np.ndarray,
        p: classes.WASNparameters,
        bypassVADcomputation=False
    ):
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
    p : WASNparameters object
        WASN parameters.
    bypassVADcomputation : bool
        If True, bypass VAD computation and return None instead of `vad`.

    Returns
    -------
    vad : [N x Nnodes x Nsources] np.ndarray (bool or int [0 or 1])
        VAD per sample, per node, and per speech source.
    wetsigs : [K x 1] list of [Nsensor[k] x N] np.ndarray (float)
        Wet (RIR-affected) speech (or noise) signal at each sensor of each node.
    """

    vad = np.zeros((xdry.shape[0], len(rirs), len(rirs[0][0])))
    wetsigs = [np.zeros((len(rirs[k]), xdry.shape[0], len(rirs[k][0])))\
        for k in range(len(rirs))]

    for k in range(len(rirs)):  # for each node
        for m in range(len(rirs[k])):  # for each microphone
            for ii in range(len(rirs[k][m])):  # for each desired source
                # Compute wet desired-only signal
                wetsig = sig.fftconvolve(xdry[:, ii], rirs[k][m][ii], axes=0)
                wetsigs[k][m, :, ii] = wetsig[:xdry.shape[0]]  # truncate

        for ii in range(len(rirs[k][p.referenceSensor])):  # for each desired source
            if bypassVADcomputation:
                vad = None
            else:
                # Inform user
                print(f'Computing/loading VAD for node {k + 1}/{len(rirs)} and desired source {ii + 1}/{len(rirs[k][p.referenceSensor])}...')
                vad[:, k, ii] = get_or_load_vad(
                    x=wetsigs[k][p.referenceSensor, :, ii],
                    eFact=p.VADenergyFactor,
                    Nw=p.VADwinLength,
                    fs=p.fs,
                    loadIfPossible=p.enableVADloadFromFile,
                    vadFilesFolder=p.vadFilesFolder
                )
    
    # Sum wet signals over sources
    wetsigs = [np.sum(wetsig, axis=-1) for wetsig in wetsigs]

    return vad, wetsigs


def get_or_load_vad(x, eFact, Nw, fs, loadIfPossible=True, vadFilesFolder='.'):
    """
    Compute or load VAD.

    Parameters
    ----------
    x : [N x 1] np.ndarray (float)
        Signal.
    eFact : float
        Energy factor.
    Nw : int
        Window length in samples.
    fs : float
        Sampling frequency in Hz.
    loadIfPossible : bool
        If True, try to load VAD from file.
    vadFilesFolder : str
        Folder where VAD files are stored.

    Returns
    -------
    vad : [N x 1] np.ndarray (bool or int [0 or 1])
        VAD.
    """
    # Compute VAD threshold
    thrsVAD = np.amax(x ** 2) / eFact
    # Compute VAD filename
    vadFilename = f'{vadFilesFolder}/vad_{array_id(x)}_thrs_{dot_to_p(np.round(thrsVAD, 3))}_Nw_{dot_to_p(np.round(Nw, 3))}_fs_{dot_to_p(fs)}.npy'
    # Check if VAD can be loaded from file
    if loadIfPossible and os.path.isfile(vadFilename):
        # Load VAD from file
        vad = np.load(vadFilename)
    else:
        # Compute VAD
        vad, _ = oracleVAD(
            x,
            tw=Nw,
            thrs=thrsVAD,
            Fs=fs
        )
        np.save(vadFilename, vad)
    return vad


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
        applyRandomRot=False,
    ):
    """
    Define node positions based on node position, number of nodes,
    and array type.

    Parameters
    ----------
    nodeCoords : [J x 3] array of real floats
        Nodes coordinates in 3-D space [m].
    Mk : int
        Number of microphones.
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



def oracleVAD(x, tw, thrs, Fs):
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
        Nw = int(tw * Fs)
    else:
        Nw = 1

    # Compute VAD
    oVAD = np.zeros(n)
    for ii in range(n):
        # Extract chunk
        idxBeg = int(np.amax([ii - Nw // 2, 0]))
        idxEnd = int(np.amin([ii + Nw // 2, len(x)]))
        # Compute VAD frame
        oVAD[ii] = compute_VAD(x[idxBeg:idxEnd], thrs)

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
    energy = np.mean(np.abs(chunk_x)**2)
    # Assign VAD value
    if energy > thrs:
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
        startComputeMetricsAt: str,
        minNoSpeechDurEndUtterance: float,
        setThoseSensorsToNoise: list=[],
        endComputeMetricsAt: str=None,
    ):
    """
    Build WASN from parameters (including asynchronicities and topology).
    
    Parameters
    ----------
    room : [augmented] `pyroomacoustics.room.ShoeBox` object
        Room (acoustic scenario) object. Augmented with VAD and 
        wet speech signals at each node's reference sensor (output of
        `build_room()`).
        NB: is `None` if `p.trueRoom` is `False`.
    vad : [N x Nnodes x Nspeechsources] np.ndarray (bool or int [0 or 1])
        VAD per sample, per node, and per speech source.
    wetSpeeches : [Nnodes x 1] list of [Nm[k] x N] np.ndarray (float)
        Wet (RIR-affected) speech signal at each sensor of each node.
        `Nm[k]` is the number of microphones at node `k`.
    wetNoises : [Nnodes x 1] list of [Nm[k] x N] np.ndarray (float)
        Wet (RIR-affected) noise signal at each sensor of each node.
    p : `WASNparameters` object
        WASN parameters
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
        Minimum duration of silence at the end of an utterance [s].
    setThoseSensorsToNoise : list
        List of sensors to set to noise (e.g., for debugging purposes).
    endComputeMetricsAt : str
        Time at which to end computing metrics. If `None`, compute metrics
        until the end of the simulation.

    Returns
    -------
    myWASN : `WASN` object
        WASN object, including all necessary values.
    """

    # Account for the `trueRoom == False` case
    if not p.trueRoom:
        sensorCoords = None
    else:
        sensorCoords = room.mic_array.R
    
    # Get sensor signals
    sensorSignals = np.zeros(
        (np.sum(p.nSensorPerNode), wetSpeeches[0].shape[-1])
    )
    for k in range(p.nNodes):
        if wetNoises is not None:  # if there are localized noise sources
            sensorSignals[p.sensorToNodeIndices == k, :] =\
                wetNoises[k] + wetSpeeches[k]
        else:
            sensorSignals[p.sensorToNodeIndices == k, :] =\
                wetSpeeches[k]

    # Create network topological map (inter-node links)
    adjacencyMatrix, neighbors = get_topo(
        p.topologyParams,
        sensorToNodeIndices=p.sensorToNodeIndices,
        sensorCoords=sensorCoords  # microphones positions
    )

    myWASN = classes.WASN()
    # If asked, compute diffuse noise
    if p.diffuseNoise:
        # Compute inter-sensor distance matrix
        totalNumSensors = np.sum(p.nSensorPerNode)
        interSensorDists = np.zeros((totalNumSensors, totalNumSensors))
        for m1 in range(totalNumSensors):
            for m2 in range(totalNumSensors):
                interSensorDists[m1, m2] = np.sqrt(np.sum(
                    (sensorCoords[:, m1] - sensorCoords[:, m2]) ** 2
                ))
        # Generate diffuse noise for each sensor
        cfg = ANFgenConfig(
            fs=p.fs,
            M=totalNumSensors,
            d=interSensorDists,
            nfType='spherical' if len(p.rd) == 3 else 'circular',
            sigType=p.typeDiffuseNoise,
            babbleFile=p.fileDiffuseBabble,
            T=p.sigDur,
        )
        diffuseNoise = pyanfgen(cfg, plot=False)
        # Normalize
        diffuseNoise /= np.amax(diffuseNoise)
        # Apply desired power factor
        diffuseNoise *= 10 ** (p.diffuseNoisePowerFactor / 20)

    for k in range(p.nNodes):
        # Apply asynchronicities
        sigs, t, fsSRO = apply_asynchronicity_at_node(
            y=sensorSignals[p.sensorToNodeIndices == k, :].T,
            fs=p.fs,
            sro=p.SROperNode[k],
            sto=0.
        )
        sigs_noSROs = sensorSignals[p.sensorToNodeIndices == k, :].T
        # If asked, add diffuse noise
        if p.diffuseNoise:
            sigs += diffuseNoise[:, p.sensorToNodeIndices == k]
            sigs_noSROs	+= diffuseNoise[:, p.sensorToNodeIndices == k]
        # Apply microphone self-noise
        selfNoise = np.zeros_like(sigs)
        for m in range(sigs.shape[-1]):
            sigs[:, m], selfNoise[:, m] = apply_self_noise(
                sigs[:, m], p.selfnoiseSNR
            )
            sigs_noSROs[:, m], _ = apply_self_noise(
                sigs_noSROs[:, m], p.selfnoiseSNR
            )
        selfNoiseRefSensor = selfNoise[:, 0]

        # vvvvvv Speech-only signals vvvvvv
        # Apply asynchronicities also to speech-only signal
        speechOnly, _, _ = apply_asynchronicity_at_node(
            y=wetSpeeches[k].T,
            fs=p.fs,
            sro=p.SROperNode[k],
            sto=0.
        )
        speechOnly_noSROs = wetSpeeches[k].T

        # vvvvvv Noise-only signals vvvvvv
        if p.nNoiseSources > 0:
            # Apply asynchronicities also to noise-only signal
            noiseOnlyWithoutAsync = wetNoises[k].T
        else:
            noiseOnlyWithoutAsync = np.zeros_like(speechOnly)
        if p.diffuseNoise:
            # Add diffuse noise to signals
            noiseOnlyWithoutAsync += diffuseNoise[:, p.sensorToNodeIndices == k]
        noiseOnly, _, _ = apply_asynchronicity_at_node(
            y=noiseOnlyWithoutAsync,
            fs=p.fs,
            sro=p.SROperNode[k],
            sto=0.
        )
        noiseOnly_noSROs = noiseOnlyWithoutAsync + selfNoiseRefSensor[:, np.newaxis]
        # Add same microphone self-noise
        noiseOnly += selfNoiseRefSensor[:, np.newaxis]

        # Get geometrical parameters
        if p.trueRoom:
            sensorPositions = sensorCoords[:, p.sensorToNodeIndices == k]
            nodePosition = np.mean(sensorPositions, axis=1)
        else:
            sensorPositions = None
            nodePosition = None
        
        # Create node
        node = classes.Node(
            index=k,
            nSensors=p.nSensorPerNode[k],
            refSensorIdx=p.referenceSensor,
            sro=p.SROperNode[k],
            fs=fsSRO,
            data=sigs,
            data_noSRO=sigs_noSROs,
            cleanspeech=speechOnly,
            cleanspeech_noSRO=speechOnly_noSROs,
            cleannoise=noiseOnly,
            cleannoise_noSRO=noiseOnly_noSROs,
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
    
    # Establish start/end times for the computation of speech enhancement
    # metrics based on the speech signal used (after 1 speech utterance -->
    # whenever the VAD has gone up and down).
    myWASN.get_metrics_key_time(
        ref=startComputeMetricsAt,
        minNoSpeechDurEndUtterance=minNoSpeechDurEndUtterance,
        timeType='start'
    )
    myWASN.get_metrics_key_time(
        ref=endComputeMetricsAt,
        minNoSpeechDurEndUtterance=minNoSpeechDurEndUtterance,
        timeType='end'
    )

    # Render specific sensors useless by replacing their signal
    # with random noise, if asked.
    if any(setThoseSensorsToNoise):
        print('Replacing signal at sensors with Python indices', setThoseSensorsToNoise, 'with random noise.')
        for m in setThoseSensorsToNoise:
            # Find node index of that sensor
            k = p.sensorToNodeIndices[m]
            # Find local microphone index of that sensor
            mLocal = np.where(p.sensorToNodeIndices == k)[0].tolist().index(m)
            # Replace signal with random noise
            myWASN.wasn[k].data[:, mLocal] = np.random.uniform(
                -1, 1, size=myWASN.wasn[k].data[:, mLocal].shape
            )
            # Replace clean speech with random noise
            myWASN.wasn[k].cleanspeech[:, mLocal] = np.random.uniform(
                -1, 1, size=myWASN.wasn[k].cleanspeech[:, mLocal].shape
            )
            # Udpate clean speech at reference sensor
            myWASN.wasn[k].cleanspeechRefSensor =\
                myWASN.wasn[k].cleanspeech[:, myWASN.wasn[k].refSensorIdx]
            # Replace clean noise with random noise
            myWASN.wasn[k].cleannoise[:, mLocal] = np.random.uniform(
                -1, 1, size=myWASN.wasn[k].cleannoise[:, mLocal].shape
            )
            # Udpate clean noise at reference sensor
            myWASN.wasn[k].cleannoiseRefSensor =\
                myWASN.wasn[k].cleannoise[:, myWASN.wasn[k].refSensorIdx]
    
    # If asked, add random-noise (unusable) signals to the nodes
    for ii, nRandNoiseSigs in enumerate(p.addedNoiseSignalsPerNode):
        print('Adding', nRandNoiseSigs, 'random-noise signals to node', ii, '...')
        # Create random-noise (unusable) signal
        sig = np.random.uniform(
            -1, 1, size=(myWASN.wasn[ii].data.shape[0], nRandNoiseSigs)
        )
        # Add to sensor signals
        myWASN.wasn[ii].data = np.concatenate(
            (myWASN.wasn[ii].data, sig), axis=1
        )
        # Add to clean speech
        myWASN.wasn[ii].cleanspeech = np.concatenate(
            (myWASN.wasn[ii].cleanspeech, sig), axis=1
        )
        # Add to clean noise
        myWASN.wasn[ii].cleannoise = np.concatenate(
            (myWASN.wasn[ii].cleannoise, sig), axis=1
        )
        # Update number of sensors
        myWASN.wasn[ii].nSensors += nRandNoiseSigs

    return myWASN


def apply_self_noise(sig, snr):
    """Apply spectrally white self-noise to a sensor signal."""
    sn = np.random.uniform(-1, 1, size=sig.shape)  # white noise
    sig, sn = apply_noise(sig, sn, snr)
    return sig, sn


def apply_noise(sig, noise, snr):
    """
    Applies a noise `noise` to a signal `sig` so that it has
    a given target SNR `snr`.
    """
    Pn = np.mean(np.abs(noise) ** 2)
    Ps = np.mean(np.abs(sig) ** 2)
    currSNR = 10 * np.log10(Ps / Pn)
    noise *= 10 ** (-(snr - currSNR) / 20)
    sig += noise
    return sig, noise


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
        NB: can be `None` if the scenario does not represent a true room.
    roomDim : [3 x 1] list or np.ndarray (float)
        Room dimensions [x, y, z] (for plotting if 
        `topoParams.plotTopo == True`).
        NB: can be `None` if the scenario does not represent a true room.

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
    if sensorCoords is not None:
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
        if sensorCoords is None:
            # Create a random topology
            topo = np.random.randint(0, 2, size=(numNodes, numNodes))
            # Make it symmetrical
            topo = np.triu(topo) + np.triu(topo, 1).T
            # Make it boolean
            topo = topo.astype(bool)  # TODO: TEST!
        else:
            # Define the topology based on the maximum communication distance
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

    # Truncate or pad
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


def random_points_on_line(
        az,
        el,
        minDistToWalls,
        maxR,
        rd,
        xOffset=0,
        yOffset=0,
        n=1,
        minSpacing=0
    ):
    """
    Generate a random point on a line.
    
    Parameters
    ----------
    az : float
        Azimuth angle of the line.
    el : float
        Elevation angle of the line.
    minDistToWalls : float
        Minimum distance to the walls [m].
    maxR : float
        Maximum distance to the origin.
    rd : [3 x 1] list (or np.ndarray) (float)l
        Room dimensions ([x, y, z]) [m].
    xOffset : float, optional
        Offset on the x axis.
    yOffset : float, optional
        Offset on the y axis.
    n : int, optional
        Number of points to generate.
    minSpacing : float, optional
        Minimum spacing between points [m].

    Returns
    -------
    x : float
        x coordinate of the point.
    y : float
        y coordinate of the point.
    z : float
        z coordinate of the point.
    """
    def get_rand_point():
        r = np.random.uniform(
            minDistToWalls,
            maxR - minDistToWalls
        )
        x = r * np.cos(az) * np.sin(el)
        y = r * np.sin(az) * np.sin(el)
        x += xOffset
        y += yOffset
        z = r * np.cos(el)
        return x, y, z
    
    maxIter = 1000 # HARDCODED

    allPoints = np.zeros((n, 3))
    for ii in range(n):
        # Ensure the points are in the room
        cond1, cond2, cond3 = True, True, True
        x, y, z = -1, -1, -1
        iterCount = 0
        while cond1 or cond2 or cond3:
            x, y, z = get_rand_point()
            # Update conditions
            cond1 = x > rd[0] - minDistToWalls or x < minDistToWalls
            cond2 = y > rd[1] - minDistToWalls or y < minDistToWalls
            cond3 = z > rd[2] - minDistToWalls or z < minDistToWalls
            iterCount += 1
            if iterCount > maxIter:
                return None
        # Ensure the points are spaced enough
        if ii > 0 and minSpacing > 0:
            while np.any(np.linalg.norm(allPoints[:ii, :] - np.array([x, y, z]), axis=1) < minSpacing):
                x, y, z = get_rand_point()
                # Update conditions
                cond1 = x > rd[0] - minDistToWalls or x < minDistToWalls
                cond2 = y > rd[1] - minDistToWalls or y < minDistToWalls
                cond3 = z > rd[2] - minDistToWalls or z < minDistToWalls
                iterCount += 1
                if iterCount > maxIter:
                    return None
        allPoints[ii, :] = np.array([x, y, z])
    return allPoints


def array_id(
        a: np.ndarray, *,
        include_dtype=False,
        include_shape=False,
        algo = 'xxhash'
    ):
    """
    Computes a unique ID for a numpy array.
    From: https://stackoverflow.com/a/64756069

    Parameters
    ----------
    a : np.ndarray
        The array to compute the ID for.
    include_dtype : bool, optional
        Whether to include the dtype in the ID.
    include_shape : bool, optional
    """
    data = bytes()
    if include_dtype:
        data += str(a.dtype).encode('ascii')
    data += b','
    if include_shape:
        data += str(a.shape).encode('ascii')
    data += b','
    data += a.tobytes()
    if algo == 'sha256':
        import hashlib
        return hashlib.sha256(data).hexdigest().upper()
    elif algo == 'xxhash':
        import xxhash
        return xxhash.xxh3_64(data).hexdigest().upper()
    else:
        assert False, algo


def dot_to_p(x):
    """
    Converts a float to a string with a 'p' instead of a '.'.
    """
    return str(x).replace('.', 'p')


def load_layout_from_yaml(pathToFile):
    """
    Loads an acoustic scenario layout (sensors + sources coordinates) from a 
    YAML file.
    """
    
    with open(pathToFile, 'r') as f:
        d = yaml.load(f, Loader=yaml.FullLoader)

    return d