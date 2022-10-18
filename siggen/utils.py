
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import resampy
from . import classes
from numba import njit
import numpy as np
import librosa
from scipy.spatial.transform import Rotation as rot
import pickle, gzip
import scipy.signal as sig

def build_room(p: classes.AcousticScenarioParameters):
    """
    Builds room, adds nodes and sources, simulates RIRs
    and computes microphone signals.
    """

    # Random generator
    rng = np.random.default_rng(p.seed)

    # Room
    room = pra.ShoeBox(
        p=p.rd,
        fs=p.fs,
        max_order=10,
        air_absorption=False,
        materials=pra.Material('rough_concrete')
    )

    for k in range(p.nNodes):
        # Generate node and sensors
        r = rng.uniform(size=(3,)) * (p.rd - 2 * p.minDistToWalls)\
            + p.minDistToWalls # node centre coordinates
        sensorsCoords = generate_array_pos(
                    r,
                    p.nSensorPerNode[k],
                    p.arrayGeometry,
                    p.interSensorDist,
                    rng
                    )
        room.add_microphone_array(sensorsCoords.T)

    # Add desired sources
    desiredSignalsRaw = np.zeros((int(p.sigDur * p.fs), p.nDesiredSources))
    for ii in range(p.nDesiredSources):
        # Load sound file
        y, fsOriginal = librosa.load(p.desiredSignalFile[ii])
        # Resample
        y = resampy.resample(y, fsOriginal, p.fs)
        # Truncate
        y = y[:int(p.sigDur * p.fs)]
        # Whiten
        y = (y - np.mean(y)) / np.std(y)  # whiten
        desiredSignalsRaw[:, ii] = y  # save (for VAD computation)
        ssrc = pra.soundsource.SoundSource(
            position=rng.uniform(size=(3,))* (p.rd - 2 * p.minDistToWalls)\
                + p.minDistToWalls, # source coordinates
            signal=y
        )
        room.add_soundsource(ssrc)

    # Add noise sources
    for ii in range(p.nNoiseSources):
        # Load sound file
        y, fsOriginal = librosa.load(p.noiseSignalFile[ii])
        # Resample
        y = resampy.resample(y, fsOriginal, p.fs)
        # Truncate
        y = y[:int(p.sigDur * p.fs)]
        # Whiten and apply gain
        y = (y - np.mean(y)) / np.std(y)    # whiten
        y *= 10 ** (-p.baseSNR / 20)        # gain
        ssrc = pra.soundsource.SoundSource(
            position=rng.uniform(size=(3,))* (p.rd - 2 * p.minDistToWalls)\
                + p.minDistToWalls, # source coordinates
            signal=y
        )
        room.add_soundsource(ssrc)

    room.compute_rir()
    room.simulate()

    # Extract 1 set of desired source RIRs per node
    rirsNodes = []
    for k in range(p.nNodes):
        rirsCurr = [room.rir[ii][:p.nDesiredSources] for ii in range(len(room.rir)) if p.sensorToNodeIndices[ii] == k]
        rirsNodes.append(rirsCurr[0])
    # Compute VAD
    vad = get_vad(rirsNodes, desiredSignalsRaw, p)

    return room, vad


def get_vad(rirs, xdry, p: classes.AcousticScenarioParameters):
    """
    Compute all node- and desired-source-specific VADs.
    """

    vad = np.zeros((xdry.shape[0], len(rirs), len(rirs[0])))
    for k in range(len(rirs)):  # for each node
        for ii in range(len(rirs[k])):  # for each desired source
            # Compute wet desired-only signal
            wetsig = sig.fftconvolve(xdry[:, ii], rirs[k][ii], axes=0)
            wetsig = wetsig[:xdry.shape[0]]  # truncate

            thrsVAD = np.amax(wetsig ** 2) / p.VADenergyFactor
            vad[:, k, ii], _ = oracleVAD(wetsig, tw=p.VADwinLength, thrs=thrsVAD, Fs=p.fs)

    return vad

def plot_mic_sigs(room: pra.room.ShoeBox, vad=None):
    """
    Quick plot of the microphone signals
    """

    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(8.5, 3.5)
    axes.plot(room.mic_array.signals.T)
    if vad is not None:
        axes.plot(vad[:, 0, 0] * np.amax(abs(room.mic_array.signals)), 'k', label='Oracle VAD')
    axes.grid()
    axes.legend()
    plt.tight_layout()
    plt.show()


def generate_array_pos(nodeCoords, Mk, arrayGeom, micSep, randGenerator, force2D=False):
    """Define node positions based on node position, number of nodes, and array type

    Parameters
    ----------
    nodeCoords : [J x 3] array of real floats.
        Nodes coordinates in 3-D space [m].
    TODO:
    randGenerator : NumPy random generator.
        Random generator with pre-specified seed.
    force2D : bool.
        If true, projects the sensor coordinates on the z=0 plane.

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
        rotvec = randGenerator.uniform(low=0, high=1, size=(3,))
        if force2D:
            rotvec[1:2] = 0
        sensorCoords = np.zeros_like(sensorCoordsBeforeRot)
        for ii in range(Mk):
            myrot = rot.from_rotvec(np.pi/2 * rotvec)
            sensorCoords[ii,:] = myrot.apply(sensorCoordsBeforeRot[ii, :]) + nodeCoords
    elif arrayGeom == 'radius':
        radius = micSep 
        sensorCoords = np.zeros((Mk,3))
        for ii in range(Mk):
            flag = False
            while not flag:
                r = randGenerator.uniform(low=0, high=radius, size=(3,))
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
                    coordFlat[counter, :] = [x[ii,jj,kk], y[ii,jj,kk], z[ii,jj,kk]]
                    counter += 1
        # Base configuration ("atomic" -- see Word journal week 39, MON)
        idx = [13,4,22,10,16,14,12]
        sensorCoords[:np.amin([len(idx), Mk]), :] = coordFlat[idx[:np.amin([len(idx), Mk])], :]
        if len(idx) < Mk:
            allIdx = np.arange(coordFlat.shape[0])
            idxValid = [ii for ii in allIdx if ii not in idx]
            for ii in range(Mk - len(idx)):
                sensorCoords[np.amin([len(idx), Mk]) + ii, :] = coordFlat[idxValid[ii], :]
    else:
        raise ValueError('No sensor array geometry defined for array type "%s"' % arrayGeom)

    return sensorCoords


def export_asc(room: pra.room.ShoeBox, path):
    """
    Export acoustic scenario (pra.room.ShoeBox object)
    as compressed archive.
    """
    pickle.dump(room, gzip.open(path, 'wb'))



def oracleVAD(x,tw,thrs,Fs):
    # oracleVAD -- Oracle Voice Activity Detection (VAD) function. Returns the
    # oracle VAD for a given speech (+ background noise) signal <x>.
    # Based on the computation of the short-time signal energy.
    #
    # >>> Inputs:
    # -x [N*1 float vector, -] - Time-domain signal.
    # -tw [float, s] - VAD window length.
    # -thrs [float, [<x>]^2] - Energy threshold.
    # -Fs [int, samples/s] - Sampling frequency.
    # >>> Outputs:
    # -oVAD [N*1 binary vector] - Oracle VAD corresponding to <x>.

    # (c) Paul Didier - 13-Sept-2021
    # SOUNDS ETN - KU Leuven ESAT STADIUS
    # ------------------------------------

    # Check input format
    x = np.array(x)     # Ensure it is an array
    if len(x.shape) > 1:
        print('<oracleVAD>: input signal is multidimensional: using 1st row as reference')
        dimsidx = range(len(x.shape))
        x = np.transpose(x, tuple(np.take(dimsidx,np.argsort(x.shape))))   # rearrange x dimensions in increasing order of size
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
    # JIT-ed time-domain VAD computation
    #
    # (c) Paul Didier - 6-Oct-2021
    # SOUNDS ETN - KU Leuven ESAT STADIUS
    # ------------------------------------
    # Compute short-term signal energy
    E = np.mean(np.abs(chunk_x)**2)
    # Assign VAD value
    if E > thrs:
        VADout = 1
    else:
        VADout = 0
    return VADout