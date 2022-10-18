import sys
import resampy
import numpy as np
from numba import njit
import soundfile as sf
import scipy.signal as sig
from dataclasses import dataclass
from pathlib import Path, PurePath
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}')
from danse.sandbox import TestParameters
import danse.testutils.dataclasses as met

@dataclass
class Signals:
    dryNoiseSources: np.ndarray = np.array([])          # Dry noise source signals
    drySpeechSources: np.ndarray = np.array([])         # Dry desired (speech) source signals
    wetIndivNoiseSources: np.ndarray = np.array([])     # Wet (convolved with RIRs) noise source signals, per indiv. noise source
    wetIndivSpeechSources: np.ndarray = np.array([])    # Wet (convolved with RIRs) desired source signals, per indiv. desired source
    wetNoise: np.ndarray = np.array([])                 # Wet (convolved with RIRs) noise source signals, all sources mixed
    wetSpeech: np.ndarray = np.array([])                # Wet (convolved with RIRs) desired source signals, all sources mixed
    sensorSignals: np.ndarray = np.array([])            # Sensor signals (all sources + RIRs)
    VAD: np.ndarray = np.array([])                      # Voice Activity Detector (1 = voice presence; 0 = noise only)
    sensorToNodeTags: np.ndarray = np.array([0])        # Tags relating each sensor to its node
    fs: np.ndarray = np.array([])                       # Sensor-specific sampling frequencies [samples/s]
    referenceSensor: int = 0                            # Index of the reference sensor at each node

@dataclass
class AcousticScenario:
    """
    Class for keeping track of acoustic scenario parameters
    [18.10.2022] -- Partial copy-paste from `AcousticScenario` in Paul Didier's `sounds-phd` repository
    """
    rirDesiredToSensors: np.ndarray = np.array([1])     # RIRs between desired sources and sensors
    rirNoiseToSensors: np.ndarray = np.array([1])       # RIRs between noise sources and sensors
    desiredSourceCoords: np.ndarray = np.array([1])     # Coordinates of desired sources
    sensorCoords: np.ndarray = np.array([1])            # Coordinates of sensors
    sensorToNodeTags: np.ndarray = np.array([1])        # Tags relating each sensor to its node
    noiseSourceCoords: np.ndarray = np.array([1])       # Coordinates of noise sources
    roomDimensions: np.ndarray = np.array([1])          # Room dimensions   
    absCoeff: float = 1.                                # Absorption coefficient
    samplingFreq: float = 16000.                        # Sampling frequency
    numNodes: int = 2                                   # Number of nodes in network
    distBtwSensors: float = 0.05                        # Distance btw. sensors at one node
    topology: str = 'fully_connected'                   # WASN topology type ("fully_connected" or ...TODO)

    def __post_init__(self):
        self.numDesiredSources = len(self.desiredSourceCoords)      # number of desired sources
        self.numSensors = len(self.sensorCoords)                    # number of sensors
        self.numNoiseSources = len(self.noiseSourceCoords)          # number of noise sources
        self.numSensorPerNode = np.unique(self.sensorToNodeTags, return_counts=True)[-1]    # number of sensors per node
        return self
    
    # Save and load
    def load(self, foldername: str):
        a: AcousticScenario = met.load(self, foldername)
        return a


def load_asc(pathToASC):
    """
    Loads an acoustic scenario from an exported ASC archive (.pkl.gz).
    """
    return AcousticScenario().load(pathToASC)


def generate_signals(asc: AcousticScenario, settings: TestParameters):
    """Generates signals based on acoustic scenario and raw files.
    Parameters
    ----------
    settings : ProgramSettings object
        The settings for the current run.

    Returns
    -------
    micSignals : [N x Nsensors] np.ndarray
        Sensor signals in time domain.
    SNRs : [Nsensors x 1] np.ndarray
        Pre-enhancement, raw sensor signals SNRs.
    asc : AcousticScenario object
        Processed data about acoustic scenario (RIRs, dimensions, etc.).
    """

    # Detect conflicts
    if asc.numDesiredSources > len(settings.desiredSignalFile):
        raise ValueError(f'{settings.desiredSignalFile} "desired" signal files provided while {asc.numDesiredSources} are needed.')
    if asc.numNoiseSources > len(settings.noiseSignalFile):
        raise ValueError(f'{settings.noiseSignalFile} "noise" signal files provided while {asc.numNoiseSources} are needed.')

    # Adapt sampling frequency
    if asc.samplingFreq != settings.fs:
        # Resample RIRs
        for ii in range(asc.rirDesiredToSensors.shape[1]):
            for jj in range(asc.rirDesiredToSensors.shape[2]):
                resampled = resampy.resample(asc.rirDesiredToSensors[:, ii, jj], asc.samplingFreq, settings.fs)
                if ii == 0 and jj == 0:
                    rirDesiredToSensors_resampled = np.zeros((resampled.shape[0], asc.rirDesiredToSensors.shape[1], asc.rirDesiredToSensors.shape[2]))
                rirDesiredToSensors_resampled[:, ii, jj] = resampled
            for jj in range(asc.rirNoiseToSensors.shape[2]):
                resampled = resampy.resample(asc.rirNoiseToSensors[:, ii, jj], asc.samplingFreq, settings.fs)
                if ii == 0 and jj == 0:
                    rirNoiseToSensors_resampled = np.zeros((resampled.shape[0], asc.rirNoiseToSensors.shape[1], asc.rirNoiseToSensors.shape[2]))
                rirNoiseToSensors_resampled[:, ii, jj] = resampled
        asc.rirDesiredToSensors = rirDesiredToSensors_resampled
        if asc.rirNoiseToSensors.shape[2] > 0:  # account for noiseless scenario
            asc.rirNoiseToSensors = rirNoiseToSensors_resampled
        # Modify ASC object parameter
        asc.samplingFreq = settings.fs

    # Desired signal length [samples]
    signalLength = int(settings.sigDur * asc.samplingFreq) 

    # Load + pre-process dry desired signals and build wet desired signals
    dryDesiredSignals = np.zeros((signalLength, asc.numDesiredSources))
    wetDesiredSignals = np.zeros((signalLength, asc.numDesiredSources, asc.numSensors))
    oVADsourceSpecific = np.zeros((signalLength, asc.numDesiredSources))
    for ii in range(asc.numDesiredSources):
        # Load signal
        rawSignal, fsRawSignal = sf.read(settings.desiredSignalFile[ii])

        # Pre-process (resample, truncate, whiten)
        dryDesiredSignals[:, ii] = pre_process_signal(rawSignal,
                                                    settings.sigDur,
                                                    fsRawSignal,
                                                    asc.samplingFreq)

        # Convolve with RIRs to create wet signals - TO GET THE VAD
        for jj in range(asc.numSensors):
            tmp = sig.fftconvolve(dryDesiredSignals[:, ii], asc.rirDesiredToSensors[:, jj, ii])
            wetDesiredSignals[:, ii, jj] = tmp[:signalLength]

        # Voice Activity Detection (pre-truncation/resampling)
        thrsVAD = np.amax(wetDesiredSignals[:, ii, 0] ** 2) / settings.VADenergyFactor
        oVADsourceSpecific[:, ii], _ = oracleVAD(wetDesiredSignals[:, ii, 0], settings.VADwinLength, thrsVAD, asc.samplingFreq)

        # Whiten dry signal 
        dryDesiredSignals[:, ii] = whiten(dryDesiredSignals[:, ii], oVADsourceSpecific[:, ii])

        # Convolve with RIRs to create wet signals - For actual use
        for jj in range(asc.numSensors):
            tmp = sig.fftconvolve(dryDesiredSignals[:, ii], asc.rirDesiredToSensors[:, jj, ii])
            wetDesiredSignals[:, ii, jj] = tmp[:signalLength]


    # Get VAD consensus
    oVADsourceSpecific = np.sum(oVADsourceSpecific, axis=1)
    oVAD = np.zeros_like(oVADsourceSpecific)
    oVAD[oVADsourceSpecific == asc.numDesiredSources] = 1   # only set global VAD = 1 when all sources are active

    # Load + pre-process dry noise signals and build wet noise signals
    dryNoiseSignals = np.zeros((signalLength, asc.numNoiseSources))
    wetNoiseSignals = np.zeros((signalLength, asc.numNoiseSources, asc.numSensors))
    for ii in range(asc.numNoiseSources):

        rawSignal, fsRawSignal = sf.read(settings.noiseSignalFile[ii])
        tmp = pre_process_signal(rawSignal,
                                    settings.sigDur,
                                    fsRawSignal,
                                    asc.samplingFreq)

        # Whiten signal 
        tmp = whiten(tmp, oVAD)

        # Set SNR
        dryNoiseSignals[:, ii] = 10 ** (-settings.baseSNR / 20) * tmp

        # Convolve with RIRs to create wet signals
        for jj in range(asc.numSensors):
            tmp = sig.fftconvolve(dryNoiseSignals[:, ii], asc.rirNoiseToSensors[:, jj, ii])
            wetNoiseSignals[:, ii, jj] = tmp[:signalLength]

    # Build speech-only and noise-only signals
    wetNoise = np.sum(wetNoiseSignals, axis=1)      # sum all noise sources at each sensor
    wetSpeech = np.sum(wetDesiredSignals, axis=1)   # sum all speech sources at each sensor
    wetNoise_norm = wetNoise / np.amax(np.abs(wetNoise + wetSpeech))    # Normalize
    wetSpeech_norm = wetSpeech / np.amax(np.abs(wetNoise + wetSpeech))  # Normalize

    # Build sensor signals
    sensorSignals = wetSpeech_norm + wetNoise_norm

    # Add self-noise to microphones
    rng = np.random.default_rng(settings.randSeed)
    for k in range(sensorSignals.shape[-1]):
        selfnoise = 10 ** (settings.selfnoiseSNR / 20) * np.amax(np.abs(sensorSignals[:, k])) * whiten(rng.uniform(-1, 1, (signalLength,)))
        sensorSignals[:, k] += selfnoise
    
    signals = Signals(
        dryNoiseSources=dryNoiseSignals,
        drySpeechSources=dryDesiredSignals,
        wetIndivNoiseSources=wetNoiseSignals,
        wetIndivSpeechSources=wetDesiredSignals,
        wetNoise=wetNoise_norm,
        wetSpeech=wetSpeech_norm,
        sensorSignals=sensorSignals,
        VAD=oVAD,
        sensorToNodeTags=asc.sensorToNodeTags,
        referenceSensor=settings.referenceSensor,
    )

    return signals


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


def oracleVAD(x, tw, thrs, Fs):
    """
    Oracle Voice Activity Detection (VAD) function. Returns the
    oracle VAD for a given speech (+ background noise) signal <x>.
    Based on the computation of the short-time signal energy.
    
    Parameters
    ----------
    x : [N*1] np.ndarray (float)
        Time-domain signal.
    tw : float
        VAD window length [s].
    thrs : float
        Energy threshold.
    Fs : float
        Sampling frequency [Hz].

    Returns
    -------
    oVAD : [N*1] np.ndarray (binary)
        Oracle VAD corresponding to `x`.

    (c) Paul Didier - 13-Sept-2021
    SOUNDS ETN - KU Leuven ESAT STADIUS
    ------------------------------------
    """

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



def pre_process_signal(rawSignal, desiredDuration, originalFs, targetFs):
    """Truncates/extends, resamples, centers, and scales a signal to match a target.
    Computes VAD estimate before whitening. 

    Parameters
    ----------
    rawSignal : [N_in x 1] np.ndarray
        Raw signal to be processed.
    desiredDuration : float
        Desired signal duration [s].
    originalFs : int
        Original raw signal sampling frequency [samples/s].
    targetFs : int
        Target sampling frequency [samples/s].
    VADenergyFactor : float or int
        VAD energy factor (VAD threshold = max(energy signal)/VADenergyFactor).
    VADwinLength : float
        VAD window duration [s].
    vadGiven : [N x 1] np.ndarray (binary float)
        Pre-computed VAD. If not `[]`, `VADenergyFactor` and `VADwinLength` arguments are ignored.

    Returns
    -------
    sig_out : [N_out x 1] np.ndarray
        Processed signal.
    """

    signalLength = int(desiredDuration * targetFs)   # desired signal length [samples]
    if originalFs != targetFs:
        # Resample signal so that its sampling frequency matches the target
        rawSignal = resampy.resample(rawSignal, originalFs, targetFs)
        # rawSignal = sig.resample(rawSignal, signalLength) 

    while len(rawSignal) < signalLength:
        sig_out = np.concatenate([rawSignal, rawSignal])             # extend too short signals
    if len(rawSignal) > signalLength:
        sig_out = rawSignal[:int(desiredDuration * targetFs)]  # truncate too long signals

    return sig_out



def whiten(sig, vad=[]):
    """
    Renders a sequence zero-mean and unit-variance.
    
    Parameters
    ----------
    sig : [N x 1] np.ndarray (real floats)
        Non-white input sequence.
    vad : [N x 1] np.ndarray (binary)
        Corresponding oracle Voice Activity Detector.

    Returns
    -------
    sig_out : [N x 1] np.ndarray
        Whitened input.
    """
    
    if vad == []:
        sig_out = (sig - np.mean(sig)) / np.std(sig)
    else:
        sig_out = (sig - np.mean(sig)) / np.std(sig[vad == 1])

    return sig_out