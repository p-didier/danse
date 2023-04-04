import numpy as np
from pesq import pesq
import scipy.signal as sig
from scipy.signal import stft
from dataclasses import dataclass, field
from danse_toolbox.mypystoi import stoi_any_fs as stoi_fcn


@dataclass
class EnhancementMeasures:
    """Class for storing speech enhancement metrics values"""
    snr: dict = field(default_factory=dict)         # Unweighted SNR
    fwSNRseg: dict = field(default_factory=dict)    # Frequency-weighted segmental SNR
    stoi: dict = field(default_factory=dict)        # Short-Time Objective Intelligibility
    pesq: dict = field(default_factory=dict)        # Perceptual Evaluation of Speech Quality
    startIdx: int = 0   # sample index at which the metrics were started to be computed

@dataclass
class Metric:
    """Class for storing objective speech enhancement metrics"""
    before: float = 0.          # metric value before enhancement
    after: float = 0.           # metric value after enhancement
    diff: float = 0.            # difference between before and after enhancement
    afterLocal: float = 0.      # metric value after _local_ enhancement (only using local sensors)
    diffLocal: float = 0.       # difference between before and after local enhancement
    afterCentr: float = 0.      # metric value after _centralized_ enhancement (only all sensors in network)
    diffCentr: float = 0.       # difference between before and after centralized enhancement
    dynamicFlag: bool = False   # True if a dynamic version of the metric was computed

    def import_dynamic_metric(self, dynObject):
        self.dynamicMetric = dynObject
        self.dynamicFlag = True     # set flag
        return self


class DynamicMetricsParameters:
    """Parameters for computing objective speech enhancement metrics
    dynamically as the signal comes in ("online" fashion)"""
    def __init__(self,
            chunkDuration=1.,
            chunkOverlap=0.5,
            dynamicSNR=False,
            dynamicSTOI=False,
            dynamicfwSNRseg=False,
            dynamicPESQ=False):
        self.chunkDuration = chunkDuration          # [s] duration of the signal chunk over which to compute the metrics
        self.chunkOverlap = chunkOverlap            # [/100%] percentage overlap between consecutive signal chunks
        # flags
        self.dynamicSNR = dynamicSNR                # if True, compute SNR dynamically
        self.dynamicSTOI = dynamicSTOI              # if True, compute STOI dynamically
        self.dynamicfwSNRseg = dynamicfwSNRseg      # if True, compute fwSNRseg dynamically
        self.dynamicPESQ = dynamicPESQ              # if True, compute PESQ dynamically

    def get_chunk_size(self, fs):
        self.chunkSize = int(np.floor(self.chunkDuration * fs))


class DynamicMetric:
    """Class for storing dynamic objective speech enhancement metrics"""
    def __init__(self, totalSigLength, N, ovlp):
        nChunks = int(np.floor(totalSigLength / (N * (1 - ovlp))))
        self.before = np.zeros(nChunks)         # dynamic metric value before enhancement
        self.after = np.zeros(nChunks)          # dynamic metric value after enhancement
        self.diff = np.zeros(nChunks)           # dynamic difference between before and after enhancement
        self.afterLocal = np.zeros(nChunks)     # dynamic metric value after _local_ enhancement (only using local sensors )
        self.diffLocal = np.zeros(nChunks)      # dynamic difference between before and after local enhancement
        self.afterCentr = 0.                    # dynamic metric value after _centralized_ enhancement (only all sensors in network)
        self.diffCentr = 0.                     # dynamic difference between before and after centralized enhancement
        self.timeStamps = np.zeros(nChunks)     # [s] true time stamps associated to each chunk 


def get_metrics(
        clean,
        noiseOnly,
        noisy,
        filteredSpeech,
        filteredNoise,
        filteredSpeech_c,
        filteredNoise_c,
        filteredSpeech_l,
        filteredNoise_l,
        enhan,
        enhan_c=None,
        enhan_l=None,
        fs=16e3,
        vad=None,
        dynamic: DynamicMetricsParameters=None,
        startIdx=0,
        gamma=0.2,
        fLen=0.03
    ):
    """
    Compute evaluation metrics for signal enhancement
    given a single-channel signal.

    Parameters
    ----------
    clean : [N x 1] np.ndarray (float)
        The clean, noise-free signal used as reference.
    noiseOnly : [N x 1] np.ndarray (float)
        The speech-free (noise only) signal captured at the reference mic.
    noisy : [N x 1] np.ndarray (float)
        The noisy signal (pre-signal enhancement).
    filteredSpeech : [N x 1] np.ndarray (float)
        The filtered speech-only signal (post-signal enhancement).
    filteredNoise : [N x 1] np.ndarray (float)
        The filtered noise-only signal (post-signal enhancement).
    filteredSpeech_c : [N x 1] np.ndarray (float)
        The filtered speech-only signal (after centralised processing).
    filteredNoise_c : [N x 1] np.ndarray (float)
        The filtered noise-only signal (after centralised processing).
    filteredSpeech_l : [N x 1] np.ndarray (float)
        The filtered speech-only signal (after local processing).
    filteredNoise_l : [N x 1] np.ndarray (float)
        The filtered noise-only signal (after local processing).
    enhan : [N x 1] np.ndarray (float)
        The enhanced signal (post-signal enhancement).
    enhan_c : [N x 1] np.ndarray (float)
        The enhanced signal (after centralised processing).
    enhan_l : [N x 1] np.ndarray (float)
        The enhanced signal (after local processing).
    fs : int
        Sampling frequency [samples/s].
    vad : [N x 1] np.ndarray (float)
        Voice Activity Detector (1: voice + noise; 0: noise only).
    dynamic : DynamicMetricsParameters object
        Parameters for dynamic metrics estimation
    startIdx : int
        Sample index to start at when computing the metrics.
    gamma : float
        Gamma exponent for fwSNRseg computation.
    fLen : float
        Time window duration for fwSNRseg computation [s].
    
    Returns
    -------
    snr : Metric object
        Unweighted signal-to-noise ratio (SNR).
    fwSNRseg : Metric object
        Frequency-weighted segmental SNR.
    myStoi : Metric object
        Short-Time Objective Intelligibility.
    myPesq : Metric object
        Perceptual Evaluation of Speech Quality.
    """

    # Init output arrays
    snr = Metric()
    fwSNRseg = Metric()
    myStoi = Metric()
    myPesq = Metric()

    # Trim to correct lengths (avoiding initial filter convergence
    # in calculation of metrics)
    clean_c = clean[startIdx:]
    clean_l = clean[startIdx:]
    clean = clean[startIdx:]
    # noiseOnly_c = noiseOnly[startIdx:]
    # noiseOnly_l = noiseOnly[startIdx:]
    noiseOnly = noiseOnly[startIdx:]
    if enhan_c is not None:
        enhan_c = enhan_c[startIdx:]
        vad_c = vad[startIdx:]
    if enhan_l is not None:
        enhan_l = enhan_l[startIdx:]
        vad_l = vad[startIdx:]
    enhan = enhan[startIdx:]
    noisy = noisy[startIdx:]
    vad = vad[startIdx:]
    filteredSpeech = filteredSpeech[startIdx:]
    filteredNoise = filteredNoise[startIdx:]
    if filteredSpeech_c is not None:
        filteredSpeech_c = filteredSpeech_c[startIdx:]
        filteredNoise_c = filteredNoise_c[startIdx:]
    if filteredSpeech_l is not None:
        filteredSpeech_l = filteredSpeech_l[startIdx:]
        filteredNoise_l = filteredNoise_l[startIdx:]
    
    # Unweighted SNR
    snr.before = get_snr(clean, noiseOnly, vad)
    snr.after = get_snr(filteredSpeech, filteredNoise, vad)
    snr.diff = snr.after - snr.before
    # Frequency-weight segmental SNR
    fwSNRseg_allFrames = get_fwsnrseg(
        clean, noisy, fs, fLen, gamma
    )
    fwSNRseg.before = np.mean(fwSNRseg_allFrames)
    fwSNRseg_allFrames = get_fwsnrseg(
        clean, enhan, fs, fLen, gamma
    )
    fwSNRseg.after = np.mean(fwSNRseg_allFrames)
    fwSNRseg.diff = fwSNRseg.after - fwSNRseg.before
    # Short-Time Objective Intelligibility (STOI)
    myStoi.before = stoi_fcn(clean, noisy, fs, extended=True)
    myStoi.after = stoi_fcn(clean, enhan, fs, extended=True)
    myStoi.diff = myStoi.after - myStoi.before
    # Perceptual Evaluation of Speech Quality (PESQ)
    if fs in [8e3, 16e3]:
        if fs == 8e3:
            mode = 'nb'  # narrowband PESQ
        elif fs == 16e3:
            mode = 'wb'  # wideband PESQ
            
        myPesq.before = pesq(fs, clean, noisy, mode)
        myPesq.after = pesq(fs, clean, enhan, mode)
        myPesq.diff = myPesq.after - myPesq.before
        if enhan_c is not None:
            myPesq.afterCentr = pesq(fs, clean_c, enhan_c, mode)
        if enhan_l is not None:
            myPesq.afterLocal = pesq(fs, clean_l, enhan_l, mode)
    else:
        print(f'Cannot calculate PESQ for fs different from 16 or 8 kHz (current value: {fs/1e3} kHz). Keeping `myPesq` attributes at 0.')

    # Consider centralised and local estimates
    if enhan_c is not None:
        snr.afterCentr = get_snr(filteredSpeech_c, filteredNoise_c, vad_c)
        fwSNRseg_allFrames = get_fwsnrseg(
            clean_c, enhan_c, fs, fLen, gamma
        )
        fwSNRseg.afterCentr = np.mean(fwSNRseg_allFrames)
        myStoi.afterCentr = stoi_fcn(clean_c, enhan_c, fs, extended=True)
    if enhan_l is not None:
        snr.afterLocal = get_snr(filteredSpeech_l, filteredNoise_l, vad_l)
        fwSNRseg_allFrames = get_fwsnrseg(
            clean_l, enhan_l, fs, fLen, gamma
        )
        fwSNRseg.afterLocal = np.mean(fwSNRseg_allFrames)
        myStoi.afterLocal = stoi_fcn(clean_l, enhan_l, fs, extended=True)

    # Compute dynamic metrics
    # TODO: go through this if needed + account for centralised
    # and local estimates.
    if dynamic is not None:
        dynFcns = []    # list function objects to be used
        if dynamic.dynamicSNR:
            dynFcns.append(get_snr)
        if dynamic.dynamicfwSNRseg:
            dynFcns.append(get_fwsnrseg)
        if dynamic.dynamicSTOI:
            dynFcns.append(stoi_fcn)
        if dynamic.dynamicPESQ:
            dynFcns.append(pesq)

        dynMetrics = get_dynamic_metric(
            dynFcns, clean, noisy, enhan, fs, vad, dynamic, gamma, fLen)

        # Store dynamic metrics
        for key, value in dynMetrics.items():
            if key == 'get_snr':
                snr.import_dynamic_metric(value)
            elif key == 'get_fwsnrseg':
                fwSNRseg.import_dynamic_metric(value)
            elif 'stoi' in key:
                myStoi.import_dynamic_metric(value)
            elif 'pesq' in key:
                myPesq.import_dynamic_metric(value)

    return snr, fwSNRseg, myStoi, myPesq


def get_dynamic_metric(
        fcns,
        clean, noisy, enhan,
        fs, VAD,
        dynamic: DynamicMetricsParameters,
        gamma=0.2, fLen=0.03):
    """
    Computes a given speech enhancement objective metric dynamically over
    a long signal, to observe the evolution of that metric over time.
    
    Parameters
    ----------
    fcn : function object or list of function objects
        Function(s) to use to compute the desired metric.
    __other parameters : see `get_metrics()`
    
    Returns
    -------
    dynObjects : dict of DynamicMetric objects
        Dynamic metric(s).
    """

    # Pre-process inputs
    if not isinstance(fcns, list):
        fcns = [fcns]   # make sure `fcns` is a list

    # Get useful values
    dynamic.get_chunk_size(fs)  # derive chunk size

    # Initialize dynamic metric objects
    dynObjs = dict(
        [(s.__name__, DynamicMetric(
            totalSigLength=clean.shape[0],
            N=dynamic.chunkSize,
            ovlp=dynamic.chunkOverlap
        )) for s in fcns]
    )

    c = 0
    while 1:    # loop over signal, chunk by chunk 
        i0 = int(c * dynamic.chunkSize * (1 - dynamic.chunkOverlap))
        i1 = int(i0 + dynamic.chunkSize)
        if i1 > clean.shape[0]:
            break   # end `while`-loop

        for idxFcn in range(len(fcns)):

            fcn = fcns[idxFcn]  # select current function object

            ref = fcn.__name__  # function reference (name)

            if ref == 'get_snr':           # Unweighted SNR
                dynObjs[ref].before[c] = fcn(noisy[i0:i1], VAD[i0:i1])
                dynObjs[ref].after[c] = fcn(enhan[i0:i1], VAD[i0:i1])
            elif ref == 'get_fwsnrseg':    # fwSNRseg
                tmp = fcn(clean[i0:i1], noisy[i0:i1], fs, fLen, gamma)
                dynObjs[ref].before[c] = np.mean(tmp)
                tmp = fcn(clean[i0:i1], enhan[i0:i1], fs, fLen, gamma)
                dynObjs[ref].after[c] = np.mean(tmp)
            elif 'stoi' in ref:            # STOI
                dynObjs[ref].before[c] = fcn(clean[i0:i1], noisy[i0:i1], fs)
                dynObjs[ref].after[c] = fcn(clean[i0:i1], enhan[i0:i1], fs)
            elif 'pesq' in ref:            # PESQ
                if fs in [8e3, 16e3]:
                    if fs == 8e3:
                        mode = 'nb'  # narrowband
                    elif fs == 16e3:
                        mode = 'wb'  # wideband
                        
                    try:
                        tmp = fcn(fs, clean[i0:i1], noisy[i0:i1], mode)
                    except RuntimeError:
                        print(f'Dynamic PESQ computation over {dynamic.chunkDuration}s chunks: "NoUtterancesError". Keeping `myPesq` dynamic attributes as 0.')
                    else:
                        dynObjs[ref].before[c] = fcn(
                            fs, clean[i0:i1], noisy[i0:i1], mode
                        )
                        dynObjs[ref].after[c] = fcn(
                            fs, clean[i0:i1], enhan[i0:i1], mode
                        )
                else:
                    print(f'Cannot calculate PESQ for fs != 16kHz or 8kHz (current value: {fs/1e3} kHz). Keeping `myPesq` attributes at 0.')
        c += 1     # increment chunk index

        # Emergency stop of `while`-loop
        if c > 2 * int(np.floor(
            clean.shape[0] / (dynamic.chunkSize * (1 - dynamic.chunkOverlap))
        )):
            raise ValueError(f'Seemingly infinite while-loop bug while computing dynamic metric for function "{fcn.__name__}".')

    for idxFcn in range(len(fcns)):
        ref = fcns[idxFcn].__name__  # function reference (name)
        # Compute differences
        dynObjs[ref].diff = dynObjs[ref].after - dynObjs[ref].before

        # Include time stamps
        timeShift = dynamic.chunkDuration * (1 - dynamic.chunkOverlap)
        dynObjs[ref].timeStamps = np.arange(
            start=timeShift,
            stop=clean.shape[0] / fs,
            step=timeShift
        )

    stop = 1

    return dynObjs


def get_sisnr(x, Fs, VAD):

    # Speech intelligibility indices (ANSI-S3.5-1997)
    Indices = 1e-4 * np.array([83.0, 95.0, 150.0, 289.0, 440.0, 578.0, 653.0,\
        711.0, 818.0, 844.0, 882.0, 898.0, 868.0, 844.0, 771.0, 527.0, 364.0,\
        185.0])   
    fc = np.array([160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0,\
        1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0,\
        6300.0, 8000.0])  # corresp. 1/3-octave centre freqs

    sisnr = 0
    for ii, fc_curr in enumerate(fc):

        # Filter in 1/3-octave bands
        Wn = 1/Fs * np.array([fc_curr*2**(-1/6), fc_curr*2**(1/6)])
        sos = sig.butter(
            10, Wn, btype='bandpass', analog=False, output='sos', fs=2*np.pi
        )
        x_filtered = sig.sosfilt(sos, x)

        # Build the SI-SNR sum
        sisnr += Indices[ii] * get_snr(x_filtered,VAD)

    return sisnr


def getSNR(timeDomainSignal, VAD):
    """
    Sub-function for `SNRest()`.
    """

    # Ensure correct input formats
    VAD = np.array(VAD)

    # Check input lengths
    if len(timeDomainSignal) < len(VAD):
        print('WARNING: VAD is longer than provided signal (possibly due to non-integer Tmax*Fs/(L-R)) --> truncating VAD')
        VAD = VAD[:len(timeDomainSignal)]

    # Number of time frames where VAD is active/inactive
    Ls = np.count_nonzero(VAD)
    Ln = len(VAD) - Ls

    if Ls > 0 and Ln > 0:
        noisePower = np.mean(np.power(timeDomainSignal[VAD == 0], 2))
        signalPower = np.mean(np.power(timeDomainSignal[VAD == 1], 2))
        speechPower = signalPower - noisePower
        if speechPower < 0:
            SNRout = -1 * float('inf')
        else:
            SNRout = 20 * np.log10(speechPower / noisePower)
    elif Ls == 0:
        SNRout = -1 * float('inf')
    elif Ln == 0:
        SNRout = float('inf')

    return SNRout


def get_snr(s: np.ndarray, n: np.ndarray, vad: np.ndarray):
    """
    Estimate SNR based on (filtered) speech and (filtered) noise (+ VAD).

    Parameters
    ----------
    s : [Nt x Nchannels] np.ndarray[float]
        Time-domain (filtered) speech signal (no noise).
    n : [Nt x Nchannels] np.ndarray[float]
        Time-domain (filtered) noise signal (no speech).
    vad : [Nt x Nchannels] np.ndarray[bool or int (1 or 0) or float (1. or 0.)]
        Corresponding voice activity detector (VAD).
    
    Returns
    -------
    snrEst : [Nchannels x 1] np.ndarray[float]
        Signal-to-noise ratio estimate [dB].
    """
    # Check for single-channel case
    if s.ndim == 1:
        s = s[:, np.newaxis]
    if n.ndim == 1:
        n = n[:, np.newaxis]
    if vad.ndim == 1:
        vad = vad[:, np.newaxis]

    vad = vad.astype(bool)  # convert to boolean
    nChannels = s.shape[-1]

    snrEst = np.zeros(nChannels)
    for c in range(nChannels):
        snrEst[c] = 10 * np.log10(
            np.mean(np.abs(s[vad[:, c], c]) ** 2) /\
            np.mean(np.abs(n[vad[:, c], c]) ** 2)
        )
    
    return snrEst


def get_snr_old(Y: np.ndarray, VAD):
    """
    SNRest -- Estimate SNR from time-domain VAD.
    
    Parameters
    ----------
    Y : [Nt x 1] /or/ [Nf x J] np.ndarray (float)
        Time-domain signal(s) /or/ frames x channels.
    VAD : [Nt x 1] np.ndarray (bool or int [0 or 1])
        Voice activity detector output.
    
    Returns
    -------
    SNR : float
        Signal-to-Noise Ratio [dB].
    """

    # Input format check
    if len(Y.shape) == 2:
        if Y.shape[1] > Y.shape[0]:
            Y = Y.T
    elif len(Y.shape) == 1:
        Y = Y[:, np.newaxis] 

    nChannels = Y.shape[1]
    SNRy = np.zeros(nChannels)
    for ii in range(nChannels):
        SNRy[ii] = getSNR(Y[:, ii], VAD)

    return SNRy


# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# vvvv FROM PYSEPM PACKAGE vvvv  https://github.com/schmiph2/pysepm
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
def get_fwsnrseg(cleanSig, enhancedSig, fs, frameLen=0.03, overlap=0.75, gamma=0.2):
    """
    Extracted (and slightly adapted) from pysepm.qualityMeasures package
    See https://github.com/schmiph2/pysepm

    See inline comments for EDITS w.r.t. original implementation.
    """

    # Addition on 22.03.2023 -- Robustness to input array dimensions
    if cleanSig.ndim == 2 and enhancedSig.ndim == 1:
        cleanSig = cleanSig[:, 0]
    if cleanSig.ndim == 1 and enhancedSig.ndim == 2:
        enhancedSig = enhancedSig[:, 0]

    if cleanSig.shape!=enhancedSig.shape:
        raise ValueError('The two signals do not match!')
    eps=np.finfo(np.float64).eps
    cleanSig=cleanSig.astype(np.float64)+eps
    enhancedSig=enhancedSig.astype(np.float64)+eps
    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples
    max_freq    = fs/2 #maximum bandwidth
    num_crit    = 25# number of critical bands
    n_fft       = 2**np.ceil(np.log2(2*winlength))
    n_fftby2    = int(n_fft/2)

    cent_freq=np.zeros((num_crit,))
    bandwidth=np.zeros((num_crit,))

    cent_freq[0]  = 50.0000;   bandwidth[0]  = 70.0000;
    cent_freq[1]  = 120.000;   bandwidth[1]  = 70.0000;
    cent_freq[2]  = 190.000;   bandwidth[2]  = 70.0000;
    cent_freq[3]  = 260.000;   bandwidth[3]  = 70.0000;
    cent_freq[4]  = 330.000;   bandwidth[4]  = 70.0000;
    cent_freq[5]  = 400.000;   bandwidth[5]  = 70.0000;
    cent_freq[6]  = 470.000;   bandwidth[6]  = 70.0000;
    cent_freq[7]  = 540.000;   bandwidth[7]  = 77.3724;
    cent_freq[8]  = 617.372;   bandwidth[8]  = 86.0056;
    cent_freq[9] =  703.378;   bandwidth[9] =  95.3398;
    cent_freq[10] = 798.717;   bandwidth[10] = 105.411;
    cent_freq[11] = 904.128;   bandwidth[11] = 116.256;
    cent_freq[12] = 1020.38;   bandwidth[12] = 127.914;
    cent_freq[13] = 1148.30;   bandwidth[13] = 140.423;
    cent_freq[14] = 1288.72;   bandwidth[14] = 153.823;
    cent_freq[15] = 1442.54;   bandwidth[15] = 168.154;
    cent_freq[16] = 1610.70;   bandwidth[16] = 183.457;
    cent_freq[17] = 1794.16;   bandwidth[17] = 199.776;
    cent_freq[18] = 1993.93;   bandwidth[18] = 217.153;
    cent_freq[19] = 2211.08;   bandwidth[19] = 235.631;
    cent_freq[20] = 2446.71;   bandwidth[20] = 255.255;
    cent_freq[21] = 2701.97;   bandwidth[21] = 276.072;
    cent_freq[22] = 2978.04;   bandwidth[22] = 298.126;
    cent_freq[23] = 3276.17;   bandwidth[23] = 321.465;
    cent_freq[24] = 3597.63;   bandwidth[24] = 346.136;


    W=np.array([0.003,0.003,0.003,0.007,0.010,0.016,0.016,0.017,0.017,0.022,0.027,0.028,0.030,0.032,0.034,0.035,0.037,0.036,0.036,0.033,0.030,0.029,0.027,0.026,
    0.026])

    bw_min=bandwidth[0]
    min_factor = np.exp (-30.0 / (2.0 * 2.303));#      % -30 dB point of filter

    all_f0=np.zeros((num_crit,))
    crit_filter=np.zeros((num_crit,int(n_fftby2)))
    j = np.arange(0,n_fftby2)


    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby2)
        all_f0[i] = np.floor(f0);
        bw = (bandwidth[i] / max_freq) * (n_fftby2);
        norm_factor = np.log(bw_min) - np.log(bandwidth[i]);
        crit_filter[i,:] = np.exp (-11 *(((j - np.floor(f0))/bw)**2) + norm_factor)
        crit_filter[i,:] = crit_filter[i,:]*(crit_filter[i,:] > min_factor)

    num_frames = len(cleanSig)/skiprate-(winlength/skiprate)# number of frames
    start      = 1 # starting sample
    #window     = 0.5*(1 - cos(2*pi*(1:winlength).T/(winlength+1)));


    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    f,t,Zxx=stft(cleanSig[0:int(num_frames)*skiprate+int(winlength-skiprate)], fs=fs, window=hannWin, nperseg=winlength, noverlap=winlength-skiprate, nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=False)
    clean_spec=np.abs(Zxx)
    clean_spec=clean_spec[:-1,:]
    clean_spec=(clean_spec/clean_spec.sum(0))
    f,t,Zxx=stft(enhancedSig[0:int(num_frames)*skiprate+int(winlength-skiprate)], fs=fs, window=hannWin, nperseg=winlength, noverlap=winlength-skiprate, nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=False)
    enh_spec=np.abs(Zxx)
    enh_spec=enh_spec[:-1,:]
    enh_spec=(enh_spec/enh_spec.sum(0))

    clean_energy=(crit_filter.dot(clean_spec))
    processed_energy=(crit_filter.dot(enh_spec))
    error_energy=np.power(clean_energy-processed_energy,2)
    error_energy[error_energy<eps]=eps
    W_freq=np.power(clean_energy,gamma)
    SNRlog=10*np.log10((clean_energy**2)/error_energy)
    fwSNR=np.sum(W_freq*SNRlog,0)/np.sum(W_freq,0)
    distortion=fwSNR.copy()
    # distortion[distortion<-10]=-10    # ORIGINAL IMPLEMENTATION: -10 dB bound (see hu2008a / hansen1998a) 
    distortion[distortion<0]=0          # modification by Paul Didier (04/04/2022, 14h21) -- ensures absence of negative fwSNRseg values
    distortion[distortion>35]=35

    return distortion
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# ^^^^ FROM PYSEPM PACKAGE ^^^^  https://github.com/schmiph2/pysepm
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------



