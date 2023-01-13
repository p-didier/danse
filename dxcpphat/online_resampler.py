import numpy as np
from scipy.signal.windows import hann

class OnlineResampler:

    '''
    Compensates SRO & STO in online manner.
    NOTICE: Due to the buffer mechanism, process() always yields the synchronized block
    from 2 iteratios earlier. Hence, the synced signal has a latency of two blockSizes
    w.r.t. to the asynchronous signal.
    
    Latency results from (a) the 50% overlap in overlap-add procedure (fft is applied to 2 blocks) and
    (b) the requirement to access [at least] one additional future block for compensating positive shift

    Note: STO compensation not yet tested.
    Note: signalBlock length corresponds to DXCPP fft shift (usually 2**11)
    '''

    def __init__(self, blockSize=2**11):
        self.blockSize = blockSize
        self.fftSize = blockSize*4 # 2*blockSize will be transformed, 2x interpolated fft
        self.shift = 0
        self.k = np.fft.fftshift(np.arange(-self.fftSize/2, self.fftSize/2))
        self.win  = hann(blockSize*2, sym=False)
        self.inputBuffer = np.zeros((blockSize*4,)) # prev - current - next - next2
        self.outputBuffer = np.zeros((blockSize*3,)) # output(overlap-added) - lastSigBlock - zero
        self.ell = 0
        self.overflowWarned = 'None' #'None'/'Left'/'Right'


    def process(self, signalBlock, sro, sto=0):
        '''
        Process signalBlock and return synchronized block from 2 iterations earlier.
        Input:
            sro (scalar): current SRO in ppm
            sto (scalar): current STO in smp 
        Output:
            Synchronized signal block from 2 iterations earlier.
        '''
        # Insert frame in buffer
        self.inputBuffer[:(3*self.blockSize)] = self.inputBuffer[self.blockSize:]
        self.inputBuffer[3*self.blockSize:] = signalBlock
        # Update accumulated shift, separate
        self.shift += sro*1e-6 * self.blockSize
        accShift = self.shift + sto
        integer_shift = np.round(accShift)
        rest_shift = integer_shift - accShift
        # Draw output from buffer: Range from start of second block to end of third, compensated by int shift.
        selectStart = int(self.blockSize + integer_shift)
        selectEnd = int((self.blockSize+2*self.blockSize) + integer_shift)
        # Correct indices in case of overflow
        if selectStart < 0:
            if self.overflowWarned != 'Left':
                print('Warning: Negative shift too large, cannot compensate fully.', str(selectStart))
                self.overflowWarned = 'Left'
            self.shift -= sro*1e-6 * self.blockSize # undo
            selectEnd = selectEnd - selectStart
            selectStart = 0
        elif selectEnd >= np.size(self.inputBuffer):
            if self.overflowWarned != 'Right':
                print('Warning: Positive shift too large, cannot compensate fully.', str(selectEnd-np.size(self.inputBuffer)))
                self.overflowWarned = 'Right'
            self.shift -= sro*1e-6 * self.blockSize # undo
            selectStart = selectStart - (selectEnd-np.size(self.inputBuffer))
            selectEnd = np.size(self.inputBuffer)
        else:
            self.overflowWarned = 'None'
        selectedBlocks = self.inputBuffer[selectStart:selectEnd]
        # Compensate rest shift via phase shift (fft 2x interpolation)
        selectedBlocks_fft = np.fft.fft(self.win * selectedBlocks, self.fftSize) #fft incl. interpol.
        selectedBlocks_fft *= np.exp(-1j * 2 * np.pi * self.k / self.fftSize * rest_shift)
        # Overlap add (into 2nd and 3rd frame of output buffer)
        self.outputBuffer[self.blockSize:] = self.outputBuffer[self.blockSize:] \
                                            + np.real(np.fft.ifft(selectedBlocks_fft))[:int(self.blockSize*2)]
        self.outputBuffer[:2*self.blockSize] = self.outputBuffer[self.blockSize:]
        self.outputBuffer[2*self.blockSize:] = np.zeros((self.blockSize,))
        return self.outputBuffer[:self.blockSize]
