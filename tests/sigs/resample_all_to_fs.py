# Purpose of script:
# Resample all signals to a common sampling rate.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import resampy
import soundfile as sf
from pathlib import Path

ROOT_FOLDER = Path(__file__).parent
COMMON_FS = 16000  # [Hz]

def main(commonFs=COMMON_FS):
    """Main function (called by default when running script)."""

    if not isinstance(commonFs, int):
        raise TypeError(f'`commonFs` must be an integer (current value: {commonFs} Hz).')
    
    # List all audio files in root folder
    audioFiles = list(ROOT_FOLDER.glob('**/*.wav')) +\
        list(ROOT_FOLDER.glob('**/*.flac'))
    
    # Resample all audio files to a common sampling rate
    for audioFile in audioFiles:
        print(f'Resampling {audioFile}...')
        audioData, fs = sf.read(audioFile)
        if fs != commonFs:
            audioData = resampy.resample(
                x=audioData,
                sr_orig=fs,
                sr_new=commonFs,
            )
            sf.write(
                file=f'{str(audioFile)[:-4]}_{int(commonFs)}Hz.wav',
                data=audioData,
                samplerate=commonFs,
            )
        print(f'Finished resampling {audioFile}.')
    
    stop = 1

if __name__ == '__main__':
    sys.exit(main())