from collections.abc import Iterable
from paderbox.array import segment_axis
from paderbox.io import load_audio

class AudioReader:

    def __init__(self,
                 block_length=None,
                 block_shift=None,
                 node_ids=None,
                 mic_ids=None,
                 data_root='data/'
                 ): 
        self.block_length = block_length
        self.block_shift = block_shift
        self.node_ids = node_ids
        self.mic_ids = mic_ids
        self.data_root = data_root

    def __call__(self, example):
        audio_paths = example['audio_paths']
        mic_sigs = dict()
        if self.node_ids is not None:
            node_ids = self.node_ids
        else:
            node_ids = audio_paths.keys()
        for node_id in node_ids:
            mic_sigs[node_id] = dict()
            if self.mic_ids is not None:
                assert (isinstance(self.mic_ids, dict)
                        or isinstance(self.mic_ids, str)
                        or isinstance(self.mic_ids, Iterable))
                if isinstance(self.mic_ids, dict):
                    mic_ids = self.mic_ids[node_id]
                elif isinstance(self.mic_ids, str):
                    mic_ids = [self.mic_ids]
                elif isinstance(self.mic_ids, Iterable):
                    mic_ids = self.mic_ids
            else:
                mic_ids = audio_paths[node_id].keys()
            for mic_id in mic_ids:
                path = self.data_root+audio_paths[node_id][mic_id]
                if self.block_length is None:
                    mic_sigs[node_id][mic_id] = load_audio(path, stop=2*60*16000)
                else:
                    mic_sigs[node_id][mic_id] = \
                        segment_axis(load_audio(path),  self.block_length,
                                     self.block_shift, end='cut')
        example['audio'] = mic_sigs
        return example
