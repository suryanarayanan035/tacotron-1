from concurrent.futures import ProcessPoolExecutor
from functools import partial 
import numpy as np
import os 
from util import audio 

def build_from_path(in_dir,out_dir,num_workers=1,tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1

    with open(os.path.join(in_dir,'dev.tsv'),encoding='utf-8') as f:
        for line in f:
            parts=line.strip().split('\t')
            wav_path = os.path.join(in_dir,'clips','%s'  % parts[1])
            text=parts[2]
            futures.append(executor.submit(partial(_process_utterance,out_dir,index,wav_path,text)))
            index+=1
        return [future.result() for future in tqdm(futures)]
def _process_utterance(out_dir,index,wav_path,text):
    wav = audio.load_wav(wav_path)
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    spectrogram_filename = 'mozilla-spec-%05d.npy'% index
    mel_filename = 'mozilla-mel-%05d.npy'%index
    np.save(os.path.join(out_dir,spectrogram_filename),spectrogram.T,allow_pickle=False)
    np.save(os.path.join(out_dir,mel_filename),spectrogram.T,allow_pickle=False)

    return (spectrogram_filename,mel_filename,n_frames,text)